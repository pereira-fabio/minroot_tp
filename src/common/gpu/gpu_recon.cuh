// =============================================================================
// gpu_recon.cuh — reconstruction product on the device, as a reduction tree
//
// With a warm cbrt cache, reconstruction is the product of n ~|M|-bit terms
// mod M (n ~ 800 for k32 at 24576 bits). The host computes it as a tree over
// its cores (kf_product_mod_tree); this computes the same tree on the GPU:
//
//   level 0      x_i -> x_i * R mod M            n blocks   (into Montgomery form)
//   levels 1..   pair (2i, 2i+1) -> one product   ceil(cnt/2) blocks per level
//   last         y -> y * R^-1 mod M              1 block    (out of Montgomery form)
//
// ceil(log2 n) + 2 kernel launches; every node is one thread block running
// kf_mont_mul_block (gpu_mont.h). An odd element at the end of a level is
// carried up unchanged. The result is the same residue the host computes.
//
// Chosen per run with `--recon auto|device|host`; the default is auto, which
// sends a product to the device only when kf_parallel_worth says it is big
// enough (n * limbs^2 >= 2e6). Measured on an RTX 2080 SUPER: at 2048 bits
// (66 terms, below the gate) the host was faster, 0.075 vs 0.18 ms; at 8192
// bits (268 and 528 terms, above it) the device was, 1.3x for k32 and 2.6x for
// k16. It replaces only the final product in kf_compute_cbrt_from_factors_omp;
// the per-factor cube roots, cache and exponents stay on the host. It falls
// back to the host tree when M is even or the per-block shared memory does not
// fit the device.
//
// Device 0 does the work. The caller's current device is restored afterwards.
// =============================================================================

#pragma once

#include "gpu_common.h"
#include "gpu_mont.h"
#include "factorization.h"

#include <gmp.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define GPU_RECON_THREADS 256

struct gpu_recon_sync
{
    __device__ void bar() { __syncthreads(); }
    __device__ int bar_or(int p) { return __syncthreads_or(p); }
};

// out[i] = in[i] * b_fixed * R^-1          when b_fixed != nullptr (grid n_in)
// out[i] = in[2i] * in[2i+1] * R^-1        otherwise (grid ceil(n_in/2));
//          an unpaired last element is copied through.
__global__ void gpu_recon_level_kernel(const uint32_t *in, int n_in, uint32_t *out,
                                       const uint32_t *b_fixed,
                                       const uint32_t *M, const uint32_t *Mp, int L)
{
    extern __shared__ uint64_t gpu_recon_smem[]; // uint64_t: 8-byte aligned
    int i = (int)blockIdx.x;
    const uint32_t *a, *b;
    if (b_fixed)
    {
        a = in + (size_t)i * L;
        b = b_fixed;
    }
    else
    {
        a = in + (size_t)(2 * i) * L;
        if (2 * i + 1 >= n_in)
        {
            for (int k = (int)threadIdx.x; k < L; k += (int)blockDim.x)
                out[(size_t)i * L + k] = a[k];
            return; // whole block takes this branch: no barrier is skipped
        }
        b = a + L;
    }
    gpu_recon_sync sync;
    kf_mont_mul_block(a, b, out + (size_t)i * L, M, Mp, L,
                      (uint32_t *)gpu_recon_smem, (int)threadIdx.x, (int)blockDim.x, sync);
}

// Per-modulus device state, built on first use and rebuilt when M changes.
struct gpu_recon_state
{
    bool ready = false;
    bool usable = false;
    mpz_t M;
    int L = 0;
    size_t smem = 0;
    uint32_t *d_M = nullptr, *d_Mp = nullptr, *d_R2 = nullptr, *d_one = nullptr;
    uint32_t *d_buf[2] = {nullptr, nullptr};
    uint32_t *h_buf = nullptr; // pinned staging for uploads
    int cap = 0;               // terms the buffers hold
};

static gpu_recon_state g_gpu_recon;

static void gpu_recon_upload(uint32_t *dst, const mpz_t v, int L)
{
    uint32_t *tmp = (uint32_t *)calloc((size_t)L, sizeof(uint32_t));
    size_t cnt = 0;
    if (mpz_sgn(v) != 0)
        mpz_export(tmp, &cnt, -1, sizeof(uint32_t), 0, 0, v);
    CUDA_CHECK(cudaMemcpy(dst, tmp, (size_t)L * sizeof(uint32_t), cudaMemcpyHostToDevice));
    free(tmp);
}

static void gpu_recon_release(gpu_recon_state *s)
{
    cudaFree(s->d_M);
    cudaFree(s->d_Mp);
    cudaFree(s->d_R2);
    cudaFree(s->d_one);
    cudaFree(s->d_buf[0]);
    cudaFree(s->d_buf[1]);
    cudaFreeHost(s->h_buf);
    s->d_M = s->d_Mp = s->d_R2 = s->d_one = nullptr;
    s->d_buf[0] = s->d_buf[1] = nullptr;
    s->h_buf = nullptr;
    s->cap = 0;
}

// Precompute -M^-1 mod R and R^2 mod M for this modulus. Untimed in practice:
// it runs once per modulus, on the first (cold) reconstruction.
static void gpu_recon_prepare(gpu_recon_state *s, const mpz_t modulo)
{
    if (s->ready && mpz_cmp(s->M, modulo) == 0)
        return;
    if (!s->ready)
        mpz_init(s->M);
    gpu_recon_release(s);
    mpz_set(s->M, modulo);
    s->ready = true;
    s->L = (int)((mpz_sizeinbase(modulo, 2) + 31) / 32);
    s->smem = kf_mont_smem_words(s->L) * sizeof(uint32_t) + sizeof(uint64_t);

    int optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    s->usable = mpz_odd_p(modulo) && s->smem <= (size_t)optin;
    if (!s->usable)
    {
        fprintf(stderr, "[recon] device tree unavailable (%s); using the host tree\n",
                mpz_odd_p(modulo) ? "shared memory per block too small" : "even modulus");
        return;
    }
    CUDA_CHECK(cudaFuncSetAttribute(gpu_recon_level_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)s->smem));

    mpz_t R, Mp, R2, one;
    mpz_inits(R, Mp, R2, one, NULL);
    mpz_setbit(R, (mp_bitcnt_t)32 * s->L);
    mpz_invert(Mp, modulo, R);
    mpz_sub(Mp, R, Mp); // -M^-1 mod R
    mpz_mul(R2, R, R);
    mpz_mod(R2, R2, modulo);
    mpz_set_ui(one, 1);

    size_t bytes = (size_t)s->L * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&s->d_M, bytes));
    CUDA_CHECK(cudaMalloc(&s->d_Mp, bytes));
    CUDA_CHECK(cudaMalloc(&s->d_R2, bytes));
    CUDA_CHECK(cudaMalloc(&s->d_one, bytes));
    gpu_recon_upload(s->d_M, modulo, s->L);
    gpu_recon_upload(s->d_Mp, Mp, s->L);
    gpu_recon_upload(s->d_R2, R2, s->L);
    gpu_recon_upload(s->d_one, one, s->L);
    mpz_clears(R, Mp, R2, one, NULL);
}

// kf_product_fn: result = vals[0] * ... * vals[n-1] mod M, all vals < M.
static void gpu_recon_product(mpz_t result, mpz_t *vals, int n,
                              const mpz_t modulo, int nthreads)
{
    if (n <= 1)
    {
        kf_product_mod_tree(result, vals, n, modulo, nthreads);
        return;
    }
    int prev_dev = 0;
    CUDA_CHECK(cudaGetDevice(&prev_dev));
    CUDA_CHECK(cudaSetDevice(0));

    gpu_recon_state *s = &g_gpu_recon;
    gpu_recon_prepare(s, modulo);
    if (!s->usable)
    {
        CUDA_CHECK(cudaSetDevice(prev_dev));
        kf_product_mod_tree(result, vals, n, modulo, nthreads);
        return;
    }

    const int L = s->L;
    const size_t term = (size_t)L * sizeof(uint32_t);
    if (n > s->cap)
    {
        cudaFree(s->d_buf[0]);
        cudaFree(s->d_buf[1]);
        cudaFreeHost(s->h_buf);
        int cap = s->cap ? s->cap : 256;
        while (cap < n)
            cap *= 2;
        CUDA_CHECK(cudaMalloc(&s->d_buf[0], cap * term));
        CUDA_CHECK(cudaMalloc(&s->d_buf[1], cap * term));
        CUDA_CHECK(cudaMallocHost(&s->h_buf, cap * term));
        s->cap = cap;
    }

    memset(s->h_buf, 0, n * term);
    for (int i = 0; i < n; i++)
    {
        size_t cnt = 0;
        if (mpz_sgn(vals[i]) != 0)
            mpz_export(s->h_buf + (size_t)i * L, &cnt, -1, sizeof(uint32_t), 0, 0, vals[i]);
    }
    CUDA_CHECK(cudaMemcpy(s->d_buf[0], s->h_buf, n * term, cudaMemcpyHostToDevice));

    // Into Montgomery form, then the tree, then back out.
    gpu_recon_level_kernel<<<n, GPU_RECON_THREADS, s->smem>>>(
        s->d_buf[0], n, s->d_buf[1], s->d_R2, s->d_M, s->d_Mp, L);
    CUDA_CHECK(cudaGetLastError());
    int cur = 1, cnt = n;
    while (cnt > 1)
    {
        int next = (cnt + 1) / 2;
        gpu_recon_level_kernel<<<next, GPU_RECON_THREADS, s->smem>>>(
            s->d_buf[cur], cnt, s->d_buf[cur ^ 1], nullptr, s->d_M, s->d_Mp, L);
        CUDA_CHECK(cudaGetLastError());
        cur ^= 1;
        cnt = next;
    }
    gpu_recon_level_kernel<<<1, GPU_RECON_THREADS, s->smem>>>(
        s->d_buf[cur], 1, s->d_buf[cur ^ 1], s->d_one, s->d_M, s->d_Mp, L);
    CUDA_CHECK(cudaGetLastError());
    cur ^= 1;

    CUDA_CHECK(cudaMemcpy(s->h_buf, s->d_buf[cur], term, cudaMemcpyDeviceToHost));
    mpz_import(result, L, -1, sizeof(uint32_t), 0, 0, s->h_buf);
    CUDA_CHECK(cudaSetDevice(prev_dev));
}

// --recon auto: the device for products big enough to pay for the launches.
static void gpu_recon_auto(mpz_t result, mpz_t *vals, int n,
                           const mpz_t modulo, int nthreads)
{
    if (kf_parallel_worth((double)n, modulo))
        gpu_recon_product(result, vals, n, modulo, nthreads);
    else
        kf_product_mod_tree(result, vals, n, modulo, nthreads);
}

// Strip `--recon auto|host|device` from argv (so the solver's own parser never
// sees it) and install the chosen product backend; auto when the flag is
// absent. Call first thing in main().
static inline void gpu_recon_parse_args(int *argc, char **argv)
{
    const char *mode = "auto";
    int w = 1;
    for (int i = 1; i < *argc; i++)
    {
        if (!strcmp(argv[i], "--recon") && i + 1 < *argc)
        {
            mode = argv[++i];
            continue;
        }
        argv[w++] = argv[i];
    }
    *argc = w;
    argv[w] = nullptr;

    if (!strcmp(mode, "auto"))
        kf_set_product_backend(gpu_recon_auto);
    else if (!strcmp(mode, "device"))
        kf_set_product_backend(gpu_recon_product);
    else if (!strcmp(mode, "host"))
        kf_set_product_backend(nullptr);
    else
    {
        fprintf(stderr, "Error: --recon must be auto, host or device, got '%s'\n", mode);
        exit(1);
    }
    printf("Reconstruction product: %s%s\n", mode,
           strcmp(mode, "auto") ? " tree" : " (device tree for large products, host otherwise)");
}
