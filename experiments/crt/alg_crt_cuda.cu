// =============================================================================
// alg_crt_cuda.cu — GPU CRT exponentiation solver.
//
// Computes the inverse root x = c^b mod M (b = e^-1 mod (M-1)) by alpha-ary
// exponentiation. Each step c <- c^alpha mod M is done by the CRT. Two recon modes
// (--recon), selectable at run time from ONE build:
//
//   Phase ① Residue (always on GPU): per prime p_i, r_i = (c mod p_i)^alpha mod p_i.
//           crt_residue_kernel — one thread/prime; reuses gpu_common.h:mod_big_by_u32.
//
//   Phase ② Reconstruction  r = sum_i r_i*K_i mod prod, then mod M:
//     --recon host  (Option A): download r; host does sum_i r_i*K_i with GMP.
//     --recon gpu   (Option B): crt_mac_kernel does the multiply-accumulate on the
//                   device (one thread per output limb column, coalesced over a
//                   prime-major K table), download the L column sums, host assembles
//                   + reduces mod prod / mod M. Moves the ~alpha x-dominant MAC to the
//                   GPU; the final mod-prod reduction stays on the host (B-full = that
//                   reduction on-device too, is future work).
//
// Baseline is the CORRECT inverse c^b (mpz_powm), not the broken forward
// gpu_measure_powm_baseline. Single GPU for now; multi-GPU is a follow-up.
//
// CLI:
//   ./alg_crt_cuda [<challenge> <modulo> | --data DIR --size N]
//     --alpha A --exponent E --n-challenges N --recon {host|gpu} --json FILE
//
// UNTESTED off-cluster (no CUDA/GPU locally) — build + `result^e == challenge`
// gate on IRIS before trusting any number.
// =============================================================================

#include "gpu_common.h" // CUDA_CHECK, MAX_LIMBS, mod_big_by_u32, read_mpz_from_file

#include <gmp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include "../common/common.h" // verify_eth_root

// ---------------------------------------------------------------------------
// Local CRT context (mirrors crt_common.cpp but MPI-free, so it links into the
// GPU build). Deterministic — picks ~31-bit primes until prod > M^alpha.
// ---------------------------------------------------------------------------
struct CrtCtx
{
    uint32_t *primes; // p_0 .. p_{k-1}
    int k;
    mpz_t *K;   // K_i = M_i * P_i  (< prod)
    mpz_t prod; // prod_i p_i
    unsigned long alpha;
};

static void crt_ctx_build(CrtCtx *ctx, const mpz_t modulo, unsigned long alpha)
{
    ctx->alpha = alpha;
    mpz_t bound;
    mpz_init(bound);
    mpz_pow_ui(bound, modulo, alpha);
    int cap = 16;
    ctx->primes = (uint32_t *)malloc(cap * sizeof(uint32_t));
    ctx->k = 0;
    mpz_init_set_ui(ctx->prod, 1);
    mpz_t p;
    mpz_init_set_ui(p, 1u << 30);
    while (mpz_cmp(ctx->prod, bound) <= 0)
    {
        mpz_nextprime(p, p);
        if (mpz_cmp_ui(p, 0xFFFFFFFFUL) >= 0)
        {
            fprintf(stderr, "crt_ctx_build: exhausted 32-bit primes (alpha=%lu, "
                            "|M|=%zu bits); lower alpha.\n",
                    alpha, mpz_sizeinbase(modulo, 2));
            exit(1);
        }
        if (ctx->k == cap)
        {
            cap *= 2;
            ctx->primes = (uint32_t *)realloc(ctx->primes, cap * sizeof(uint32_t));
        }
        ctx->primes[ctx->k++] = (uint32_t)mpz_get_ui(p);
        mpz_mul_ui(ctx->prod, ctx->prod, mpz_get_ui(p));
    }
    mpz_clear(p);
    mpz_clear(bound);
    ctx->K = (mpz_t *)malloc((size_t)ctx->k * sizeof(mpz_t));
    mpz_t Pi, Mi, pm;
    mpz_inits(Pi, Mi, pm, NULL);
    for (int i = 0; i < ctx->k; i++)
    {
        mpz_divexact_ui(Pi, ctx->prod, ctx->primes[i]);
        mpz_set_ui(pm, ctx->primes[i]);
        mpz_invert(Mi, Pi, pm);
        mpz_init(ctx->K[i]);
        mpz_mul(ctx->K[i], Mi, Pi);
    }
    mpz_clears(Pi, Mi, pm, NULL);
}

static void crt_ctx_free(CrtCtx *ctx)
{
    for (int i = 0; i < ctx->k; i++)
        mpz_clear(ctx->K[i]);
    free(ctx->K);
    free(ctx->primes);
    mpz_clear(ctx->prod);
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------
__constant__ uint64_t d_c_limbs[MAX_LIMBS];
__constant__ int d_nlimbs;

__device__ static uint32_t powmod_u32(uint64_t base, uint64_t exp, uint64_t mod)
{
    uint64_t result = 1 % mod;
    base %= mod;
    while (exp)
    {
        if (exp & 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return (uint32_t)result;
}

// Phase ①: r_i = (c mod p_i)^alpha mod p_i, one thread per prime.
__global__ void crt_residue_kernel(const uint32_t *__restrict__ primes, int64_t n,
                                   unsigned long alpha, uint32_t *out_r)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int nlimbs = d_nlimbs;
    for (int64_t i = idx; i < n; i += stride)
    {
        uint64_t p = (uint64_t)primes[i];
        uint64_t ci = mod_big_by_u32(d_c_limbs, nlimbs, p);
        out_r[i] = powmod_u32(ci, (uint64_t)alpha, p);
    }
}

// Phase ② (Option B): column j of sum_i r_i*K_i = sum_i r_i * Kflat[i*L + j].
// Kflat is prime-major (row i = limbs of K_i), so threads j (adjacent columns) read
// contiguous memory at each i -> coalesced. Each column sum fits 128 bits for
// realistic k (<= ~2^110). Host does the cross-column carry + reduction mod prod.
__global__ void crt_mac_kernel(const uint32_t *__restrict__ r,
                               const uint64_t *__restrict__ Kflat,
                               int k, int L, uint64_t *colsum_lo, uint64_t *colsum_hi)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= L)
        return;
    unsigned __int128 acc = 0;
    for (int i = 0; i < k; i++)
        acc += (unsigned __int128)r[i] * Kflat[(size_t)i * L + j];
    colsum_lo[j] = (uint64_t)acc;
    colsum_hi[j] = (uint64_t)(acc >> 64);
}

// ---------------------------------------------------------------------------
// Device/host buffers for one modulus (built once per size)
// ---------------------------------------------------------------------------
struct CrtGpu
{
    int recon_gpu; // 0 = host (Option A), 1 = gpu MAC (Option B)
    int k, L;
    uint32_t *d_primes, *d_r, *h_r;
    uint64_t *d_Kflat, *d_colsum_lo, *d_colsum_hi, *h_colsum_lo, *h_colsum_hi;
};

static void crt_gpu_setup(CrtGpu *g, const CrtCtx *ctx, int nlimbs, int recon_gpu)
{
    g->recon_gpu = recon_gpu;
    g->k = ctx->k;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &nlimbs, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g->d_primes, (size_t)ctx->k * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&g->d_r, (size_t)ctx->k * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(g->d_primes, ctx->primes, (size_t)ctx->k * sizeof(uint32_t), cudaMemcpyHostToDevice));
    g->h_r = (uint32_t *)malloc((size_t)ctx->k * sizeof(uint32_t));

    g->d_Kflat = nullptr;
    g->d_colsum_lo = g->d_colsum_hi = nullptr;
    g->h_colsum_lo = g->h_colsum_hi = nullptr;
    g->L = 0;
    if (recon_gpu)
    {
        int L = (int)((mpz_sizeinbase(ctx->prod, 2) + 63) / 64);
        g->L = L;
        // Prime-major K table: row i = L little-endian limbs of K_i (< prod, so <= L limbs).
        uint64_t *h_Kflat = (uint64_t *)calloc((size_t)ctx->k * L, sizeof(uint64_t));
        if (!h_Kflat)
        {
            fprintf(stderr, "OOM K table (%.0f MB)\n", (double)ctx->k * L * 8 / 1e6);
            exit(1);
        }
        for (int i = 0; i < ctx->k; i++)
        {
            size_t cnt = 0;
            mpz_export(h_Kflat + (size_t)i * L, &cnt, -1, sizeof(uint64_t), 0, 0, ctx->K[i]);
            // limbs [cnt..L-1] stay 0 (calloc)
        }
        CUDA_CHECK(cudaMalloc(&g->d_Kflat, (size_t)ctx->k * L * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(g->d_Kflat, h_Kflat, (size_t)ctx->k * L * sizeof(uint64_t), cudaMemcpyHostToDevice));
        free(h_Kflat);
        CUDA_CHECK(cudaMalloc(&g->d_colsum_lo, (size_t)L * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&g->d_colsum_hi, (size_t)L * sizeof(uint64_t)));
        g->h_colsum_lo = (uint64_t *)malloc((size_t)L * sizeof(uint64_t));
        g->h_colsum_hi = (uint64_t *)malloc((size_t)L * sizeof(uint64_t));
    }
}

static void crt_gpu_free(CrtGpu *g)
{
    free(g->h_r);
    cudaFree(g->d_primes);
    cudaFree(g->d_r);
    if (g->recon_gpu)
    {
        cudaFree(g->d_Kflat);
        cudaFree(g->d_colsum_lo);
        cudaFree(g->d_colsum_hi);
        free(g->h_colsum_lo);
        free(g->h_colsum_hi);
    }
}

// ---------------------------------------------------------------------------
// One CRT step: c_inout <- c_inout^alpha mod M
// ---------------------------------------------------------------------------
static const int THREADS = 256, BLOCKS = 256;

static void crt_step(mpz_t c_inout, const CrtCtx *ctx, const mpz_t modulo, CrtGpu *g)
{
    // Upload c (little-endian uint64 limbs, matching mod_big_by_u32's Horner order).
    uint64_t climbs[MAX_LIMBS];
    memset(climbs, 0, sizeof(climbs));
    size_t cnt = 0;
    mpz_export(climbs, &cnt, -1, sizeof(uint64_t), 0, 0, c_inout);
    CUDA_CHECK(cudaMemcpyToSymbol(d_c_limbs, climbs, sizeof(climbs)));

    // Phase ①: residues (always GPU)
    crt_residue_kernel<<<BLOCKS, THREADS>>>(g->d_primes, ctx->k, ctx->alpha, g->d_r);
    CUDA_CHECK(cudaGetLastError());

    mpz_t acc;
    mpz_init_set_ui(acc, 0);
    if (!g->recon_gpu)
    {
        // Option A: download residues, host multiply-accumulate.
        CUDA_CHECK(cudaMemcpy(g->h_r, g->d_r, (size_t)ctx->k * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int i = 0; i < ctx->k; i++)
            if (g->h_r[i])
                mpz_addmul_ui(acc, ctx->K[i], (unsigned long)g->h_r[i]);
    }
    else
    {
        // Option B: GPU MAC -> column sums; host assembles + reduces.
        int mac_blocks = (g->L + THREADS - 1) / THREADS;
        crt_mac_kernel<<<mac_blocks, THREADS>>>(g->d_r, g->d_Kflat, ctx->k, g->L,
                                                g->d_colsum_lo, g->d_colsum_hi);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(g->h_colsum_lo, g->d_colsum_lo, (size_t)g->L * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(g->h_colsum_hi, g->d_colsum_hi, (size_t)g->L * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        // sum = import(colsum_lo) + import(colsum_hi) << 64
        mpz_t Slo, Shi;
        mpz_inits(Slo, Shi, NULL);
        mpz_import(Slo, (size_t)g->L, -1, sizeof(uint64_t), 0, 0, g->h_colsum_lo);
        mpz_import(Shi, (size_t)g->L, -1, sizeof(uint64_t), 0, 0, g->h_colsum_hi);
        mpz_mul_2exp(Shi, Shi, 64);
        mpz_add(acc, Slo, Shi);
        mpz_clears(Slo, Shi, NULL);
    }
    mpz_mod(acc, acc, ctx->prod); // = c^alpha exact
    mpz_mod(acc, acc, modulo);    // c^alpha mod M
    mpz_set(c_inout, acc);
    mpz_clear(acc);
}

// ---------------------------------------------------------------------------
// Full solve for one challenge: result = c^b mod M via alpha-ary exponentiation.
// Returns the timed compute (ms) of the alpha-ary loop.
// ---------------------------------------------------------------------------
static double crt_solve_gpu(mpz_t result, const mpz_t challenge, const mpz_t modulo,
                            const CrtCtx *ctx, unsigned long exponent, CrtGpu *g)
{
    unsigned long alpha = ctx->alpha;
    mpz_t b, phi, ebase;
    mpz_inits(b, phi, ebase, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(ebase, exponent);
    mpz_invert(b, ebase, phi);

    std::vector<unsigned long> digits;
    {
        mpz_t bb;
        mpz_init_set(bb, b);
        while (mpz_sgn(bb) > 0)
        {
            digits.push_back(mpz_tdiv_ui(bb, alpha));
            mpz_tdiv_q_ui(bb, bb, alpha);
        }
        mpz_clear(bb);
        if (digits.empty())
            digits.push_back(0);
    }

    mpz_t cmod;
    mpz_init(cmod);
    mpz_mod(cmod, challenge, modulo);
    mpz_t *gtab = (mpz_t *)malloc((size_t)alpha * sizeof(mpz_t));
    for (unsigned long d = 0; d < alpha; d++)
    {
        mpz_init(gtab[d]);
        if (d == 0)
            mpz_set_ui(gtab[d], 1);
        else
        {
            mpz_mul(gtab[d], gtab[d - 1], cmod);
            mpz_mod(gtab[d], gtab[d], modulo);
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    mpz_set(result, gtab[digits.back()]);
    for (int j = (int)digits.size() - 2; j >= 0; j--)
    {
        crt_step(result, ctx, modulo, g);
        mpz_mul(result, result, gtab[digits[j]]);
        mpz_mod(result, result, modulo);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    for (unsigned long d = 0; d < alpha; d++)
        mpz_clear(gtab[d]);
    free(gtab);
    mpz_clear(cmod);
    mpz_clears(b, phi, ebase, NULL);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Minimal self-contained JSON (schema matches analysis/crt_best_alpha.py).
// ---------------------------------------------------------------------------
static void write_json(const char *path, const char *recon, unsigned long exponent,
                       unsigned long alpha, size_t challenge_bits, double powm_baseline_ms,
                       const std::vector<double> &warm_total_ms)
{
    if (!path || !path[0])
        return;
    double sum = 0;
    for (double t : warm_total_ms)
        sum += t;
    double avg = warm_total_ms.empty() ? 0.0 : sum / warm_total_ms.size();
    double spd = (avg > 0) ? powm_baseline_ms / avg : 0.0;
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "cannot write %s\n", path);
        return;
    }
    fprintf(f, "{\n  \"metadata\": {\n");
    fprintf(f, "    \"algorithm\": \"alg_crt\",\n    \"method\": \"crt\",\n");
    fprintf(f, "    \"recon\": \"%s\",\n", recon);
    fprintf(f, "    \"exponent\": %lu,\n    \"alpha\": %lu,\n", exponent, alpha);
    fprintf(f, "    \"challenge_bits\": %zu,\n    \"n_gpus\": 1,\n", challenge_bits);
    fprintf(f, "    \"runs_warm\": %zu,\n", warm_total_ms.size());
    fprintf(f, "    \"powm_baseline_ms\": %.4f\n  },\n", powm_baseline_ms);
    fprintf(f, "  \"warm_avg\": {\n    \"total_ms\": %.4f,\n    \"speedup_total\": %.4f\n  },\n", avg, spd);
    fprintf(f, "  \"warm_runs\": {\n    \"total_ms\": [");
    for (size_t i = 0; i < warm_total_ms.size(); i++)
        fprintf(f, "%s%.4f", i ? ", " : "", warm_total_ms[i]);
    fprintf(f, "]\n  }\n}\n");
    fclose(f);
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    const char *challenge_file = nullptr, *modulo_file = nullptr;
    const char *data_dir = nullptr, *json_path = nullptr, *recon = "host";
    int size_bits = 0, n_challenges = 10, recon_gpu = 0;
    unsigned long exponent = 3, alpha = 16;

    int positional = 0;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--data") && i + 1 < argc)
            data_dir = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc)
            size_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha") && i + 1 < argc)
            alpha = (unsigned long)atol(argv[++i]);
        else if (!strcmp(argv[i], "--exponent") && i + 1 < argc)
            exponent = (unsigned long)atol(argv[++i]);
        else if (!strcmp(argv[i], "--n-challenges") && i + 1 < argc)
            n_challenges = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--recon") && i + 1 < argc)
            recon = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_path = argv[++i];
        else if (argv[i][0] != '-')
        {
            if (positional == 0)
                challenge_file = argv[i];
            else if (positional == 1)
                modulo_file = argv[i];
            positional++;
        }
        else
        {
            fprintf(stderr, "Unknown option '%s'\n", argv[i]);
            return 1;
        }
    }
    if (alpha < 2)
    {
        fprintf(stderr, "--alpha must be >= 2\n");
        return 1;
    }
    if (n_challenges < 1)
        n_challenges = 1;
    if (!strcmp(recon, "gpu"))
        recon_gpu = 1;
    else if (strcmp(recon, "host"))
    {
        fprintf(stderr, "--recon must be host|gpu\n");
        return 1;
    }

    static char mbuf[512], cbuf0[512];
    if (data_dir && size_bits > 0)
    {
        snprintf(mbuf, sizeof(mbuf), "%s/modulo_alg3_%d.txt", data_dir, size_bits);
        snprintf(cbuf0, sizeof(cbuf0), "%s/challenge_0_alg3_%d.txt", data_dir, size_bits);
        modulo_file = mbuf;
        challenge_file = cbuf0;
    }
    else if (!challenge_file || !modulo_file)
    {
        fprintf(stderr, "Usage: %s (<challenge> <modulo> | --data DIR --size N) "
                        "[--alpha A] [--exponent E] [--n-challenges N] [--recon host|gpu] [--json FILE]\n",
                argv[0]);
        return 1;
    }

    mpz_t modulo;
    mpz_init(modulo);
    read_mpz_from_file(modulo, modulo_file);
    size_t challenge_bits = mpz_sizeinbase(modulo, 2);
    int nlimbs = (int)challenge_bits / 64 + 2;
    if (nlimbs > MAX_LIMBS)
    {
        fprintf(stderr, "modulus too large for MAX_LIMBS=%d\n", MAX_LIMBS);
        return 1;
    }

    printf("=== alg_crt_cuda (GPU CRT, recon=%s) ===\n", recon);
    printf("Modulus: %zu bits | alpha: %lu | exponent: %lu\n", challenge_bits, alpha, exponent);

    CrtCtx ctx;
    crt_ctx_build(&ctx, modulo, alpha);
    CrtGpu gpu;
    crt_gpu_setup(&gpu, &ctx, nlimbs, recon_gpu);
    printf("CRT: %d primes | prod = %zu bits%s\n", ctx.k, mpz_sizeinbase(ctx.prod, 2),
           recon_gpu ? "" : " | host reconstruction");
    if (recon_gpu)
        printf("K table on GPU: %d x %d limbs (%.0f MB)\n",
               ctx.k, gpu.L, (double)ctx.k * gpu.L * 8 / 1e6);

    mpz_t b, phi, ebase;
    mpz_inits(b, phi, ebase, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(ebase, exponent);
    mpz_invert(b, ebase, phi);

    mpz_t result, challenge, challenge_mod;
    mpz_inits(result, challenge, challenge_mod, NULL);
    std::vector<double> warm_total_ms;
    double powm_baseline_ms = 0.0;
    bool gate_checked = false;

    for (int ci = 0; ci < n_challenges; ci++)
    {
        char cf[512];
        if (data_dir && size_bits > 0)
            snprintf(cf, sizeof(cf), "%s/challenge_%d_alg3_%d.txt", data_dir, ci, size_bits);
        else if (ci == 0)
            snprintf(cf, sizeof(cf), "%s", challenge_file);
        else
            break;
        FILE *fp = fopen(cf, "r");
        if (!fp)
        {
            if (ci == 0)
            {
                fprintf(stderr, "missing %s\n", cf);
                return 1;
            }
            break;
        }
        fclose(fp);
        read_mpz_from_file(challenge, cf);
        mpz_mod(challenge_mod, challenge, modulo);

        double ms = crt_solve_gpu(result, challenge, modulo, &ctx, exponent, &gpu);

        if (!gate_checked)
        {
            verify_eth_root(result, challenge_mod, modulo, exponent);
            mpz_t tmp;
            mpz_init(tmp);
            auto t0 = std::chrono::high_resolution_clock::now();
            mpz_powm(tmp, challenge, b, modulo);
            powm_baseline_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - t0)
                                   .count();
            mpz_clear(tmp);
            gate_checked = true;
        }
        if (ci > 0)
            warm_total_ms.push_back(ms);
        printf("  challenge %d: %.3f ms%s\n", ci, ms, ci == 0 ? " (cold, discarded)" : "");
    }

    double warm_sum = 0.0;
    for (double t : warm_total_ms)
        warm_sum += t;
    double warm_avg = warm_total_ms.empty() ? 0.0 : warm_sum / warm_total_ms.size();
    double speedup = (warm_avg > 0) ? powm_baseline_ms / warm_avg : 0.0;
    printf("recon=%s | powm_baseline_ms = %.4f | warm avg total_ms = %.4f | speedup_total = %.4f\n",
           recon, powm_baseline_ms, warm_avg, speedup);
    write_json(json_path, recon, exponent, alpha, challenge_bits, powm_baseline_ms, warm_total_ms);

    crt_gpu_free(&gpu);
    crt_ctx_free(&ctx);
    mpz_clears(b, phi, ebase, result, challenge, challenge_mod, modulo, NULL);
    return 0;
}