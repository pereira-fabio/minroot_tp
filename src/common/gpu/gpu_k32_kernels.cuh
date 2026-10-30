// =============================================================================
// gpu_k32_kernels.cuh — the k32 trial-division kernel and its constant memory
//
// Algorithm 2 (k32), Algorithm 3 and Algorithm 4 all sieve a candidate against
// the same ~203 M primes in (65536, 2^32]; they differ only in what they accept
// afterwards, on the host. The kernel is therefore identical in all of them and
// lives here instead of being copied into each solver. Algorithm 2 (k35) uses
// it as its phase-1 kernel over the same lower range.
//
// Each solver is a separate executable, so the __constant__ symbols below get
// one instance per binary — the usual one-definition concerns do not apply.
//
// Requires gpu_common.h (MAX_LIMBS, MAX_GPU_FACTORS, divides_big_u32).
// =============================================================================

#pragma once

#include "gpu_common.h"

#include <gmp.h>
#include <vector>

#include <cstdint>

// ---------------------------------------------------------------------------
// Constant memory: blinded value y (one limb array, updated per attempt)
// ---------------------------------------------------------------------------

__constant__ uint64_t d_y_limbs[MAX_LIMBS];
__constant__ int d_nlimbs;

// ---------------------------------------------------------------------------
// Trial-divide y (in constant memory) by all k32 primes.
//
// KF_TD_ILP = 2 (default) makes each thread test two primes per pass over y
// (divides_big_u32_x2): primes[i] and primes[i + stride], so loads stay
// coalesced. Measured on an RTX 2080 SUPER, k32 8192-bit table, factor_ms
// warm 64.1 -> 43.0 ms (cold 73.9 -> 50.0) against one prime per thread; the
// two independent chains hide the multiply latency for 38 registers instead
// of 32. The other widths stay available for comparison (KF_TD_ILP=4: four
// primes per thread, primes[i + k*stride] for k = 0..3):
//     make -B gpu GPU_DEFS=-DKF_TD_ILP=1   # -B: make cannot see GPU_DEFS change
//     make -B gpu GPU_DEFS=-DKF_TD_ILP=4
// ---------------------------------------------------------------------------

#ifndef KF_TD_ILP
#define KF_TD_ILP 2
#endif

__device__ __forceinline__ void k32_record_factor(uint64_t p, uint64_t *out_factors,
                                                  int *out_count)
{
    int slot = atomicAdd(out_count, 1);
    if (slot < MAX_GPU_FACTORS)
        out_factors[slot] = p;
}

__global__ void k32_trial_divide_kernel(
    const uint32_t *__restrict__ primes, int64_t nprimes,
    uint64_t *out_factors, int *out_count)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int nlimbs = d_nlimbs;

#if KF_TD_ILP == 4
    for (int64_t i = idx; i < nprimes; i += 4 * stride)
    {
        if (i + 3 * stride < nprimes)
        {
            uint32_t q[4];
            bool d[4];
#pragma unroll
            for (int k = 0; k < 4; k++)
                q[k] = primes[i + k * stride];
            divides_big_u32_x4(d_y_limbs, nlimbs, q, d);
#pragma unroll
            for (int k = 0; k < 4; k++)
                if (d[k])
                    k32_record_factor(q[k], out_factors, out_count);
        }
        else
        {
            // Tail: fewer than four of this thread's primes remain.
            for (int64_t j = i; j < nprimes && j < i + 4 * stride; j += stride)
                if (divides_big_u32(d_y_limbs, nlimbs, primes[j]))
                    k32_record_factor(primes[j], out_factors, out_count);
        }
    }
#elif KF_TD_ILP == 2
    for (int64_t i = idx; i < nprimes; i += 2 * stride)
    {
        uint32_t p0 = primes[i];
        int64_t j = i + stride;
        if (j < nprimes)
        {
            uint32_t p1 = primes[j];
            bool d0, d1;
            divides_big_u32_x2(d_y_limbs, nlimbs, p0, p1, &d0, &d1);
            if (d0)
                k32_record_factor(p0, out_factors, out_count);
            if (d1)
                k32_record_factor(p1, out_factors, out_count);
        }
        else if (divides_big_u32(d_y_limbs, nlimbs, p0))
        {
            k32_record_factor(p0, out_factors, out_count);
        }
    }
#else
    for (int64_t i = idx; i < nprimes; i += stride)
    {
        uint64_t p = (uint64_t)primes[i];
        if (divides_big_u32(d_y_limbs, nlimbs, (uint32_t)p))
            k32_record_factor(p, out_factors, out_count);
    }
#endif
}

// ---------------------------------------------------------------------------
// Grouped trial division (--td-group 4): each thread reduces y once modulo the
// product of 4 of its primes (divides_big_group) instead of once per prime,
// 5/8 of the multiplies; the extra multi-word carries take back most of the
// rest. RTX 2080 SUPER, k32 8192-bit sieve, warm factor_ms over 9 runs:
// 28.57 -> 26.57 ms (-7%), 2-run batch 28.84 -> 26.87. G = 8 (63 registers)
// measured 30.14, slower than the default, so only G = 4 is offered. Its primes are
// primes[i + k*stride], k = 0..G-1, as in KF_TD_ILP = 4, so loads stay
// coalesced. A separate kernel, so the default kernel's register count and
// occupancy do not change.
// ---------------------------------------------------------------------------

template <int G>
__global__ void k32_trial_divide_group_kernel(
    const uint32_t *__restrict__ primes, int64_t nprimes,
    uint64_t *out_factors, int *out_count)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int nlimbs = d_nlimbs;
    for (int64_t i = idx; i < nprimes; i += (int64_t)G * stride)
    {
        uint32_t q[G];
        bool d[G];
#pragma unroll
        for (int k = 0; k < G; k++)
        {
            int64_t j = i + k * stride;
            q[k] = j < nprimes ? primes[j] : 1u; // 1 pads the tail group
        }
        divides_big_group<G>(d_y_limbs, nlimbs, q, d);
#pragma unroll
        for (int k = 0; k < G; k++)
            if (d[k] && q[k] != 1u)
                k32_record_factor(q[k], out_factors, out_count);
    }
}

// Launch the trial-division kernel the flags select: grouped (g_td_group,
// gpu_recon.cuh) or the default per-prime kernel. Same arguments and grid.
static inline void k32_td_launch(dim3 grid, dim3 block, const uint32_t *primes,
                                 int64_t nprimes, uint64_t *out_factors, int *out_count)
{
    if (g_td_group == 4)
        k32_trial_divide_group_kernel<4><<<grid, block>>>(primes, nprimes, out_factors, out_count);
    else
        k32_trial_divide_kernel<<<grid, block>>>(primes, nprimes, out_factors, out_count);
}

// ---------------------------------------------------------------------------
// Cofactor sharing between kernel launches (--td-phases N, default 16).
//
// Each thread's work is proportional to the limb count of y, and a smooth y
// loses ~32 bits per factor found. So the trial division runs as N launch
// rounds over successive slices of each device's primes; between rounds the
// host reads the primes found so far on every device, divides them out of a
// working copy of y, and uploads the shorter y and limb count to every
// device. A smooth y then averages about 1/2 + 1/(2N) of its size over the
// scan. RTX 2080 SUPER, k32 8192-bit: warm factor_ms 43.0 -> 30.2 ms (N = 8)
// -> 28.8 ms (N = 16); cold 50.0 -> 28.9 -> 27.7 ms. Only a y that is
// actually smooth shrinks, so in a real attack this speeds up the successful
// candidate, not the search for one.
//
// Call after uploading y, clearing the output buffers and recording ev0 on
// every device. launch(g, ph) launches round ph's slice on device g (current
// device already set); ev1 is recorded after the last round. The output
// buffers accumulate across rounds, so the caller's collect step sees every
// find. `full_nlimbs` is restored on every device afterwards. Dev needs
// d_out_factors, d_out_count and ev1 (GpuDev or a solver's own struct).
// ---------------------------------------------------------------------------

template <class Dev, class Launch>
static void gpu_td_rounds(Dev *devs, int n_gpus, int phases, const mpz_t y,
                          int full_nlimbs, uint64_t *h_buf, Launch launch)
{
    if (phases < 1)
        phases = 1;
    mpz_t cur;
    mpz_init_set(cur, y);
    std::vector<int> seen(n_gpus, 0);
    for (int ph = 0; ph < phases; ph++)
    {
        if (ph > 0)
        {
            bool shrunk = false;
            for (int g = 0; g < n_gpus; g++)
            {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaDeviceSynchronize());
                int h_count = 0;
                CUDA_CHECK(cudaMemcpy(&h_count, devs[g].d_out_count, sizeof(int),
                                      cudaMemcpyDeviceToHost));
                if (h_count > MAX_GPU_FACTORS)
                    h_count = MAX_GPU_FACTORS;
                if (h_count <= seen[g])
                    continue;
                int fresh = h_count - seen[g];
                CUDA_CHECK(cudaMemcpy(h_buf, devs[g].d_out_factors + seen[g],
                                      (size_t)fresh * sizeof(uint64_t),
                                      cudaMemcpyDeviceToHost));
                for (int j = 0; j < fresh; j++)
                    while (mpz_divisible_ui_p(cur, (unsigned long)h_buf[j]))
                    {
                        mpz_divexact_ui(cur, cur, (unsigned long)h_buf[j]);
                        shrunk = true;
                    }
                seen[g] = h_count;
            }
            if (shrunk)
            {
                uint64_t limbs[MAX_LIMBS] = {};
                size_t cnt = 0;
                mpz_export(limbs, &cnt, -1, sizeof(uint64_t), 0, 0, cur);
                int n = (int)cnt;
                for (int g = 0; g < n_gpus; g++)
                {
                    CUDA_CHECK(cudaSetDevice(g));
                    CUDA_CHECK(cudaMemcpyToSymbol(d_y_limbs, limbs, sizeof(uint64_t) * MAX_LIMBS));
                    CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &n, sizeof(int)));
                }
            }
        }
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            launch(g, ph, phases);
            CUDA_CHECK(cudaGetLastError());
            if (ph == phases - 1)
                CUDA_CHECK(cudaEventRecord(devs[g].ev1));
        }
    }
    mpz_clear(cur);
    if (phases > 1)
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaEventSynchronize(devs[g].ev1));
            CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &full_nlimbs, sizeof(int)));
        }
}

// [a, b) of a slice of n primes for round ph of phases.
static inline void gpu_td_round_range(int64_t n, int ph, int phases, int64_t *a, int64_t *b)
{
    *a = n * ph / phases;
    *b = n * (ph + 1) / phases;
}
