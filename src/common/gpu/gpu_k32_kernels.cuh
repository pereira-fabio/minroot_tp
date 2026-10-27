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
// of 32. The single-prime kernel stays available for comparison:
//     make -B gpu GPU_DEFS=-DKF_TD_ILP=1   # -B: make cannot see GPU_DEFS change
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

#if KF_TD_ILP == 2
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
