// =============================================================================
// gpu_divides.h — "does q divide y?" without a single division
//
// Trial division only needs a yes/no, not the remainder. mod_big_by_u32/u64
// compute the remainder with Horner steps that each take two 64-bit `%`
// (k35: two 128-bit `%`). NVIDIA GPUs have no integer divide instruction, so
// every `%` is a long software routine, and the kernels were bound by them.
//
// These predicates run a Montgomery reduction over y instead, least
// significant digit first:
//
//     t   = acc + d_i
//     m   = (t mod B) * (-q^-1 mod B) mod B
//     acc = (t + m*q) / B                       (exact: t + m*q ≡ 0 mod B)
//
// which leaves acc ≡ y * B^-k (mod q) with 0 <= acc <= q + 1. B is a power of
// two and q is odd, so B is invertible mod q and q | y iff acc ≡ 0, i.e. iff
// acc is 0 or q. Each digit costs a few multiply-adds; -q^-1 mod B costs a
// handful of multiplies per prime (Newton iteration).
//
// y is the usual little-endian array of nlimbs uint64_t limbs. Even q is not
// Montgomery-friendly; q = 2 is answered from the low bit, any other even q
// falls back to the Horner remainder.
//
// The predicates are __host__ __device__ (plain functions under a host
// compiler) so the exact same code is unit-tested on the CPU.
// =============================================================================

#pragma once

#include <stdint.h>

#ifdef __CUDACC__
#define KF_HD __host__ __device__ __forceinline__
#else
#define KF_HD static inline
#endif

// -q^-1 mod 2^32 for odd q. x = q is correct to 3 bits; each step doubles it.
KF_HD uint32_t kf_neg_inv32(uint32_t q)
{
    uint32_t x = q;
    for (int i = 0; i < 4; i++)
        x *= 2u - q * x;
    return 0u - x;
}

// -q^-1 mod 2^64 for odd q.
KF_HD uint64_t kf_neg_inv64(uint64_t q)
{
    uint64_t x = q;
    for (int i = 0; i < 5; i++)
        x *= 2u - q * x;
    return 0u - x;
}

KF_HD uint64_t kf_mulhi64(uint64_t a, uint64_t b)
{
#ifdef __CUDA_ARCH__
    return __umul64hi(a, b);
#else
    return (uint64_t)(((unsigned __int128)a * b) >> 64);
#endif
}

// Remainder by Horner steps, for the rare even q. Same arithmetic as
// mod_big_by_u32, kept here so this header stands alone.
KF_HD uint64_t kf_horner_mod_u32(const uint64_t *limbs, int nlimbs, uint64_t q)
{
    uint64_t b32 = ((uint64_t)1 << 32) % q;
    uint64_t b64 = (b32 * b32) % q;
    uint64_t acc = 0;
    for (int i = nlimbs - 1; i >= 0; i--)
    {
        acc = (acc * b64) % q;
        acc += limbs[i] % q;
        if (acc >= q)
            acc -= q;
    }
    return acc;
}

// q | y for 1 < q < 2^32. Digits are 32 bits: t + m*q < 2^32 + q + 2 + (2^32-1)q
// < 2^64, so everything stays in uint64_t.
KF_HD bool divides_big_u32(const uint64_t *limbs, int nlimbs, uint32_t q)
{
    if ((q & 1u) == 0)
        return q == 2 ? (nlimbs == 0 || (limbs[0] & 1u) == 0)
                      : kf_horner_mod_u32(limbs, nlimbs, q) == 0;
    uint32_t qi = kf_neg_inv32(q);
    uint64_t acc = 0;
    for (int i = 0; i < nlimbs; i++)
    {
        uint64_t limb = limbs[i];
        uint64_t t = acc + (uint32_t)limb;
        uint32_t m = (uint32_t)t * qi;
        acc = (t + (uint64_t)m * q) >> 32;
        t = acc + (limb >> 32);
        m = (uint32_t)t * qi;
        acc = (t + (uint64_t)m * q) >> 32;
    }
    return acc == 0 || acc == q;
}

// q | y for 1 < q < 2^63 (k35's upper range). Digits are 64 bits; t and
// t + m*q are 128-bit quantities carried as (hi, lo) pairs.
KF_HD bool divides_big_u64(const uint64_t *limbs, int nlimbs, uint64_t q)
{
    if ((q & 1u) == 0)
    {
        if (q == 2)
            return nlimbs == 0 || (limbs[0] & 1u) == 0;
        // No caller passes an even q > 2 (they are primes); stay correct anyway.
        uint64_t r = 0;
        for (int i = nlimbs - 1; i >= 0; i--)
        {
            for (int s = 1; s >= 0; s--)
            {
                uint64_t d = (limbs[i] >> (32 * s)) & 0xFFFFFFFFu;
                // r < q < 2^63, so r * 2^32 + d needs 96 bits: reduce in halves.
                uint64_t hi = r >> 32, lo = (r << 32) | d;
                r = (uint64_t)((((unsigned __int128)hi << 64) | lo) % q);
            }
        }
        return r == 0;
    }
    uint64_t qi = kf_neg_inv64(q);
    uint64_t acc = 0; // <= q + 1 < 2^63 + 2
    for (int i = 0; i < nlimbs; i++)
    {
        uint64_t t_lo = acc + limbs[i];
        uint64_t t_hi = (t_lo < acc); // carry out of the 64-bit add
        uint64_t m = t_lo * qi;
        uint64_t mq_lo = m * q;
        uint64_t mq_hi = kf_mulhi64(m, q);
        uint64_t sum_lo = t_lo + mq_lo; // == 0 by construction of m
        acc = t_hi + mq_hi + (sum_lo < t_lo);
    }
    return acc == 0 || acc == q;
}

// Two primes in one pass over y: the chains for q0 and q1 are independent, so
// a thread has twice the work in flight between y loads (instruction-level
// parallelism) and reads each limb once for both. Same verdicts as two
// divides_big_u32 calls; even q (never a k32 prime) takes that route.
KF_HD void divides_big_u32_x2(const uint64_t *limbs, int nlimbs,
                              uint32_t q0, uint32_t q1, bool *d0, bool *d1)
{
    if (((q0 & q1) & 1u) == 0)
    {
        *d0 = divides_big_u32(limbs, nlimbs, q0);
        *d1 = divides_big_u32(limbs, nlimbs, q1);
        return;
    }
    uint32_t i0 = kf_neg_inv32(q0), i1 = kf_neg_inv32(q1);
    uint64_t a0 = 0, a1 = 0;
    for (int i = 0; i < nlimbs; i++)
    {
        uint64_t limb = limbs[i];
        uint32_t lo = (uint32_t)limb, hi = (uint32_t)(limb >> 32);
        uint64_t t0 = a0 + lo, t1 = a1 + lo;
        uint32_t m0 = (uint32_t)t0 * i0, m1 = (uint32_t)t1 * i1;
        a0 = (t0 + (uint64_t)m0 * q0) >> 32;
        a1 = (t1 + (uint64_t)m1 * q1) >> 32;
        t0 = a0 + hi;
        t1 = a1 + hi;
        m0 = (uint32_t)t0 * i0;
        m1 = (uint32_t)t1 * i1;
        a0 = (t0 + (uint64_t)m0 * q0) >> 32;
        a1 = (t1 + (uint64_t)m1 * q1) >> 32;
    }
    *d0 = (a0 == 0 || a0 == q0);
    *d1 = (a1 == 0 || a1 == q1);
}

// Four primes in one pass over y: divides_big_u32_x2 widened to four
// independent chains (KF_TD_ILP=4). d[k] answers q[k] | y.
KF_HD void divides_big_u32_x4(const uint64_t *limbs, int nlimbs,
                              const uint32_t q[4], bool d[4])
{
    if (((q[0] & q[1] & q[2] & q[3]) & 1u) == 0)
    {
        for (int k = 0; k < 4; k++)
            d[k] = divides_big_u32(limbs, nlimbs, q[k]);
        return;
    }
    uint32_t inv[4];
    uint64_t a[4] = {0, 0, 0, 0};
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int k = 0; k < 4; k++)
        inv[k] = kf_neg_inv32(q[k]);
    for (int i = 0; i < nlimbs; i++)
    {
        uint64_t limb = limbs[i];
        uint32_t digit[2] = {(uint32_t)limb, (uint32_t)(limb >> 32)};
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
        for (int h = 0; h < 2; h++)
        {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
            for (int k = 0; k < 4; k++)
            {
                uint64_t t = a[k] + digit[h];
                uint32_t m = (uint32_t)t * inv[k];
                a[k] = (t + (uint64_t)m * q[k]) >> 32;
            }
        }
    }
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int k = 0; k < 4; k++)
        d[k] = (a[k] == 0 || a[k] == q[k]);
}
