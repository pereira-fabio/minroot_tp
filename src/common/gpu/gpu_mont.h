// =============================================================================
// gpu_mont.h — block-cooperative Montgomery multiplication of |M|-bit numbers
//
// One thread block computes  res = a * b * R^-1 mod M  for R = 2^(32L), where
// a, b, M are L-limb little-endian uint32_t arrays with a, b < M and M odd.
// This is the multiply that the device reconstruction tree (gpu_recon.cuh)
// applies at every node.
//
// Shape of the computation (all products are schoolbook, split by column):
//   T  = a * b                         2L limbs
//   m  = (T mod R) * (-M^-1 mod R)     low L limbs only
//   P  = m * M                         2L limbs
//   U  = (T + P) / R                   T + P ≡ 0 mod R, so the low half is all
//                                      zeros and its carry into the high half
//                                      is 1 exactly when T mod R != 0
//   res = U - M if U >= M else U       U < 2M
//
// Each thread owns the columns k ≡ tid (mod nthr). A column sum of up to L
// products of 32x32 bits is < L * 2^64, held as (lo64, hi32). Turning columns
// into limbs leaves per-limb carries of at most 2, resolved by rounds of
// "every limb absorbs its neighbour's carry" until none is left; for the
// random-looking values here that is a handful of rounds.
//
// The code is templated on a Sync policy (bar(), bar_or()) so the identical
// function runs as a CUDA block (__syncthreads) and, in the unit test, as host
// threads behind a std::barrier — which checks the arithmetic and the barrier
// placement against GMP without a GPU.
// =============================================================================

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#define KF_MONT_HD __host__ __device__
// The templates below call Sync::bar, which is __device__ for the CUDA policy
// and host-only for the test policy; each instantiation only runs on its side.
#define KF_MONT_NO_EXEC_CHECK _Pragma("nv_exec_check_disable")
#else
#define KF_MONT_HD
#define KF_MONT_NO_EXEC_CHECK
#endif

// Shared-memory layout, in uint32_t words, for limb count L.
struct kf_mont_smem
{
    uint64_t *col_lo; // 2L column sums, low 64 bits
    uint32_t *col_hi; // 2L column sums, bits 64..95
    uint32_t *sa, *sb, *sm, *smp; // a, b, M, -M^-1 mod R
    uint32_t *T, *P;  // 2L each
    uint32_t *mm;     // L
    uint32_t *U, *D;  // L + 2 each
    uint32_t *carry, *cin; // 2L + 2 each
};

KF_MONT_HD inline size_t kf_mont_smem_words(int L)
{
    // col_lo first so it is 8-byte aligned.
    return (size_t)4 * L + 2 * L + 4 * L + 4 * L + L + 2 * (L + 2) + 2 * (2 * L + 2);
}

KF_MONT_HD inline kf_mont_smem kf_mont_smem_carve(uint32_t *w, int L)
{
    kf_mont_smem s;
    s.col_lo = (uint64_t *)w;   w += 4 * L;
    s.col_hi = w;               w += 2 * L;
    s.sa = w;                   w += L;
    s.sb = w;                   w += L;
    s.sm = w;                   w += L;
    s.smp = w;                  w += L;
    s.T = w;                    w += 2 * L;
    s.P = w;                    w += 2 * L;
    s.mm = w;                   w += L;
    s.U = w;                    w += L + 2;
    s.D = w;                    w += L + 2;
    s.carry = w;                w += 2 * L + 2;
    s.cin = w;
    return s;
}

// Resolve pending carries: out[k] holds a 32-bit limb and carry[k] what it
// owes limb k+1. The carry out of limb K-1 is dropped (callers size K so that
// it is always zero, or want the value mod 2^(32K)).
KF_MONT_NO_EXEC_CHECK
template <class Sync>
KF_MONT_HD void kf_mont_ripple(uint32_t *out, uint32_t *carry, uint32_t *cin,
                               int K, int tid, int nthr, Sync &sync)
{
    for (;;)
    {
        int any = 0;
        for (int k = tid; k < K - 1; k += nthr)
            any |= (carry[k] != 0);
        if (!sync.bar_or(any))
            break;
        for (int k = tid; k < K; k += nthr)
            cin[k] = (k > 0) ? carry[k - 1] : 0;
        sync.bar();
        for (int k = tid; k < K; k += nthr)
        {
            uint64_t v = (uint64_t)out[k] + cin[k];
            out[k] = (uint32_t)v;
            carry[k] = (uint32_t)(v >> 32);
        }
    }
    sync.bar();
}

// Columns 0..K-1 of x * y, then normalised into K limbs of `out` (mod 2^(32K)).
KF_MONT_NO_EXEC_CHECK
template <class Sync>
KF_MONT_HD void kf_mont_mul_cols(const uint32_t *x, int Lx, const uint32_t *y, int Ly,
                                 int K, uint32_t *out, const kf_mont_smem &s,
                                 int tid, int nthr, Sync &sync)
{
    for (int k = tid; k < K; k += nthr)
    {
        uint64_t lo = 0;
        uint32_t hi = 0;
        int i0 = k - (Ly - 1) > 0 ? k - (Ly - 1) : 0;
        int i1 = k < Lx - 1 ? k : Lx - 1;
        for (int i = i0; i <= i1; i++)
        {
            uint64_t p = (uint64_t)x[i] * y[k - i];
            lo += p;
            hi += (lo < p);
        }
        s.col_lo[k] = lo;
        s.col_hi[k] = hi;
    }
    sync.bar();
    // Column k is lo + hi*2^64: its bits land in limbs k, k+1 and k+2.
    for (int k = tid; k < K; k += nthr)
    {
        uint64_t v = (uint32_t)s.col_lo[k];
        if (k >= 1)
            v += s.col_lo[k - 1] >> 32;
        if (k >= 2)
            v += s.col_hi[k - 2];
        out[k] = (uint32_t)v;
        s.carry[k] = (uint32_t)(v >> 32);
    }
    kf_mont_ripple(out, s.carry, s.cin, K, tid, nthr, sync);
}

// res = a * b * R^-1 mod M. a, b, M, Mp (= -M^-1 mod R) and res are L-limb
// arrays in any memory space; res must not alias a or b.
KF_MONT_NO_EXEC_CHECK
template <class Sync>
KF_MONT_HD void kf_mont_mul_block(const uint32_t *a, const uint32_t *b, uint32_t *res,
                                  const uint32_t *M, const uint32_t *Mp, int L,
                                  uint32_t *smem, int tid, int nthr, Sync &sync)
{
    kf_mont_smem s = kf_mont_smem_carve(smem, L);
    for (int k = tid; k < L; k += nthr)
    {
        s.sa[k] = a[k];
        s.sb[k] = b[k];
        s.sm[k] = M[k];
        s.smp[k] = Mp[k];
    }
    sync.bar();

    kf_mont_mul_cols(s.sa, L, s.sb, L, 2 * L, s.T, s, tid, nthr, sync);   // T
    kf_mont_mul_cols(s.T, L, s.smp, L, L, s.mm, s, tid, nthr, sync);      // m
    kf_mont_mul_cols(s.mm, L, s.sm, L, 2 * L, s.P, s, tid, nthr, sync);   // P

    int low_nonzero = 0;
    for (int k = tid; k < L; k += nthr)
        low_nonzero |= (s.T[k] != 0);
    uint32_t cin0 = sync.bar_or(low_nonzero) ? 1u : 0u;

    // U = T_hi + P_hi + cin0, over L+2 limbs (U < 2M, so the top two are spare).
    for (int k = tid; k < L + 2; k += nthr)
    {
        uint64_t v = (k < L) ? (uint64_t)s.T[L + k] + s.P[L + k] : 0;
        if (k == 0)
            v += cin0;
        s.U[k] = (uint32_t)v;
        s.carry[k] = (uint32_t)(v >> 32);
    }
    kf_mont_ripple(s.U, s.carry, s.cin, L + 2, tid, nthr, sync);

    // D = U - M as U + ~M + 1 over L+2 limbs; D's top limb is 0 iff U >= M.
    for (int k = tid; k < L + 2; k += nthr)
    {
        uint64_t v = (uint64_t)s.U[k] + (uint32_t)~(k < L ? s.sm[k] : 0u);
        if (k == 0)
            v += 1;
        s.D[k] = (uint32_t)v;
        s.carry[k] = (uint32_t)(v >> 32);
    }
    kf_mont_ripple(s.D, s.carry, s.cin, L + 2, tid, nthr, sync);

    const uint32_t *src = (s.D[L + 1] == 0) ? s.D : s.U;
    for (int k = tid; k < L; k += nthr)
        res[k] = src[k];
    sync.bar();
}
