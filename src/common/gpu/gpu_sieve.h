// =============================================================================
// gpu_sieve.h — host-side segmented prime sieve for the CUDA solvers
//
// --method sieve generates the prime table in memory instead of reading it from
// disk. That generation is plain host code, identical in every CUDA solver that
// offers the flag, so it lives here rather than being copied into each one.
//
// Each OpenMP thread owns its own is_comp[] segment and output vector; results
// are merged in order after the parallel section. Requires -Xcompiler -fopenmp.
//
// Requires common.h (generate_primes_up_to) — included below.
// =============================================================================

#pragma once

#include "common.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// Primes in [lo, hi] as uint32_t. Caller frees the returned array.
static inline uint32_t *sieve_primes_u32(uint64_t lo, uint64_t hi, int64_t *out_count)
{
    int n_small = 0;
    unsigned long *small = generate_primes_up_to((uint64_t)(sqrt((double)hi)) + 2, &n_small);

    const uint64_t SEG = (uint64_t)1 << 22; // 4M candidates per segment (4 MB bool vec)
    int64_t n_segs = (int64_t)((hi - lo + SEG) / SEG);
    std::vector<std::vector<uint32_t>> seg_out((size_t)n_segs);

#pragma omp parallel for schedule(dynamic)
    for (int64_t s = 0; s < n_segs; s++)
    {
        uint64_t seg_lo = lo + (uint64_t)s * SEG;
        uint64_t seg_hi = seg_lo + SEG - 1;
        if (seg_hi > hi)
            seg_hi = hi;
        uint64_t seg_sz = seg_hi - seg_lo + 1;

        std::vector<bool> is_comp(seg_sz, false);
        for (int i = 0; i < n_small; i++)
        {
            uint64_t p = (uint64_t)small[i];
            if (p * p > seg_hi)
                break;
            uint64_t r = seg_lo % p;
            uint64_t start = r ? (seg_lo + p - r) : seg_lo;
            for (uint64_t m = start; m <= seg_hi; m += p)
                is_comp[(size_t)(m - seg_lo)] = true;
        }
        for (uint64_t i = 0; i < seg_sz; i++)
            if (!is_comp[i])
                seg_out[s].push_back((uint32_t)(seg_lo + i));
    }

    free(small);
    int64_t total = 0;
    for (auto &v : seg_out)
        total += (int64_t)v.size();
    uint32_t *arr = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)total);
    int64_t j = 0;
    for (auto &v : seg_out)
    {
        memcpy(arr + j, v.data(), v.size() * sizeof(uint32_t));
        j += (int64_t)v.size();
    }
    *out_count = total;
    return arr;
}

// Primes in [lo, hi] as uint64_t, for ranges above 2^32 (k35 upper table).
// Caller frees the returned array.
static inline uint64_t *sieve_primes_u64_range(uint64_t lo, uint64_t hi, int64_t *out_count)
{
    int n_small = 0;
    unsigned long *small = generate_primes_up_to((uint64_t)(sqrt((double)hi)) + 2, &n_small);

    const uint64_t SEG = (uint64_t)1 << 22; // 4M candidates per segment
    int64_t n_segs = (int64_t)((hi - lo + SEG) / SEG);
    std::vector<std::vector<uint64_t>> seg_out((size_t)n_segs);

#pragma omp parallel for schedule(dynamic)
    for (int64_t s = 0; s < n_segs; s++)
    {
        uint64_t seg_lo = lo + (uint64_t)s * SEG;
        uint64_t seg_hi = seg_lo + SEG - 1;
        if (seg_hi > hi)
            seg_hi = hi;
        uint64_t seg_sz = seg_hi - seg_lo + 1;

        std::vector<bool> is_comp(seg_sz, false);
        for (int i = 0; i < n_small; i++)
        {
            uint64_t p = (uint64_t)small[i];
            if (p * p > seg_hi)
                break;
            uint64_t r = seg_lo % p;
            uint64_t start = r ? (seg_lo + p - r) : seg_lo;
            for (uint64_t m = start; m <= seg_hi; m += p)
                is_comp[(size_t)(m - seg_lo)] = true;
        }
        for (uint64_t i = 0; i < seg_sz; i++)
            if (!is_comp[i])
                seg_out[s].push_back(seg_lo + i);
    }

    free(small);
    int64_t total = 0;
    for (auto &v : seg_out)
        total += (int64_t)v.size();
    uint64_t *arr = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)total);
    int64_t j = 0;
    for (auto &v : seg_out)
    {
        memcpy(arr + j, v.data(), v.size() * sizeof(uint64_t));
        j += (int64_t)v.size();
    }
    *out_count = total;
    return arr;
}
