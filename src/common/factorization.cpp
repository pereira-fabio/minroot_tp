#include "factorization.h"
#include "common.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <cerrno> // for errno in kf_generate_and_save_prime_table_u64

// ---------------------------------------------------------------------------
// k32 — Small prime generator
// ---------------------------------------------------------------------------

unsigned *kf_generate_small_primes(int *count)
{
    const int limit = 65536;
    char *sieve = (char *)calloc((size_t)(limit + 1), 1);
    memset(sieve, 1, (size_t)(limit + 1));
    sieve[0] = sieve[1] = 0;
    for (int i = 2; (long long)i * i <= limit; i++)
        if (sieve[i])
            for (int j = i * i; j <= limit; j += i)
                sieve[j] = 0;
    *count = 0;
    for (int i = 2; i <= limit; i++)
        if (sieve[i])
            (*count)++;
    unsigned *primes = (unsigned *)malloc((size_t)(*count) * sizeof(unsigned));
    int idx = 0;
    for (int i = 2; i <= limit; i++)
        if (sieve[i])
            primes[idx++] = (unsigned)i;
    free(sieve);
    return primes;
}

// ---------------------------------------------------------------------------
// k32 — Prime table generation
// ---------------------------------------------------------------------------

bool kf_generate_and_save_prime_table(const char *path,
                                      int sp_count, const unsigned *sp,
                                      const char *log_prefix)
{
    printf("%sGenerating prime table [65537, 2^32] -> %s...\n", log_prefix ? log_prefix : "", path);
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "Cannot create %s\n", path);
        return false;
    }

    uint64_t magic = KF_PRIME_TABLE_MAGIC,
             count = 0;
    fwrite(&magic, 8, 1, f);
    long count_pos = ftell(f); // remember where to write the final count
    fwrite(&count, 8, 1, f);   // placeholder

    char *seg = (char *)malloc(KF_SEG_SIZE);
    uint32_t *batch = (uint32_t *)malloc(KF_WRITE_BATCH * sizeof(uint32_t));
    int bi = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t lo = 65537ULL; lo <= (uint64_t)UINT32_MAX; lo += KF_SEG_SIZE)
    {
        uint64_t hi = lo + KF_SEG_SIZE - 1;
        if (hi > (uint64_t)UINT32_MAX)
            hi = (uint64_t)UINT32_MAX;
        size_t seg_len = (size_t)(hi - lo + 1);

        memset(seg, 1, seg_len);
        for (int i = 0; i < sp_count; i++)
        {
            uint64_t p = sp[i];
            uint64_t start = ((lo + p - 1) / p) * p;
            if (start == p)
                start += p;
            for (uint64_t j = start; j <= hi; j += p)
                seg[j - lo] = 0;
        }
        for (uint64_t c = lo; c <= hi; c++)
        {
            if (!seg[c - lo])
                continue;
            batch[bi++] = (uint32_t)c;
            count++;
            if (bi == KF_WRITE_BATCH)
            {
                fwrite(batch, 4, (size_t)KF_WRITE_BATCH, f);
                bi = 0;
            }
        }
    }
    if (bi > 0)
        fwrite(batch, 4, (size_t)bi, f);
    free(seg);
    free(batch);

    // write actual prime count into the header placeholder
    fseek(f, count_pos, SEEK_SET);
    fwrite(&count, 8, 1, f);
    fclose(f);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("%sGenerated %" PRIu64 " primes in %.1f ms  (file ~%.0f MB)\n",
           log_prefix ? log_prefix : "", count, ms, (double)(16 + count * 4) / 1e6);
    return true;
}

// ---------------------------------------------------------------------------
// k32 — Load full prime table
// ---------------------------------------------------------------------------

uint32_t *kf_load_prime_table(uint64_t *out_count, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    uint64_t magic, count;
    if (fread(&magic, 8, 1, f) != 1 || magic != KF_PRIME_TABLE_MAGIC ||
        fread(&count, 8, 1, f) != 1)
    {
        fclose(f);
        return NULL;
    }
    uint32_t *primes = (uint32_t *)malloc(count * 4);
    if (!primes || fread(primes, 4, count, f) != count)
    {
        free(primes);
        fclose(f);
        return NULL;
    }
    fclose(f);
    printf("Loaded %" PRIu64 " primes from %s\n", count, path);
    *out_count = count;
    return primes;
}

// ---------------------------------------------------------------------------
// k32 — Binary search within prime table file (internal helper)
// ---------------------------------------------------------------------------

static uint64_t prime_file_lower_bound(FILE *f, uint64_t total_count,
                                       uint32_t target)
{
    uint64_t lo = 0, hi = total_count;
    while (lo < hi)
    {
        uint64_t mid = lo + (hi - lo) / 2;
        uint32_t p;
        fseek(f, (long)(16LL + (long long)mid * 4LL), SEEK_SET);
        if (fread(&p, 4, 1, f) != 1)
            break;
        if (p < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// k32 — Load rank-specific prime slice (MPI path)
// ---------------------------------------------------------------------------

uint32_t *kf_load_prime_slice(uint64_t *slice_count, const char *path,
                              uint32_t range_lo, uint32_t range_hi)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    uint64_t magic, total_count;
    if (fread(&magic, 8, 1, f) != 1 || magic != KF_PRIME_TABLE_MAGIC ||
        fread(&total_count, 8, 1, f) != 1)
    {
        fclose(f);
        return NULL;
    }

    uint64_t start_idx = prime_file_lower_bound(f, total_count, range_lo);
    uint64_t end_idx;
    if (range_hi >= (uint32_t)UINT32_MAX)
        end_idx = total_count;
    else
        end_idx = prime_file_lower_bound(f, total_count, range_hi + 1U);

    *slice_count = end_idx - start_idx;
    if (*slice_count == 0)
    {
        fclose(f);
        return (uint32_t *)malloc(4);
    } // empty, not NULL

    uint32_t *slice = (uint32_t *)malloc(*slice_count * 4);
    if (!slice)
    {
        fclose(f);
        return NULL;
    }

    fseek(f, (long)(16LL + (long long)start_idx * 4LL), SEEK_SET);
    if (fread(slice, 4, *slice_count, f) != *slice_count)
    {
        free(slice);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return slice;
}

// ---------------------------------------------------------------------------
// k32 — Trial division: sieve method
// ---------------------------------------------------------------------------

void kf_trial_divide_sieve(const mpz_t n,
                           uint64_t range_lo, uint64_t range_hi,
                           int sp_count, const unsigned *sp,
                           kfactor_list_t *found,
                           int64_t *thread_tested, int *thread_found,
                           int nthreads_out)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    kfactor_list_t *tf = (kfactor_list_t *)malloc(
        (size_t)nthreads * sizeof(kfactor_list_t));
    for (int t = 0; t < nthreads; t++)
        kfl_init(&tf[t]);

    // Shared cofactor for early termination.
    // Each thread divides out its found factors under a mutex.
    // When cofactor reaches 1, all factors have been found — stop scanning.
    mpz_t cofactor;
    mpz_init_set(cofactor, n);
    omp_lock_t cof_lock;
    omp_init_lock(&cof_lock);
    volatile int done = 0; // set to 1 when cofactor == 1

    uint64_t range_size = range_hi - range_lo + 1;

#ifdef _OPENMP
#pragma omp parallel
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int t = 0, nt = 1;
#endif
        uint64_t t_lo = range_lo + (range_size * (uint64_t)t) / (uint64_t)nt;
        uint64_t t_hi = range_lo + (range_size * (uint64_t)(t + 1)) / (uint64_t)nt - 1;
        if (t == nt - 1)
            t_hi = range_hi;

        char *seg = (char *)malloc(KF_SEG_SIZE);

        // Thread-local snapshot of cofactor for the divisibility pre-check.
        //
        // The shared mpz_t cofactor is only modified under cof_lock, but the
        // pre-check `mpz_tdiv_ui(cofactor, c)` previously read it lock-free.
        // Concurrent reads of a GMP mpz_t while another thread writes its limbs
        // is a data race: mpz_divexact_ui can change limb count and pointers,
        // so a concurrent mpz_tdiv_ui may read partially-updated state and return
        // a wrong remainder, causing a prime factor to be silently missed.
        //
        // Fix: each thread keeps local_cof, a thread-private copy used for the
        // pre-check. It is initialized under the lock and re-synced whenever
        // this thread acquires the lock and processes a factor. This guarantees:
        //   - No data race (local_cof is thread-private; cofactor only read under lock)
        //   - No missed factors: local_cof >= actual cofactor always (factors only
        //     decrease cofactor), so if c divides actual cofactor it also divides
        //     local_cof — no false negatives from the pre-check.
        //   - Occasional false positives (local_cof stale → pre-check passes but
        //     re-check inside lock fails) are harmless: e=0, nothing appended.
        mpz_t local_cof;
        omp_set_lock(&cof_lock);
        mpz_init_set(local_cof, cofactor);
        omp_unset_lock(&cof_lock);

        // Incremental segment offsets: compute the first multiple of each small
        // prime once (one division per prime), then carry it forward across
        // segments by stepping past `hi`.  This removes the per-prime,
        // per-segment division `((lo+p-1)/p)*p` from the hot loop — a leading
        // cost for the large k35 range.  The marking and factor-finding logic
        // below is byte-for-byte identical to the division-based version.
        uint64_t *next_mult = (uint64_t *)malloc((size_t)sp_count * sizeof(uint64_t));
        for (int i = 0; i < sp_count; i++)
        {
            uint64_t p = sp[i];
            uint64_t start = ((t_lo + p - 1) / p) * p;
            if (start == p) // never sieve the prime itself out
                start += p;
            next_mult[i] = start;
        }

        for (uint64_t lo = t_lo; lo <= t_hi && !done; lo += KF_SEG_SIZE)
        {
            uint64_t hi = lo + KF_SEG_SIZE - 1;
            if (hi > t_hi)
                hi = t_hi;
            size_t seg_len = (size_t)(hi - lo + 1);

            memset(seg, 1, seg_len);
            for (int i = 0; i < sp_count; i++)
            {
                uint64_t p = sp[i];
                uint64_t j = next_mult[i];
                for (; j <= hi; j += p)
                    seg[j - lo] = 0;
                next_mult[i] = j; // first multiple > hi — carry to next segment
            }
            for (uint64_t c = lo; c <= hi; c++)
            {
                if (!seg[c - lo])
                    continue;
                // Pre-check against thread-local snapshot (no data race).
                // As factors are stripped, local_cof shrinks → each test is
                // proportionally cheaper (limb count drops).
                if (mpz_tdiv_ui(local_cof, (unsigned long)c) != 0)
                    continue;
                // Factor found — extract full exponent from shared cofactor.
                omp_set_lock(&cof_lock);
                int e = 0;
                while (mpz_divisible_ui_p(cofactor, (unsigned long)c))
                {
                    mpz_divexact_ui(cofactor, cofactor, (unsigned long)c);
                    e++;
                }
                if (mpz_cmp_ui(cofactor, 1) == 0)
                    done = 1;
                // Sync thread-local snapshot so subsequent pre-checks stay
                // accurate (reflects factors removed by all threads so far).
                mpz_set(local_cof, cofactor);
                omp_unset_lock(&cof_lock);
                if (e > 0)
                    kfl_append_ul(&tf[t], (unsigned long)c, e);
            }
        }
        mpz_clear(local_cof);
        free(next_mult);
        free(seg);
    }

    omp_destroy_lock(&cof_lock);
    mpz_clear(cofactor);

    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            kfl_append_mpz(found, tf[t].items[i].prime, tf[t].items[i].exponent);
        kfl_free(&tf[t]);
    }
    free(tf);
}

// ---------------------------------------------------------------------------
// k32 — Trial division: table method
// ---------------------------------------------------------------------------

void kf_trial_divide_table(const mpz_t n,
                           const uint32_t *primes, int64_t prime_count,
                           kfactor_list_t *found,
                           int64_t *thread_tested, int *thread_found,
                           int nthreads_out)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    kfactor_list_t *tf = (kfactor_list_t *)malloc(
        (size_t)nthreads * sizeof(kfactor_list_t));
    for (int t = 0; t < nthreads; t++)
        kfl_init(&tf[t]);

    // Shared cofactor for early termination.
    mpz_t cofactor;
    mpz_init_set(cofactor, n);
    omp_lock_t cof_lock;
    omp_init_lock(&cof_lock);
    volatile int64_t stop_idx = prime_count; // threads stop when idx >= stop_idx

    // Thread-local snapshot of cofactor for the pre-check — same fix as
    // kf_trial_divide_sieve.  The shared cofactor is only written under
    // cof_lock; reading it lock-free from another thread is a data race: GMP's
    // mpz_divexact_ui can change _mp_size and realloc _mp_d, so a concurrent
    // mpz_tdiv_ui may see partially-updated state and return a corrupt value.
    // Worst case: cofactor appears as 0, making mpz_divisible_ui_p(0,p)=true
    // and mpz_divexact_ui(0,p)=0 inside the lock → infinite loop.
    // Fix: each thread keeps local_cof (thread-private), initialised under the
    // lock, re-synced after each factor is stripped.  local_cof >= actual
    // cofactor always, so no false negatives; stale hits just waste one lock
    // acquisition (e=0, nothing appended).
#ifdef _OPENMP
#pragma omp parallel
    {
        int t = omp_get_thread_num();

        mpz_t local_cof;
        omp_set_lock(&cof_lock);
        mpz_init_set(local_cof, cofactor);
        omp_unset_lock(&cof_lock);

#pragma omp for schedule(static)
        for (int64_t i = 0; i < prime_count; i++)
        {
            if (i >= stop_idx)
                continue;

            uint64_t p = (uint64_t)primes[i];
            if (thread_tested && t < nthreads_out)
                thread_tested[t]++;
            if (mpz_tdiv_ui(local_cof, (unsigned long)p) != 0)
                continue;

            omp_set_lock(&cof_lock);
            int e = 0;
            while (mpz_divisible_ui_p(cofactor, (unsigned long)p))
            {
                mpz_divexact_ui(cofactor, cofactor, (unsigned long)p);
                e++;
            }
            if (mpz_cmp_ui(cofactor, 1) == 0)
                stop_idx = i + 1;
            mpz_set(local_cof, cofactor);
            omp_unset_lock(&cof_lock);
            if (e > 0)
            {
                kfl_append_ul(&tf[t], p, e);
                if (thread_found && t < nthreads_out)
                    thread_found[t]++;
            }
        }

        mpz_clear(local_cof);
    } // omp parallel
#else
    {
        int t = 0;
        for (int64_t i = 0; i < prime_count; i++)
        {
            if (i >= stop_idx)
                continue;
            uint64_t p = (uint64_t)primes[i];
            if (thread_tested && t < nthreads_out)
                thread_tested[t]++;
            if (mpz_tdiv_ui(cofactor, (unsigned long)p) != 0)
                continue;
            int e = 0;
            while (mpz_divisible_ui_p(cofactor, (unsigned long)p))
            {
                mpz_divexact_ui(cofactor, cofactor, (unsigned long)p);
                e++;
            }
            if (mpz_cmp_ui(cofactor, 1) == 0)
                stop_idx = i + 1;
            if (e > 0)
            {
                kfl_append_ul(&tf[t], p, e);
                if (thread_found && t < nthreads_out)
                    thread_found[t]++;
            }
        }
    }
#endif

    omp_destroy_lock(&cof_lock);
    mpz_clear(cofactor);

    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            kfl_append_mpz(found, tf[t].items[i].prime, tf[t].items[i].exponent);
        kfl_free(&tf[t]);
    }
    free(tf);
}

// ---------------------------------------------------------------------------
// k16 — Horner trial division
// ---------------------------------------------------------------------------

void kf_trial_divide_k16(const unsigned char *challenge_bytes, size_t nbytes,
                         const mpz_t challenge,
                         const unsigned long *primes, int prime_count,
                         int *exponents)
{
    // OpenMP over the prime array: challenge_bytes and challenge are read-only.
    // exponents[i] is written only by the thread handling index i.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < prime_count; i++)
    {
        unsigned long rem = 0;
        for (size_t b = 0; b < nbytes; b++)
            rem = (rem * 256 + challenge_bytes[b]) % primes[i];
        if (rem != 0)
        {
            exponents[i] = 0;
            continue;
        }
        exponents[i] = count_exponent(challenge, primes[i]);
    }
}

// ---------------------------------------------------------------------------
// k64 — Miller-Rabin primality test
// ---------------------------------------------------------------------------

bool kf_is_prime(const mpz_t n)
{
    return mpz_probab_prime_p(n, 15) > 0;
}

// ---------------------------------------------------------------------------
// k64 — Pollard's rho (Brent variant)
// ---------------------------------------------------------------------------

bool kf_pollard_rho_brent(mpz_t factor, const mpz_t n, unsigned long c_val)
{
    if (mpz_cmp_ui(n, 1) == 0)
        return false;
    if (mpz_divisible_ui_p(n, 2))
    {
        mpz_set_ui(factor, 2);
        return true;
    }

    mpz_t x, y, ys, c, q, diff, g;
    mpz_inits(x, y, ys, c, q, diff, g, NULL);
    mpz_set_ui(c, c_val);

    mpz_set_ui(x, 2);
    mpz_set_ui(y, 2);
    mpz_set_ui(g, 1);

    const long m = 128; // batch GCD size
    long r = 1;

    while (mpz_cmp_ui(g, 1) == 0)
    {
        mpz_set(x, y);
        mpz_set_ui(q, 1); // reset accumulator for this Brent cycle

        // Advance y by r steps (the "giant step")
        for (long i = 0; i < r; i++)
        {
            mpz_mul(y, y, y);
            mpz_add(y, y, c);
            mpz_mod(y, y, n);
        }

        long k = 0;
        while (k < r && mpz_cmp_ui(g, 1) == 0)
        {
            mpz_set(ys, y); // save position for possible backtracking

            for (long i = 0; i < m && i < r - k; i++)
            {
                mpz_mul(y, y, y);
                mpz_add(y, y, c);
                mpz_mod(y, y, n);
                mpz_sub(diff, x, y);
                mpz_abs(diff, diff);
                mpz_mul(q, q, diff);
                mpz_mod(q, q, n);
            }
            mpz_gcd(g, q, n);
            k += m;
        }
        r *= 2;
    }

    // g == n: the batch product happened to be divisible by n (rare).
    // Backtrack from ys using individual GCD calls to find the factor.
    if (mpz_cmp(g, n) == 0)
    {
        mpz_set_ui(g, 1);
        while (mpz_cmp_ui(g, 1) == 0)
        {
            mpz_mul(ys, ys, ys);
            mpz_add(ys, ys, c);
            mpz_mod(ys, ys, n);
            mpz_sub(diff, x, ys);
            mpz_abs(diff, diff);
            mpz_gcd(g, diff, n);
        }
    }

    bool ok = (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, n) < 0);
    if (ok)
        mpz_set(factor, g);

    mpz_clears(x, y, ys, c, q, diff, g, NULL);
    return ok;
}

// ---------------------------------------------------------------------------
// k64 — Recursive complete factorization
// ---------------------------------------------------------------------------

void kf_factor_recursive(const mpz_t n, kfactor_list_t *factors,
                         unsigned long c_start)
{
    if (mpz_cmp_ui(n, 1) == 0)
        return;

    if (kf_is_prime(n))
    {
#pragma omp critical
        kfl_append_mpz(factors, n, 1);
        return;
    }

    // Small trial division fallback — handles tiny factors that Pollard's rho
    // is slow on (c=1 degeneracy), and avoids wasted parallel launches.
    for (unsigned long p = 2; p < 1000; p++)
    {
        if (mpz_divisible_ui_p(n, p))
        {
            mpz_t fp;
            mpz_init_set_ui(fp, p);
            mpz_t co;
            mpz_init(co);
            mpz_divexact_ui(co, n, p);
            kf_factor_recursive(fp, factors, c_start);
            kf_factor_recursive(co, factors, c_start + 31337);
            mpz_clears(fp, co, NULL);
            return;
        }
    }

    // Pollard's rho — threads race with independent c values.
    mpz_t found_factor;
    mpz_init(found_factor);
    volatile int success = 0;

#ifdef _OPENMP
#pragma omp parallel shared(found_factor, success)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int tid = 0, nt = 1;
#endif
        mpz_t local_factor;
        mpz_init(local_factor);
        // Each thread gets a disjoint range of c values:
        // thread 0: c_start+1, c_start+1+nt, c_start+1+2nt, ...
        // thread 1: c_start+2, c_start+2+nt, ...
        unsigned long c = c_start + (unsigned long)(tid + 1);
        const unsigned long limit = c_start + 200000UL * (unsigned long)nt;

        while (c <= limit)
        {
#pragma omp flush(success)
            if (success)
                break;

            // Skip c=0 and c=n-2 (degenerate cycles for Pollard's rho)
            mpz_t bad;
            mpz_init(bad);
            mpz_sub_ui(bad, n, 2);
            bool skip = (c == 0 || (mpz_fits_ulong_p(bad) &&
                                    mpz_get_ui(bad) == c));
            mpz_clear(bad);

            if (!skip && kf_pollard_rho_brent(local_factor, n, c))
            {
#pragma omp critical
                {
                    if (!success)
                    {
                        mpz_set(found_factor, local_factor);
                        success = 1;
                    }
                }
#pragma omp flush(success)
            }
            c += (unsigned long)nt;
        }
        mpz_clear(local_factor);
    } // end parallel region

    if (success)
    {
        mpz_t cofactor;
        mpz_init(cofactor);
        mpz_divexact(cofactor, n, found_factor);
        // Recursive calls use distinct c_start offsets to avoid re-using
        // the same c values that just succeeded (avoids correlated failures).
        kf_factor_recursive(found_factor, factors, c_start + 7919);
        kf_factor_recursive(cofactor, factors, c_start + 104729);
        mpz_clear(cofactor);
    }
    else
    {
        fprintf(stderr, "WARNING: Could not factor %zu-bit number\n",
                mpz_sizeinbase(n, 2));
    }
    mpz_clear(found_factor);
}

// ---------------------------------------------------------------------------
// k64 — Consolidate raw factors
// ---------------------------------------------------------------------------

void kf_factor_challenge(const mpz_t challenge, kfactor_list_t *out)
{
    // Collect raw prime factors (may have duplicates for repeated primes)
    kfactor_list_t raw;
    kfl_init(&raw);
    kf_factor_recursive(challenge, &raw, 1);

    // Consolidate: merge duplicate primes into (prime, exponent) pairs.
    // Mark merged entries by setting prime to 0.
    for (int i = 0; i < raw.count; i++)
    {
        if (mpz_cmp_ui(raw.items[i].prime, 0) == 0)
            continue;
        int exp = 1;
        for (int j = i + 1; j < raw.count; j++)
        {
            if (mpz_cmp(raw.items[i].prime, raw.items[j].prime) == 0)
            {
                exp++;
                mpz_set_ui(raw.items[j].prime, 0); // mark merged
            }
        }
        kfl_append_mpz(out, raw.items[i].prime, exp);
    }
    kfl_free(&raw);
}
// ===========================================================================
// cbrt cache — persistent (prime → cbrt_mod_M) lookup table
// ===========================================================================

void kf_cbrt_cache_init(cbrt_cache_t *cache)
{
    cache->items = NULL;
    cache->count = 0;
    cache->capacity = 0;
}

void kf_cbrt_cache_clear(cbrt_cache_t *cache)
{
    for (int i = 0; i < cache->count; i++)
        mpz_clear(cache->items[i].cbrt);
    free(cache->items);
    cache->items = NULL;
    cache->count = 0;
    cache->capacity = 0;
}

int kf_cbrt_cache_find(const cbrt_cache_t *cache, unsigned long prime)
{
    for (int i = 0; i < cache->count; i++)
        if (cache->items[i].prime == prime)
            return i;
    return -1;
}

void kf_cbrt_cache_append(cbrt_cache_t *cache, unsigned long prime,
                          const mpz_t val)
{
    if (cache->count == cache->capacity)
    {
        int nc = (cache->capacity == 0) ? 64 : cache->capacity * 2;
        cache->items = (cbrt_entry_t *)realloc(
            cache->items, (size_t)nc * sizeof(cbrt_entry_t));
        cache->capacity = nc;
    }
    cache->items[cache->count].prime = prime;
    mpz_init_set(cache->items[cache->count].cbrt, val);
    cache->count++;
}

bool kf_cbrt_cache_load(cbrt_cache_t *cache, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return false;

    int stored = 0;
    if (fscanf(f, "%d", &stored) != 1 || stored < 0)
    {
        fclose(f);
        return false;
    }

    kf_cbrt_cache_clear(cache);
    kf_cbrt_cache_init(cache);

    for (int i = 0; i < stored; i++)
    {
        unsigned long p = 0;
        if (fscanf(f, "%lu", &p) != 1)
        {
            fclose(f);
            kf_cbrt_cache_clear(cache);
            kf_cbrt_cache_init(cache);
            return false;
        }
        mpz_t v;
        mpz_init(v);
        if (mpz_inp_str(v, f, 10) == 0)
        {
            mpz_clear(v);
            fclose(f);
            kf_cbrt_cache_clear(cache);
            kf_cbrt_cache_init(cache);
            return false;
        }
        kf_cbrt_cache_append(cache, p, v);
        mpz_clear(v);
    }

    fclose(f);
    printf("Loaded %d cached cbrt entries from %s\n", cache->count, path);
    return true;
}

bool kf_cbrt_cache_save(const cbrt_cache_t *cache, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Cannot write cbrt cache to %s\n", path);
        return false;
    }
    fprintf(f, "%d\n", cache->count);
    for (int i = 0; i < cache->count; i++)
    {
        fprintf(f, "%lu ", cache->items[i].prime);
        mpz_out_str(f, 10, cache->items[i].cbrt);
        fputc('\n', f);
    }
    fclose(f);
    printf("Saved %d cached cbrt entries to %s\n", cache->count, path);
    return true;
}

// ===========================================================================
// k32 — reconstruction: result = ∏ cbrt(p_i)^e_i  mod M
// ===========================================================================

void kf_compute_cbrt_from_factors(mpz_t result,
                                  const kfactor_list_t *found,
                                  const mpz_t modulo,
                                  cbrt_cache_t *cache,
                                  bool *cache_dirty,
                                  unsigned long exponent)
{
    // Precompute e_inv = exponent^{-1} mod (M-1) once for this modulo.
    mpz_t phi, e_inv, e_base;
    mpz_inits(phi, e_inv, e_base, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);
    mpz_invert(e_inv, e_base, phi);
    mpz_clears(phi, e_base, NULL);

    mpz_set_ui(result, 1);

    for (int i = 0; i < found->count; i++)
    {
        unsigned long p = mpz_get_ui(found->items[i].prime);
        int idx = kf_cbrt_cache_find(cache, p);

        mpz_t cbrt_p;
        mpz_init(cbrt_p);

        if (idx >= 0)
        {
            mpz_set(cbrt_p, cache->items[idx].cbrt);
        }
        else
        {
            // Cache miss: compute p^e_inv mod M and store for next time.
            mpz_t p_mpz;
            mpz_init_set_ui(p_mpz, p);
            mpz_powm(cbrt_p, p_mpz, e_inv, modulo);
            mpz_clear(p_mpz);
            kf_cbrt_cache_append(cache, p, cbrt_p);
            if (cache_dirty)
                *cache_dirty = true;
        }

        if (found->items[i].exponent > 1)
        {
            mpz_t powered;
            mpz_init(powered);
            mpz_powm_ui(powered, cbrt_p,
                        (unsigned long)found->items[i].exponent, modulo);
            mpz_mul(result, result, powered);
            mpz_clear(powered);
        }
        else
        {
            mpz_mul(result, result, cbrt_p);
        }
        mpz_mod(result, result, modulo);
        mpz_clear(cbrt_p);
    }

    mpz_clear(e_inv);
}

// ===========================================================================
// blinding_pair_t table — precomputation phase
// ===========================================================================

blinding_pair_t *kf_blinding_table_alloc(int count, const mpz_t modulo,
                                         unsigned long exponent)
{
    blinding_pair_t *table =
        (blinding_pair_t *)malloc((size_t)count * sizeof(blinding_pair_t));
    if (!table)
        return NULL;

    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    gmp_randseed_ui(rng, (unsigned long)time(NULL) ^ (unsigned long)(uintptr_t)table);

    mpz_t r;
    mpz_init(r);

    for (int i = 0; i < count; i++)
    {
        mpz_inits(table[i].r, table[i].r_blind, table[i].r_inv, NULL);

        do
        {
            mpz_urandomm(r, rng, modulo);
        } while (mpz_cmp_ui(r, 2) < 0);

        mpz_set(table[i].r, r);
        mpz_powm_ui(table[i].r_blind, r, exponent, modulo); // r^e mod M
        mpz_invert(table[i].r_inv, r, modulo);  // r^{-1} mod M
    }

    mpz_clear(r);
    gmp_randclear(rng);
    return table;
}

void kf_blinding_table_free(blinding_pair_t *table, int count)
{
    if (!table)
        return;
    for (int i = 0; i < count; i++)
        mpz_clears(table[i].r, table[i].r_blind, table[i].r_inv, NULL);
    free(table);
}

// ===========================================================================
// k16 — Precompute and reconstruct
// ===========================================================================

void kf_precompute_cbrt_table_k16(mpz_t *cbrt_table, const unsigned long *primes,
                                  int prime_count, const mpz_t modulo,
                                  unsigned long exponent)
{
    mpz_t phi, e_inv, e_base;
    mpz_inits(phi, e_inv, e_base, NULL);

    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);

    // e_inv = exponent^{-1} mod (M-1)
    if (!mpz_invert(e_inv, e_base, phi))
    {
        fprintf(stderr, "[ERROR] Modulo not suitable for %lu-th roots (gcd(%lu, M-1) != 1)\n",
                exponent, exponent);
        exit(1);
    }

    printf("[SETUP] Precomputing %d %lu-th roots...\n", prime_count, exponent);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < prime_count; i++)
    {
        mpz_t p;
        mpz_init_set_ui(p, primes[i]);
        mpz_init(cbrt_table[i]);
        mpz_powm(cbrt_table[i], p, e_inv, modulo);
        mpz_clear(p);
    }

    mpz_clears(phi, e_inv, e_base, NULL);
}

void kf_save_cbrt_table_k16(const mpz_t *cbrt_table, int prime_count, const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return;
    fwrite(&prime_count, sizeof(int), 1, f);
    for (int i = 0; i < prime_count; i++)
    {
        mpz_out_raw(f, cbrt_table[i]);
    }
    fclose(f);
}

bool kf_load_cbrt_table_k16(mpz_t *cbrt_table, int prime_count, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        return false;
    int stored_count;
    if (fread(&stored_count, sizeof(int), 1, f) != 1 || stored_count != prime_count)
    {
        fclose(f);
        return false;
    }
    for (int i = 0; i < prime_count; i++)
    {
        mpz_init(cbrt_table[i]);
        if (mpz_inp_raw(cbrt_table[i], f) == 0)
        {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

/**
 * Reconstructs the final cube root using the homomorphic property:
 * result = prod(cbrt_table[i]^exponents[i]) mod M
 */
void kf_compute_cube_root_from_factors_k16(mpz_t result, const int *exponents,
                                           const mpz_t *cbrt_table, int prime_count,
                                           const mpz_t modulo)
{
    mpz_set_ui(result, 1);
    for (int i = 0; i < prime_count; i++)
    {
        if (exponents[i] == 0)
            continue;

        if (exponents[i] == 1)
        {
            mpz_mul(result, result, cbrt_table[i]);
        }
        else
        {
            mpz_t tmp;
            mpz_init(tmp);
            mpz_powm_ui(tmp, cbrt_table[i], (unsigned long)exponents[i], modulo);
            mpz_mul(result, result, tmp);
            mpz_clear(tmp);
        }
        mpz_mod(result, result, modulo);
    }
}
// ---------------------------------------------------------------------------
// kf_try_factor_b64smooth — factor n via Pollard's rho, check 2^64-smooth
// ---------------------------------------------------------------------------

bool kf_try_factor_b64smooth(const mpz_t n,
                             kfactor_list_t *factors,
                             unsigned long c_start)
{
    if (mpz_cmp_ui(n, 1) == 0)
        return true;

    // If n fits in 64 bits, it is automatically 2^64-smooth.
    // Use kf_factor_recursive to fully factor it.
    if (mpz_sizeinbase(n, 2) <= 64)
    {
        kf_factor_recursive(n, factors, c_start);
        return true;
    }

    // n > 2^64: test if it is prime (would mean not 2^64-smooth)
    if (kf_is_prime(n))
        return false; // prime factor > 2^64 — not smooth

    // n is composite and > 2^64 — find a factor via Pollard's rho
    mpz_t found;
    mpz_init(found);
    volatile int success = 0;

#ifdef _OPENMP
#pragma omp parallel shared(found, success)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int tid = 0, nt = 1;
#endif
        mpz_t local_f;
        mpz_init(local_f);
        unsigned long c = c_start + (unsigned long)(tid + 1);
        const unsigned long limit = c_start + 500000UL * (unsigned long)nt;

        while (c <= limit)
        {
#pragma omp flush(success)
            if (success)
                break;

            if (kf_pollard_rho_brent(local_f, n, c))
            {
#pragma omp critical
                if (!success)
                {
                    mpz_set(found, local_f);
                    success = 1;
                }
#pragma omp flush(success)
            }
            c += (unsigned long)nt;
        }
        mpz_clear(local_f);
    }

    bool ok = false;
    if (success)
    {
        // Check both halves for 2^64-smoothness recursively
        mpz_t cofactor;
        mpz_init(cofactor);
        mpz_divexact(cofactor, n, found);

        kfactor_list_t left, right;
        kfl_init(&left);
        kfl_init(&right);

        bool left_ok = kf_try_factor_b64smooth(found, &left, c_start + 7919);
        bool right_ok = kf_try_factor_b64smooth(cofactor, &right, c_start + 104729);
        ok = left_ok && right_ok;

        if (ok)
        {
            for (int i = 0; i < left.count; i++)
                kfl_append_mpz(factors, left.items[i].prime, left.items[i].exponent);
            for (int i = 0; i < right.count; i++)
                kfl_append_mpz(factors, right.items[i].prime, right.items[i].exponent);
        }
        kfl_free(&left);
        kfl_free(&right);
        mpz_clear(cofactor);
    }
    mpz_clear(found);
    return ok;
}

// =============================================================================
// k35 upper-range prime table  — uint64_t primes in (2^32, 2^35]
// =============================================================================

// ---------------------------------------------------------------------------
// kf_generate_and_save_prime_table_u64
// ---------------------------------------------------------------------------
bool kf_generate_and_save_prime_table_u64(
    const char *path,
    uint64_t range_lo, uint64_t range_hi,
    int sp_count, const unsigned *sp,
    const char *log_prefix)
{
    // Use the same 512 KiB segment as the sieve
    const uint64_t SEG = (uint64_t)KF_SEG_SIZE;

    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "%sERROR: cannot create %s: %s\n",
                log_prefix, path, strerror(errno));
        return false;
    }

    // Write placeholder header (magic + count = 0)
    uint64_t header[2] = {K35U_PRIME_TABLE_MAGIC, 0};
    if (fwrite(header, sizeof(uint64_t), 2, f) != 2)
    {
        fclose(f);
        return false;
    }

    uint64_t total = 0;
    // Write primes in batches to reduce fwrite calls
    uint64_t batch[KF_WRITE_BATCH];
    int bsz = 0;

    // Temporary sieve buffer — one byte per candidate offset
    unsigned char *sieve = (unsigned char *)malloc(SEG);
    if (!sieve)
    {
        fclose(f);
        return false;
    }

    for (uint64_t seg_lo = range_lo; seg_lo <= range_hi; seg_lo += SEG)
    {
        uint64_t seg_hi = seg_lo + SEG - 1;
        if (seg_hi > range_hi)
            seg_hi = range_hi;
        uint64_t seg_sz = seg_hi - seg_lo + 1;

        // Mark all as prime (0 = prime candidate)
        memset(sieve, 0, seg_sz);

        // Sieve with each small prime
        for (int i = 0; i < sp_count; i++)
        {
            uint64_t p = (uint64_t)sp[i];
            // First multiple of p in segment
            uint64_t start = ((seg_lo + p - 1) / p) * p;
            if (start < p * 2)
                start = p * 2; // don't mark p itself
            for (uint64_t j = start; j <= seg_hi; j += p)
                sieve[j - seg_lo] = 1;
        }

        // Collect unmarked candidates
        for (uint64_t j = 0; j < seg_sz; j++)
        {
            if (!sieve[j])
            {
                batch[bsz++] = seg_lo + j;
                total++;
                if (bsz == KF_WRITE_BATCH)
                {
                    fwrite(batch, sizeof(uint64_t), KF_WRITE_BATCH, f);
                    bsz = 0;
                }
            }
        }

        // Progress: every 256 segments (~128 MB range)
        if (((seg_lo - range_lo) / SEG) % 256 == 0)
        {
            double pct = 100.0 * (double)(seg_lo - range_lo) / (double)(range_hi - range_lo);
            fprintf(stderr, "%s  %.1f%%  (%" PRIu64 " primes so far)\r",
                    log_prefix, pct, total);
            fflush(stderr);
        }
    }

    // Flush remaining batch
    if (bsz > 0)
        fwrite(batch, sizeof(uint64_t), bsz, f);

    // Update count in header
    fseeko(f, (off_t)sizeof(uint64_t), SEEK_SET);
    fwrite(&total, sizeof(uint64_t), 1, f);
    fclose(f);
    free(sieve);

    fprintf(stderr, "%s  100.0%%  %" PRIu64 " primes written to %s\n",
            log_prefix, total, path);
    return true;
}

// ---------------------------------------------------------------------------
// kf_load_prime_slice_u64
//
// Binary-searches the file for the first prime >= lo, then reads until > hi.
// Returns malloc'd array.  Caller must free().
// ---------------------------------------------------------------------------
uint64_t *kf_load_prime_slice_u64(int64_t *count_out, const char *path,
                                  uint64_t lo, uint64_t hi)
{
    *count_out = 0;

    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "[k35u] Cannot open %s\n", path);
        return nullptr;
    }

    uint64_t magic, total;
    if (fread(&magic, 8, 1, f) != 1 || fread(&total, 8, 1, f) != 1 || magic != K35U_PRIME_TABLE_MAGIC)
    {
        fprintf(stderr, "[k35u] Bad header in %s\n", path);
        fclose(f);
        return nullptr;
    }

    if (total == 0)
    {
        fclose(f);
        return nullptr;
    }

    // Binary search: find index of first prime >= lo
    const off_t DATA_OFF = 2 * (off_t)sizeof(uint64_t);
    int64_t left = 0, right = (int64_t)total - 1, start_idx = (int64_t)total;
    while (left <= right)
    {
        int64_t mid = left + (right - left) / 2;
        uint64_t val;
        fseeko(f, DATA_OFF + (off_t)mid * 8, SEEK_SET);
        if (fread(&val, 8, 1, f) != 1)
            break;
        if (val < lo)
        {
            left = mid + 1;
        }
        else
        {
            start_idx = mid;
            right = mid - 1;
        }
    }

    if (start_idx == (int64_t)total)
    {
        fclose(f);
        return nullptr;
    }

    // Count entries in [lo, hi]
    fseeko(f, DATA_OFF + (off_t)start_idx * 8, SEEK_SET);
    int64_t count = 0;
    {
        uint64_t buf[4096];
        int64_t idx = start_idx;
        while (idx < (int64_t)total)
        {
            size_t rd = fread(buf, 8, 4096, f);
            if (rd == 0)
                break;
            for (size_t k = 0; k < rd; k++)
            {
                if (buf[k] > hi)
                {
                    idx = (int64_t)total;
                    break;
                }
                count++;
                idx++;
            }
        }
    }

    if (count == 0)
    {
        fclose(f);
        return nullptr;
    }

    // Load the slice
    uint64_t *slice = (uint64_t *)malloc((size_t)count * sizeof(uint64_t));
    if (!slice)
    {
        fclose(f);
        return nullptr;
    }
    fseeko(f, DATA_OFF + (off_t)start_idx * 8, SEEK_SET);
    if ((int64_t)fread(slice, 8, (size_t)count, f) != count)
    {
        free(slice);
        fclose(f);
        return nullptr;
    }
    fclose(f);

    *count_out = count;
    return slice;
}

// ---------------------------------------------------------------------------
// kf_trial_divide_table_u64
//
// Same logic as kf_trial_divide_table but for uint64_t primes.
// On 64-bit Linux, unsigned long == uint64_t so mpz_divisible_ui_p works.
// ---------------------------------------------------------------------------
void kf_trial_divide_table_u64(
    const mpz_t n,
    const uint64_t *primes, int64_t prime_count,
    kfactor_list_t *found,
    int64_t *thread_tested, int *thread_found, int nthreads_out)
{
    if (!primes || prime_count <= 0)
        return;

    // Use all available OMP threads unless caller specifies count
    int nt = nthreads_out > 0 ? nthreads_out : 1;
#ifdef _OPENMP
    if (nthreads_out <= 0)
        nt = omp_get_max_threads();
#endif

#pragma omp parallel num_threads(nt)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        kfactor_list_t local;
        kfl_init(&local);
        int64_t tested = 0;
        int found_cnt = 0;

        // Thread-private shrinking copy of n (Bug 13 fix).  Each thread strips
        // out only the primes it personally finds, so later divisibility tests
        // run against a smaller number (cheaper mpz_divisible_ui_p).  This is
        // lock-free and cannot miss a factor: a prime p is tested by exactly
        // one thread (static partition over distinct primes), and at the moment
        // that thread tests p, local_n still contains p at full multiplicity
        // (only OTHER primes this thread already found were removed).  So
        // p | n  <=>  p | local_n  at the test point.
        mpz_t local_n;
        mpz_init_set(local_n, n);

#pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < prime_count; i++)
        {
            unsigned long p = (unsigned long)primes[i];
            tested++;
            if (mpz_divisible_ui_p(local_n, p))
            {
                // Count exponent, stripping p from this thread's private copy.
                int e = 0;
                while (mpz_divisible_ui_p(local_n, p))
                {
                    mpz_divexact_ui(local_n, local_n, p);
                    e++;
                }
                kfl_append_ul(&local, p, e);
                found_cnt++;
            }
        }
        mpz_clear(local_n);

        // Merge local into shared found
#pragma omp critical
        {
            for (int i = 0; i < local.count; i++)
                kfl_append_ul(found,
                              (unsigned long)mpz_get_ui(local.items[i].prime),
                              local.items[i].exponent);
            if (thread_tested && tid < nthreads_out)
                thread_tested[tid] += tested;
            if (thread_found && tid < nthreads_out)
                thread_found[tid] += found_cnt;
        }
        kfl_free(&local);
    }
}
// Edit: 2026-06-11
