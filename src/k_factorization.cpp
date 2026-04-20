#include "k_factorization.h"
#include "k_common.h"

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
                           kfactor_list_t *found)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    kfactor_list_t *tf = (kfactor_list_t *)malloc(
        (size_t)nthreads * sizeof(kfactor_list_t));
    for (int t = 0; t < nthreads; t++)
        kfl_init(&tf[t]);

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

        for (uint64_t lo = t_lo; lo <= t_hi; lo += KF_SEG_SIZE)
        {
            uint64_t hi = lo + KF_SEG_SIZE - 1;
            if (hi > t_hi)
                hi = t_hi;
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
                if (mpz_tdiv_ui(n, (unsigned long)c) != 0)
                    continue;
                int e = count_exponent(n, (unsigned long)c);
                if (e > 0)
                    kfl_append_ul(&tf[t], (unsigned long)c, e);
            }
        }
        free(seg);
    }

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
                           kfactor_list_t *found)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    kfactor_list_t *tf = (kfactor_list_t *)malloc(
        (size_t)nthreads * sizeof(kfactor_list_t));
    for (int t = 0; t < nthreads; t++)
        kfl_init(&tf[t]);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < prime_count; i++)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        unsigned long p = (unsigned long)primes[i];
        if (mpz_tdiv_ui(n, p) != 0)
            continue;
        int e = count_exponent(n, p);
        if (e > 0)
            kfl_append_ul(&tf[t], p, e);
    }

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
                                  bool *cache_dirty)
{
    // Precompute e = 3^{-1} mod (M-1) once for this modulo.
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);
    mpz_clears(phi, three, NULL);

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
            // Cache miss: compute p^e mod M and store for next time.
            mpz_t p_mpz;
            mpz_init_set_ui(p_mpz, p);
            mpz_powm(cbrt_p, p_mpz, e, modulo);
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

    mpz_clear(e);
}

// ===========================================================================
// kf_factorize_with_blinding — Algorithm 2 retry loop (single-node)
// ===========================================================================
//
// On attempt 0 we set r=1 (try the challenge unblinded first — succeeds
// immediately for the constructed test vectors where the challenge IS the
// smooth number).  For attempt > 0 we draw r uniformly from [2, M-1].
//
// "Fully factored" check: remove every found (p^e) from the blinded
// challenge; if the remainder is 1 the factorisation is complete.
// ===========================================================================

bool kf_factorize_with_blinding(
    mpz_t result,
    const mpz_t challenge,
    const mpz_t modulo,
    int sp_count, const unsigned *sp,
    const uint32_t *prime_table, int64_t prime_table_count,
    cbrt_cache_t *cache, bool *cache_dirty,
    int max_attempts,
    int *attempts_out)
{
    if (attempts_out)
        *attempts_out = 0;

    // Seed GMP's Mersenne-Twister RNG.
    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    gmp_randseed_ui(rng, (unsigned long)time(NULL) ^ (unsigned long)(uintptr_t)&rng);

    mpz_t r, blinded, rem, cbrt_y;
    mpz_inits(r, blinded, rem, cbrt_y, NULL);

    bool success = false;

    for (int attempt = 0; attempt < max_attempts && !success; attempt++)
    {
        if (attempts_out)
            *attempts_out = attempt + 1;

        // ---- Choose blinding factor r ----
        if (attempt == 0)
        {
            // Fast path: try challenge directly (r = 1).
            mpz_set_ui(r, 1);
            mpz_set(blinded, challenge);
        }
        else
        {
            // Random r in [2, M-1]: urandomm gives [0, M), re-draw if < 2.
            do
            {
                mpz_urandomm(r, rng, modulo);
            } while (mpz_cmp_ui(r, 2) < 0);

            // blinded = challenge * r^3 mod M
            blind_challenge(blinded, challenge, r, modulo);
        }

        // ---- Factor the blinded challenge ----
        kfactor_list_t found;
        kfl_init(&found);

        if (prime_table && prime_table_count > 0)
            kf_trial_divide_table(blinded, prime_table, prime_table_count,
                                  &found);
        else
            kf_trial_divide_sieve(blinded, 65537ULL, (uint64_t)UINT32_MAX,
                                  sp_count, sp, &found);

        // ---- Check completeness: rem = blinded / ∏ p_i^{e_i} ----
        mpz_set(rem, blinded);
        for (int i = 0; i < found.count; i++)
        {
            unsigned long p = mpz_get_ui(found.items[i].prime);
            for (int ex = 0; ex < found.items[i].exponent; ex++)
                mpz_divexact_ui(rem, rem, p);
        }

        if (mpz_cmp_ui(rem, 1) == 0)
        {
            // ---- Reconstruct cbrt(blinded) ----
            kf_compute_cbrt_from_factors(cbrt_y, &found, modulo,
                                         cache, cache_dirty);

            // ---- Unblind: cbrt(challenge) = cbrt(y) * r^{-1} mod M ----
            if (attempt == 0)
            {
                // r = 1 → no unblinding needed
                mpz_set(result, cbrt_y);
            }
            else
            {
                unblind_root(result, cbrt_y, r, modulo);
            }
            success = true;
        }

        kfl_free(&found);
    }

    mpz_clears(r, blinded, rem, cbrt_y, NULL);
    gmp_randclear(rng);
    return success;
}