// =============================================================================
// k32.cpp — Cube root solver for challenges with 32-bit prime factors
//           Single-node version (OpenMP only)
//
// Why a different approach from k16:
//   k16 precomputes cbrt(q) for all 6,542 primes q <= 2^16. That table is ~1MB.
//   For k32 there are ~203 million primes <= 2^32: a full table would be ~52GB.
//   Instead we generate primes on-the-fly using a segmented sieve and compute
//   cbrt(p) on-the-fly only for the handful of primes that actually divide.
//
// Algorithm:
//   1. Generate "small primes" <= 65536 (needed to sieve 32-bit segments)
//   2. Segmented sieve over [65537, 2^32]:
//        - Process 512K-entry segments at a time (fits in L2 cache)
//        - Mark composites using small primes
//        - For each surviving prime candidate: check divisibility with mpz_tdiv_ui
//        - For divisors: record (prime, exponent) — typically ~N_bits/32 hits
//   3. For each found prime p: compute cbrt(p) = p^e mod M  (on-the-fly powm)
//   4. Reconstruct: result = prod cbrt(p_i)^exp_i  mod M
//
// Parallelism:
//   The sieve range is split evenly across OpenMP threads. Each thread has its
//   own private segment buffer and factor list — no shared-memory contention
//   during the hot trial division loop.
//
// For MPI+OpenMP multi-node version see k32_mpi.cpp.
// =============================================================================

#include "k_common.h"

#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// Segment size for the sieve: 512K entries (one byte per candidate).
// Must fit in L2 cache to avoid memory bandwidth becoming the bottleneck.
#define SEG_SIZE (1 << 19)

// ---------------------------------------------------------------------------
// Small prime generation — Sieve of Eratosthenes up to sqrt(2^32) = 65536
//
// These primes are used to mark composites inside each sieve segment.
// All primes <= 65536 are needed to correctly sieve any 32-bit range segment
// (because every composite <= 2^32 has a factor <= sqrt(2^32) = 65536).
// ---------------------------------------------------------------------------

static unsigned *generate_small_primes(int *count)
{
    const int limit = 65536;
    char *sieve = (char *)calloc(limit + 1, 1);
    memset(sieve, 1, limit + 1);
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
            primes[idx++] = i;
    free(sieve);
    return primes;
}

// ---------------------------------------------------------------------------
// Factor list — (prime, exponent) pairs
//
// Stores the distinct prime factors found during trial division.
// Uses unsigned long (sufficient for 32-bit primes on all platforms).
// ---------------------------------------------------------------------------

typedef struct
{
    unsigned long prime;
    int exponent;
} factor_t;
typedef struct
{
    factor_t *items;
    int count;
    int capacity;
} found_factors_t;

static void found_factors_init(found_factors_t *f)
{
    f->items = NULL;
    f->count = 0;
    f->capacity = 0;
}

static void found_factors_append(found_factors_t *f, unsigned long prime, int exp)
{
    if (f->count == f->capacity)
    {
        int nc = (f->capacity == 0) ? 64 : f->capacity * 2;
        f->items = (factor_t *)realloc(f->items, (size_t)nc * sizeof(factor_t));
        f->capacity = nc;
    }
    f->items[f->count].prime = prime;
    f->items[f->count].exponent = exp;
    f->count++;
}

static void found_factors_clear(found_factors_t *f)
{
    free(f->items);
    f->items = NULL;
    f->count = 0;
    f->capacity = 0;
}

// ---------------------------------------------------------------------------
// trial_divide_range — segmented sieve trial division with OpenMP
//
// Finds all primes in [range_lo, range_hi] that divide n, recording each
// as a (prime, exponent) pair in *found.
//
// Parallelism model:
//   - The range [range_lo, range_hi] is split into nthreads equal sub-ranges
//   - Each thread independently sieves its sub-range in 512K segments
//   - Each thread maintains a private factor list (no locking in hot loop)
//   - After the parallel region, thread lists are merged sequentially
//
// Divisibility check:
//   mpz_tdiv_ui(n, c) computes n mod c using GMP's internal limb arithmetic.
//   This is fast (O(limbs)) and avoids allocation — called for every prime
//   candidate surviving the sieve. count_exponent() (which allocates a copy
//   of n) is called only for the rare primes that actually divide.
// ---------------------------------------------------------------------------

static void trial_divide_range(
    const mpz_t n,
    uint64_t range_lo, uint64_t range_hi,
    int sp_count, const unsigned *sp,
    found_factors_t *found)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    // each thread gets its own factor list — avoids locking during the hot loop
    found_factors_t *tf = (found_factors_t *)malloc(
        (size_t)nthreads * sizeof(found_factors_t));
    for (int t = 0; t < nthreads; t++)
        found_factors_init(&tf[t]);

    uint64_t range_size = range_hi - range_lo + 1;

#ifdef _OPENMP
#pragma omp parallel
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
#else
    {
        int t = 0, nt = 1;
#endif
        // this thread's sub-range — contiguous slice for good cache locality
        uint64_t t_lo = range_lo + (range_size * (uint64_t)t) / (uint64_t)nt;
        uint64_t t_hi = range_lo + (range_size * (uint64_t)(t + 1)) / (uint64_t)nt - 1;
        if (t == nt - 1)
            t_hi = range_hi;

        char *seg = (char *)malloc(SEG_SIZE); // private sieve buffer per thread

        for (uint64_t lo = t_lo; lo <= t_hi; lo += SEG_SIZE)
        {
            uint64_t hi = lo + SEG_SIZE - 1;
            if (hi > t_hi)
                hi = t_hi;
            size_t seg_len = (size_t)(hi - lo + 1);

            // initialize segment: all candidates marked as prime
            memset(seg, 1, seg_len);

            // mark composites: for each small prime p, find first multiple >= lo
            // that is not p itself, then stride through segment
            for (int i = 0; i < sp_count; i++)
            {
                uint64_t p = sp[i];
                uint64_t start = ((lo + p - 1) / p) * p;
                if (start == p)
                    start += p; // don't mark p itself as composite
                for (uint64_t j = start; j <= hi; j += p)
                    seg[j - lo] = 0;
            }

            // trial division: surviving candidates are primes — check divisibility
            for (uint64_t candidate = lo; candidate <= hi; candidate++)
            {
                if (!seg[candidate - lo])
                    continue; // composite — skip

                // fast divisibility test (no GMP allocation)
                if (mpz_tdiv_ui(n, (unsigned long)candidate) != 0)
                    continue;

                // n is divisible — count exact multiplicity
                int exp = count_exponent(n, (unsigned long)candidate);
                if (exp > 0)
                    found_factors_append(&tf[t], (unsigned long)candidate, exp);
            }
        }
        free(seg);
    }

    // merge thread-local results into the caller's list
    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            found_factors_append(found, tf[t].items[i].prime,
                                 tf[t].items[i].exponent);
        found_factors_clear(&tf[t]);
    }
    free(tf);
}

// ---------------------------------------------------------------------------
// compute_cube_root_from_found — on-the-fly reconstruction
//
// For each found prime p with exponent e:
//   cbrt(p^e) = cbrt(p)^e = (p^inv3)^e  mod M
// where inv3 = 3^-1 mod (M-1) is computed once.
//
// Unlike k16, there is no precomputed table here — but there are only
// ~N_bits/32 factors (e.g. ~34 for 1024-bit), so powm is called only ~34
// times. Each call is independent so the loop is OpenMP-parallelized.
// ---------------------------------------------------------------------------

static void compute_cube_root_from_found(
    mpz_t result,
    const found_factors_t *found,
    const mpz_t modulo)
{
    // e = 3^-1 mod (modulo-1), computed once for all factors
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);

    int n = found->count;
    mpz_t *parts = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    for (int i = 0; i < n; i++)
        mpz_init(parts[i]);

    // all powm calls are independent — safe to parallelize
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++)
    {
        mpz_t p_mpz, cbrt_p;
        mpz_inits(p_mpz, cbrt_p, NULL);
        mpz_set_ui(p_mpz, found->items[i].prime);
        mpz_powm(cbrt_p, p_mpz, e, modulo); // cbrt(p) = p^e mod M

        if (found->items[i].exponent == 1)
            mpz_set(parts[i], cbrt_p);
        else
            // cbrt(p^k) = cbrt(p)^k — avoids k separate powm calls
            mpz_powm_ui(parts[i], cbrt_p,
                        (unsigned long)found->items[i].exponent, modulo);

        mpz_clears(p_mpz, cbrt_p, NULL);
    }

    // combine: result = prod parts[i]  mod M
    mpz_set_ui(result, 1);
    for (int i = 0; i < n; i++)
    {
        mpz_mul(result, result, parts[i]);
        mpz_mod(result, result, modulo);
        mpz_clear(parts[i]);
    }
    free(parts);
    mpz_clears(phi, e, three, NULL);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *challenge_file = "data/challenges/k32/challenge_32_1024.txt";
    const char *modulo_file = "data/challenges/k32/modulo_32_1024.txt";
    int bench_count = 1;

    int positional_index = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--bench") == 0)
        {
            if (argi + 1 >= argc)
            {
                fprintf(stderr, "Missing --bench value\n");
                return 1;
            }
            bench_count = atoi(argv[++argi]);
            continue;
        }
        if (positional_index == 0)
            challenge_file = argv[argi];
        else if (positional_index == 1)
            modulo_file = argv[argi];
        else
        {
            fprintf(stderr, "Unexpected argument '%s'\n", argv[argi]);
            return 1;
        }
        positional_index++;
    }

    printf("Challenge file: %s\n", challenge_file);
    printf("Modulo file:    %s\n", modulo_file);
    if (bench_count > 1)
        printf("Benchmark mode: %d runs\n", bench_count);

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);
    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);

    mpz_t result;
    mpz_init(result);

#ifdef _OPENMP
    printf("OpenMP enabled with %d threads available.\n", omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i; // warm up
#endif

    // generate small primes once — reused by every sieve segment and bench run
    printf("Generating small primes for sieve...\n");
    int sp_count;
    unsigned *sp = generate_small_primes(&sp_count);
    printf("Generated %d small primes up to 65536.\n", sp_count);

    size_t challenge_bits = mpz_sizeinbase(challenge, 2);
    printf("Challenge: %zu bits\n", challenge_bits);
    printf("~%zu on-the-fly powm calls expected\n", challenge_bits / 32);

    double total_factor_ms = 0.0, total_root_ms = 0.0;

    for (int bench = 0; bench < bench_count; bench++)
    {
        found_factors_t found;
        found_factors_init(&found);

        auto t0 = std::chrono::high_resolution_clock::now();

        // trial division over the full 32-bit prime range above 16-bit boundary
        trial_divide_range(challenge, 65537, UINT32_MAX, sp_count, sp, &found);

        // verify that all factors were accounted for (remaining should be 1)
        mpz_t remaining;
        mpz_init_set(remaining, challenge);
        int total_factors = 0;
        for (int i = 0; i < found.count; i++)
            for (int e = 0; e < found.items[i].exponent; e++)
            {
                mpz_divexact_ui(remaining, remaining, found.items[i].prime);
                total_factors++;
            }

        auto t1 = std::chrono::high_resolution_clock::now();
        total_factor_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (bench == 0)
        {
            printf("\nFactored into %d primes (distinct: %d)\n",
                   total_factors, found.count);
            if (mpz_cmp_ui(remaining, 1) != 0)
            {
                // remaining > 1 means challenge has factors above 2^32
                printf("WARNING: Not fully factored. Remaining: ");
                gmp_printf("%Zd\n", remaining);
            }
        }

        if (mpz_cmp_ui(remaining, 1) == 0)
        {
            auto t2 = std::chrono::high_resolution_clock::now();
            compute_cube_root_from_found(result, &found, modulo);
            auto t3 = std::chrono::high_resolution_clock::now();
            total_root_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
        }

        mpz_clear(remaining);
        found_factors_clear(&found);
    }

    free(sp);

    if (bench_count == 1)
    {
        printf("Factoring time:      %.3f ms\n", total_factor_ms);
        printf("Reconstruction time: %.3f ms\n", total_root_ms);
    }
    else
    {
        printf("\nResults over %d runs:\n", bench_count);
        printf("  Average factoring time:      %.3f ms\n", total_factor_ms / bench_count);
        printf("  Average reconstruction time: %.3f ms\n", total_root_ms / bench_count);
        printf("  Average total:               %.3f ms\n",
               (total_factor_ms + total_root_ms) / bench_count);
    }

    mpz_t challenge_mod;
    mpz_init(challenge_mod);
    mpz_mod(challenge_mod, challenge, modulo);
    verify_cube_root(result, challenge_mod, modulo);

    mpz_clears(challenge, modulo, result, challenge_mod, NULL);
    return 0;
}