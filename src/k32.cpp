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

// ---------------------------------------------------------------------------
// Segmented sieve — generates primes in [range_lo, range_hi] without storing
// all primes in memory. Memory usage is O(SEG_SIZE) not O(range size).
// ---------------------------------------------------------------------------

#define SEG_SIZE (1 << 19) // 512K entries per segment — fits in L2 cache

// Generate small primes up to sqrt(2^32) = 65536 needed to sieve segments
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
// Found factor list — (prime, exponent) pairs
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

static void found_factors_append(found_factors_t *f, unsigned long prime, int exponent)
{
    if (f->count == f->capacity)
    {
        int new_cap = (f->capacity == 0) ? 64 : f->capacity * 2;
        f->items = (factor_t *)realloc(f->items, (size_t)new_cap * sizeof(factor_t));
        f->capacity = new_cap;
    }
    f->items[f->count].prime = prime;
    f->items[f->count].exponent = exponent;
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
// Trial division using segmented sieve
// Fills found_factors with all primes in [range_lo, range_hi] that divide n
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

    // each thread gets its own found-factors list to avoid contention
    found_factors_t *thread_found = (found_factors_t *)malloc(
        (size_t)nthreads * sizeof(found_factors_t));
    for (int t = 0; t < nthreads; t++)
        found_factors_init(&thread_found[t]);

    // divide range evenly across threads
    uint64_t range_size = range_hi - range_lo + 1;

#ifdef _OPENMP
#pragma omp parallel
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int t = 0;
        int nt = 1;
#endif
        uint64_t t_lo = range_lo + (range_size * (uint64_t)t) / (uint64_t)nt;
        uint64_t t_hi = range_lo + (range_size * (uint64_t)(t + 1)) / (uint64_t)nt - 1;
        if (t == nt - 1)
            t_hi = range_hi;

        char *seg = (char *)malloc(SEG_SIZE);

        for (uint64_t lo = t_lo; lo <= t_hi; lo += SEG_SIZE)
        {
            uint64_t hi = lo + SEG_SIZE - 1;
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

            for (uint64_t candidate = lo; candidate <= hi; candidate++)
            {
                if (!seg[candidate - lo])
                    continue;

                // replace the byte loop with this
                unsigned long rem = mpz_tdiv_ui(n, (unsigned long)candidate);
                if (rem != 0)
                    continue;

                int exp = count_exponent(n, (unsigned long)candidate);
                if (exp > 0)
                    found_factors_append(&thread_found[t],
                                         (unsigned long)candidate, exp);
            }
        }

        free(seg);
    }

    // merge thread-local results into main found list
    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < thread_found[t].count; i++)
            found_factors_append(found,
                                 thread_found[t].items[i].prime,
                                 thread_found[t].items[i].exponent);
        found_factors_clear(&thread_found[t]);
    }
    free(thread_found);
}

// ---------------------------------------------------------------------------
// Cube root reconstruction from (prime, exponent) pairs
// No precomputed table — compute powm on-the-fly for each found factor
// This is cheap because there are only ~N_bits/32 factors
// ---------------------------------------------------------------------------

static void compute_cube_root_from_found(
    mpz_t result,
    const found_factors_t *found,
    const mpz_t modulo)
{
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);

    int n = found->count;
    mpz_t *cbrt_parts = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    for (int i = 0; i < n; i++)
        mpz_init(cbrt_parts[i]);

    // all powm calls are independent — parallelize across threads
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++)
    {
        mpz_t p_mpz, cbrt_p;
        mpz_inits(p_mpz, cbrt_p, NULL);
        mpz_set_ui(p_mpz, found->items[i].prime);
        mpz_powm(cbrt_p, p_mpz, e, modulo);

        if (found->items[i].exponent == 1)
            mpz_set(cbrt_parts[i], cbrt_p);
        else
            mpz_powm_ui(cbrt_parts[i], cbrt_p,
                        (unsigned long)found->items[i].exponent, modulo);

        mpz_clears(p_mpz, cbrt_p, NULL);
    }

    // sequential combine — only n multiplications
    mpz_set_ui(result, 1);
    for (int i = 0; i < n; i++)
    {
        mpz_mul(result, result, cbrt_parts[i]);
        mpz_mod(result, result, modulo);
        mpz_clear(cbrt_parts[i]);
    }
    free(cbrt_parts);

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
                fprintf(stderr, "Missing value for --bench\n");
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
// warm up thread pool
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    // --- setup: small primes for sieving (up to sqrt(2^32) = 65536) ---
    printf("Generating small primes for sieve...\n");
    int sp_count;
    unsigned *sp = generate_small_primes(&sp_count);
    printf("Generated %d small primes up to 65536.\n", sp_count);

    size_t challenge_bits = mpz_sizeinbase(challenge, 2);
    printf("Challenge: %zu bits\n", challenge_bits);
    printf("~%zu on-the-fly powm calls expected\n", challenge_bits / 32);

    // --- benchmark loop ---
    double total_factor_ms = 0.0;
    double total_root_ms = 0.0;

    for (int bench = 0; bench < bench_count; bench++)
    {
        found_factors_t found;
        found_factors_init(&found);

        auto factor_start = std::chrono::high_resolution_clock::now();

        // trial division over [65537, 2^32] using segmented sieve
        trial_divide_range(challenge,
                           65537, UINT32_MAX,
                           sp_count, sp,
                           &found);

        // verify full factorization
        mpz_t remaining;
        mpz_init_set(remaining, challenge);
        int total_factors = 0;
        for (int i = 0; i < found.count; i++)
            for (int e = 0; e < found.items[i].exponent; e++)
            {
                mpz_divexact_ui(remaining, remaining, found.items[i].prime);
                total_factors++;
            }

        auto factor_end = std::chrono::high_resolution_clock::now();
        total_factor_ms += std::chrono::duration<double, std::milli>(factor_end - factor_start).count();

        if (bench == 0)
        {
            printf("\nFactored into %d primes (distinct: %d)\n",
                   total_factors, found.count);
            if (mpz_cmp_ui(remaining, 1) != 0)
            {
                printf("WARNING: Not fully factored. Remaining: ");
                gmp_printf("%Zd\n", remaining);
            }
        }

        if (mpz_cmp_ui(remaining, 1) == 0)
        {
            auto root_start = std::chrono::high_resolution_clock::now();
            compute_cube_root_from_found(result, &found, modulo);
            auto root_end = std::chrono::high_resolution_clock::now();
            total_root_ms += std::chrono::duration<double, std::milli>(root_end - root_start).count();
        }

        mpz_clear(remaining);
        found_factors_clear(&found);
    }

    free(sp);

    // --- report ---
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

    // --- verification ---
    mpz_t challenge_mod;
    mpz_init(challenge_mod);
    mpz_mod(challenge_mod, challenge, modulo);
    verify_cube_root(result, challenge_mod, modulo);

    // --- cleanup ---
    mpz_clears(challenge, modulo, result, challenge_mod, NULL);

    return 0;
}