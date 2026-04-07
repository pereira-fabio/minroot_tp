#include "k_common.h"

#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Found factor list — (prime, exponent) pairs
// Identical structure to k32 but uses mpz_t for primes (64-bit don't fit
// in unsigned long on all platforms)
// ---------------------------------------------------------------------------

typedef struct {
    mpz_t prime;
    int   exponent;
} factor64_t;

typedef struct {
    factor64_t *items;
    int         count;
    int         capacity;
} factor64_list_t;

static void f64_list_init(factor64_list_t *f)
    { f->items = NULL; f->count = 0; f->capacity = 0; }

static void f64_list_append(factor64_list_t *f, const mpz_t prime, int exponent)
{
    if (f->count == f->capacity)
    {
        int nc = (f->capacity == 0) ? 16 : f->capacity * 2;
        f->items = (factor64_t *)realloc(f->items,
                       (size_t)nc * sizeof(factor64_t));
        f->capacity = nc;
    }
    mpz_init_set(f->items[f->count].prime, prime);
    f->items[f->count].exponent = exponent;
    f->count++;
}

static void f64_list_free(factor64_list_t *f)
{
    for (int i = 0; i < f->count; i++) mpz_clear(f->items[i].prime);
    free(f->items);
    f->items = NULL; f->count = 0; f->capacity = 0;
}

// ---------------------------------------------------------------------------
// Pollard's rho — finds a non-trivial factor of n using random parameter c
//
// Uses Brent's improvement (faster cycle detection than Floyd's):
//   - maintains r (power of 2 cycle length estimate)
//   - batches gcd calls with a product accumulator for speed
//
// Returns true and sets factor if a factor is found.
// Returns false if c leads to a trivial cycle (try different c).
//
// The random parameter c is the "r" your supervisor refers to — different
// values of c give independent random walks through Z/nZ, so parallel
// threads each try a different c and race to find a factor.
// ---------------------------------------------------------------------------

static bool pollard_rho(mpz_t factor, const mpz_t n, const mpz_t c)
{
    // trivial checks
    if (mpz_cmp_ui(n, 1) == 0) return false;
    if (mpz_divisible_ui_p(n, 2)) { mpz_set_ui(factor, 2); return true; }

    mpz_t x, y, q, diff, g;
    mpz_inits(x, y, q, diff, g, NULL);

    // start: x = y = 2
    mpz_set_ui(x, 2);
    mpz_set_ui(y, 2);
    mpz_set_ui(q, 1);

    // f(x) = x² + c mod n
    // Brent's cycle detection with batch gcd
    long r     = 1;      // current cycle length estimate
    long m     = 128;    // batch size for gcd accumulation
    long count = 0;

    mpz_set_ui(g, 1);

    while (mpz_cmp_ui(g, 1) == 0)
    {
        mpz_set(x, y);
        for (long i = 0; i < r; i++)
        {
            // y = y² + c mod n
            mpz_mul(y, y, y);
            mpz_add(y, y, c);
            mpz_mod(y, y, n);
        }

        long k = 0;
        while (k < r && mpz_cmp_ui(g, 1) == 0)
        {
            mpz_set(q, q);   // keep accumulator
            for (long i = 0; i < m && i < r - k; i++)
            {
                // y = y² + c mod n
                mpz_mul(y, y, y);
                mpz_add(y, y, c);
                mpz_mod(y, y, n);

                // accumulate |x - y| into product q
                mpz_sub(diff, x, y);
                mpz_abs(diff, diff);
                mpz_mul(q, q, diff);
                mpz_mod(q, q, n);
            }
            mpz_gcd(g, q, n);
            k += m;
            count++;

            // safety valve — if we've iterated too long, give up
            if (count > 1000000)
            {
                mpz_clears(x, y, q, diff, g, NULL);
                return false;
            }
        }
        r *= 2;
    }

    // g = n means we hit a cycle with no factor — bad c, retry
    if (mpz_cmp(g, n) == 0)
    {
        mpz_clears(x, y, q, diff, g, NULL);
        return false;
    }

    mpz_set(factor, g);
    mpz_clears(x, y, q, diff, g, NULL);
    return true;
}

// ---------------------------------------------------------------------------
// Miller-Rabin primality test
// Returns true if n is (probably) prime, false if definitely composite.
// Uses deterministic witnesses sufficient for n < 2^64.
// ---------------------------------------------------------------------------

static bool is_prime(const mpz_t n)
{
    // GMP's built-in Miller-Rabin with 15 rounds is sufficient for our use
    // (probability of false positive is negligible for 64-bit numbers)
    return mpz_probab_prime_p(n, 15) > 0;
}

// ---------------------------------------------------------------------------
// Fully factor n into prime factors using Pollard's rho recursively.
// Adds all prime factors (with exponents) to the factor list.
// Uses OpenMP to try different random c values in parallel.
//
// The parallelism here is at the factor-finding level: multiple threads
// each try a different random c value and race to find a factor.
// When one succeeds, others stop via a shared flag.
// ---------------------------------------------------------------------------

static void factor_recursive(const mpz_t n, factor64_list_t *factors,
                              unsigned long seed_offset)
{
    // base cases
    if (mpz_cmp_ui(n, 1) == 0) return;

    if (is_prime(n))
    {
        // n is prime — count its exponent in the original challenge
        // (we pass n directly so exponent is always 1 here; exponents
        //  are handled by the caller via repeated division)
        #pragma omp critical
        f64_list_append(factors, n, 1);
        return;
    }

    // n is composite — find a factor using parallel Pollard's rho
    // each thread tries a different value of c (the random parameter)
    mpz_t found_factor;
    mpz_init(found_factor);
    int success = 0;

    #ifdef _OPENMP
    #pragma omp parallel shared(found_factor, success)
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
    #else
    {
        int tid = 0, nt = 1;
    #endif

        mpz_t c, local_factor;
        mpz_inits(c, local_factor, NULL);

        // each thread starts with a different c value
        // c ranges: thread 0 tries c=1,nt+1,2nt+1,...
        //           thread 1 tries c=2,nt+2,2nt+2,...
        // this ensures no two threads duplicate work
        unsigned long c_val = (unsigned long)(tid + 1) + seed_offset;

        while (!success)
        {
            mpz_set_ui(c, c_val);

            // skip c = 0 and c = n-2 (known bad values for Pollard's rho)
            mpz_t bad;
            mpz_init(bad);
            mpz_sub_ui(bad, n, 2);
            if (mpz_cmp_ui(c, 0) == 0 || mpz_cmp(c, bad) == 0)
            {
                c_val += (unsigned long)nt;
                mpz_clear(bad);
                continue;
            }
            mpz_clear(bad);

            if (pollard_rho(local_factor, n, c))
            {
                #pragma omp critical
                {
                    if (!success)
                    {
                        mpz_set(found_factor, local_factor);
                        success = 1;
                    }
                }
            }

            c_val += (unsigned long)nt;

            // limit search per thread
            if (c_val > seed_offset + 100000UL * (unsigned long)nt) break;
        }

        mpz_clears(c, local_factor, NULL);
    }

    if (!success)
    {
        // fallback: try small trial division before giving up
        // (handles cases where Pollard's rho struggles with small factors)
        for (unsigned long p = 2; p < 1000; p++)
        {
            if (mpz_divisible_ui_p(n, p))
            {
                mpz_set_ui(found_factor, p);
                success = 1;
                break;
            }
        }
    }

    if (success)
    {
        // recursively factor both halves
        mpz_t cofactor;
        mpz_init(cofactor);
        mpz_divexact(cofactor, n, found_factor);

        // recurse on the two parts with different seed offsets
        factor_recursive(found_factor, factors, seed_offset + 7919);
        factor_recursive(cofactor,     factors, seed_offset + 104729);

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
// Factor challenge completely, collecting (prime, exponent) pairs
// ---------------------------------------------------------------------------

static void factor_challenge(const mpz_t challenge, factor64_list_t *factors)
{
    // work on a copy so we can divide out found factors
    mpz_t remaining;
    mpz_init_set(remaining, challenge);

    // collect all prime factors (with multiplicity) via recursive Pollard's rho
    factor64_list_t raw_factors;
    f64_list_init(&raw_factors);
    factor_recursive(remaining, &raw_factors, 1);

    // sort and consolidate: count exponents for repeated primes
    // raw_factors may have duplicates if the same prime appears multiple times
    // simple O(n²) consolidation — n is small (~16 for 1024-bit challenge)
    for (int i = 0; i < raw_factors.count; i++)
    {
        if (mpz_cmp_ui(raw_factors.items[i].prime, 0) == 0) continue; // already merged

        int exp = 1;
        for (int j = i + 1; j < raw_factors.count; j++)
        {
            if (mpz_cmp(raw_factors.items[i].prime,
                        raw_factors.items[j].prime) == 0)
            {
                exp++;
                mpz_set_ui(raw_factors.items[j].prime, 0); // mark as merged
            }
        }
        f64_list_append(factors, raw_factors.items[i].prime, exp);
    }

    f64_list_free(&raw_factors);
    mpz_clear(remaining);
}

// ---------------------------------------------------------------------------
// Cube root reconstruction — identical logic to k32 but uses mpz_t primes
// ---------------------------------------------------------------------------

static void reconstruct_cube_root(
    mpz_t                   result,
    const factor64_list_t  *factors,
    const mpz_t             modulo)
{
    // e = 3^-1 mod (modulo - 1)
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);

    int n = factors->count;
    mpz_t *parts = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    for (int i = 0; i < n; i++) mpz_init(parts[i]);

    // all powm calls independent — parallelize
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++)
    {
        mpz_t cbrt_p;
        mpz_init(cbrt_p);
        mpz_powm(cbrt_p, factors->items[i].prime, e, modulo);

        if (factors->items[i].exponent == 1)
            mpz_set(parts[i], cbrt_p);
        else
            mpz_powm_ui(parts[i], cbrt_p,
                        (unsigned long)factors->items[i].exponent, modulo);

        mpz_clear(cbrt_p);
    }

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
    const char *challenge_file = "data/challenges/k64/challenge_64_1024.txt";
    const char *modulo_file    = "data/challenges/k64/modulo_64_1024.txt";
    int bench_count            = 1;

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
        if (positional_index == 0)      challenge_file = argv[argi];
        else if (positional_index == 1) modulo_file    = argv[argi];
        else
        {
            fprintf(stderr, "Unexpected argument '%s'\n", argv[argi]);
            return 1;
        }
        positional_index++;
    }

    printf("Challenge file: %s\n", challenge_file);
    printf("Modulo file:    %s\n", modulo_file);
    if (bench_count > 1) printf("Benchmark mode: %d runs\n", bench_count);

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);
    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);

    mpz_t result;
    mpz_init(result);

#ifdef _OPENMP
    printf("OpenMP enabled with %d threads available.\n", omp_get_max_threads());
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++) (void)i;
#endif

    printf("Challenge: %zu bits\n", mpz_sizeinbase(challenge, 2));
    printf("Estimated factors: ~%zu (assuming 64-bit primes)\n",
           mpz_sizeinbase(challenge, 2) / 64);

    double total_factor_ms = 0.0;
    double total_root_ms   = 0.0;

    for (int bench = 0; bench < bench_count; bench++)
    {
        factor64_list_t factors;
        f64_list_init(&factors);

        auto t0 = std::chrono::high_resolution_clock::now();
        factor_challenge(challenge, &factors);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_factor_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (bench == 0)
        {
            printf("\nFactored into %d distinct primes:\n", factors.count);
            for (int i = 0; i < factors.count; i++)
            {
                gmp_printf("  p%d = %Zd", i, factors.items[i].prime);
                if (factors.items[i].exponent > 1)
                    printf("^%d", factors.items[i].exponent);
                printf("  (%zu bits)\n",
                       mpz_sizeinbase(factors.items[i].prime, 2));
            }

            // verify full factorization
            mpz_t check;
            mpz_init_set_ui(check, 1);
            for (int i = 0; i < factors.count; i++)
                for (int e = 0; e < factors.items[i].exponent; e++)
                    mpz_mul(check, check, factors.items[i].prime);
            if (mpz_cmp(check, challenge) == 0)
                printf("Factorization verified: product matches challenge.\n");
            else
                printf("WARNING: product does NOT match challenge!\n");
            mpz_clear(check);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        reconstruct_cube_root(result, &factors, modulo);
        auto t3 = std::chrono::high_resolution_clock::now();
        total_root_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        f64_list_free(&factors);
    }

    if (bench_count == 1)
    {
        printf("\nFactoring time:      %.3f ms\n", total_factor_ms);
        printf("Reconstruction time: %.3f ms\n", total_root_ms);
        printf("Total time:          %.3f ms\n",
               total_factor_ms + total_root_ms);
    }
    else
    {
        printf("\nResults over %d runs:\n", bench_count);
        printf("  Average factoring time:      %.3f ms\n",
               total_factor_ms / bench_count);
        printf("  Average reconstruction time: %.3f ms\n",
               total_root_ms   / bench_count);
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