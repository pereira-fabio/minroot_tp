#include "k_common.h"

#include <gmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

void read_mpz_from_file(mpz_t num, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // try decimal text first (e.g. challenge files written as plain integers)
    if (mpz_inp_str(num, f, 10) == 0)
    {
        // fall back to GMP binary format (mpz_out_raw / mpz_inp_raw)
        rewind(f);
        if (mpz_inp_raw(num, f) == 0)
        {
            fprintf(stderr, "Error: Failed to read number from %s\n", filename);
            fclose(f);
            exit(1);
        }
    }
    fclose(f);
    printf("Successfully read number from %s\n", filename);
}

// ---------------------------------------------------------------------------
// Prime generation — Sieve of Eratosthenes
// ---------------------------------------------------------------------------

unsigned long *generate_primes_up_to(unsigned long limit, int *count)
{
    char *sieve = (char *)malloc(limit + 1);
    if (!sieve)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    memset(sieve, 1, limit + 1);
    sieve[0] = sieve[1] = 0;

    // mark composites: for each prime p, mark p², p²+p, p²+2p, ...
    for (unsigned long i = 2; i * i <= limit; i++)
        if (sieve[i])
            for (unsigned long j = i * i; j <= limit; j += i)
                sieve[j] = 0;

    *count = 0;
    for (unsigned long i = 2; i <= limit; i++)
        if (sieve[i])
            (*count)++;

    unsigned long *primes = (unsigned long *)malloc(
        (size_t)(*count) * sizeof(unsigned long));
    if (!primes)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(sieve);
        exit(1);
    }

    int idx = 0;
    for (unsigned long i = 2; i <= limit; i++)
        if (sieve[i])
            primes[idx++] = i;

    free(sieve);
    return primes;
}

// ---------------------------------------------------------------------------
// Cube root via Fermat's little theorem — naive baseline
// ---------------------------------------------------------------------------

bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo)
{
    mpz_t phi, inverse, three;
    mpz_inits(phi, inverse, three, NULL);

    // phi = modulo - 1  (Euler's totient for a prime)
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);

    // compute e = 3^-1 mod phi  (exists iff gcd(3, phi) = 1)
    if (mpz_invert(inverse, three, phi) == 0)
    {
        // cube root does not exist for this modulo
        mpz_set_ui(result, 0);
        mpz_clears(phi, inverse, three, NULL);
        return false;
    }

    // result = value^e mod modulo — this is the cube root
    mpz_powm(result, value, inverse, modulo);
    mpz_clears(phi, inverse, three, NULL);
    return true;
}

// ---------------------------------------------------------------------------
// Precomputed cube root table — k16 optimization
//
// Instead of computing x^e mod M once for the full challenge, we precompute
// p^e mod M for each of the 6542 primes up to 2^16. Then reconstruction is
// just table lookups + multiplications — no large exponentiations per challenge.
// ---------------------------------------------------------------------------

void precompute_cbrt_table(mpz_t *cbrt_table, const unsigned long *primes,
                           int prime_count, const mpz_t modulo)
{
    // e = 3^-1 mod (modulo-1): computed ONCE and shared across all threads
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);

    // each iteration is fully independent — safe to parallelize
    // prime_mpz is declared inside the loop so each thread has its own copy
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < prime_count; i++)
    {
        mpz_t prime_mpz; // thread-private
        mpz_init_set_ui(prime_mpz, primes[i]);
        mpz_init(cbrt_table[i]);
        mpz_powm(cbrt_table[i], prime_mpz, e, modulo);
        mpz_clear(prime_mpz);
    }

    mpz_clears(phi, e, three, NULL);
}

void save_cbrt_table(const mpz_t *cbrt_table, int prime_count, const char *filename)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Error: Cannot write table to %s\n", filename);
        exit(1);
    }
    fprintf(f, "%d\n", prime_count);
    for (int i = 0; i < prime_count; i++)
    {
        mpz_out_str(f, 10, cbrt_table[i]);
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Saved cbrt table to %s\n", filename);
}

bool load_cbrt_table(mpz_t *cbrt_table, int prime_count, const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f)
        return false; // not yet computed for this modulo

    int stored_count;
    if (fscanf(f, "%d\n", &stored_count) != 1 || stored_count != prime_count)
    {
        fclose(f);
        return false; // stale table (different prime set) — recompute
    }

    for (int i = 0; i < prime_count; i++)
    {
        mpz_init(cbrt_table[i]);
        if (mpz_inp_str(cbrt_table[i], f, 10) == 0)
        {
            for (int j = 0; j <= i; j++)
                mpz_clear(cbrt_table[j]);
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Reconstruction from factorization
//
// By the homomorphism cbrt(a*b) = cbrt(a)*cbrt(b) mod M:
//   cbrt(challenge) = prod cbrt_table[i]^exponents[i]  mod M
//
// Parallelized with OpenMP: each thread accumulates a partial product,
// which are combined sequentially at the end.
//
// Modular reduction is deferred until the accumulator exceeds 4× the modulus
// size — reduces the number of expensive mpz_mod calls significantly.
//
// ---------------------------------------------------------------------------
// Note on mpn_powm vs mpz_powm_ui
//
// The supervisor suggested investigating mpn_powm — a lower-level GMP function
// that operates directly on mp_limb_t arrays without the mpz_t wrapper overhead
// (no internal scratch allocation, no normalization bookkeeping per call).
//
// Investigation result: mpn_powm is declared in <gmp-impl.h>, which is an
// internal GMP header NOT installed with the library (it lives in the GMP
// build tree only). It is therefore not linkable from user code without
// patching the GMP installation.
//
// Decision: keep mpz_powm_ui for the exponents[i] > 1 case.
// mpz_powm_ui is the next best option — it takes an unsigned long exponent
// directly, avoiding the arbitrary-precision exponent path that mpz_powm uses.
// Since our exponents are always small (at most ~20 for typical challenges),
// mpz_powm_ui is both correct and the most efficient available public function.
// ---------------------------------------------------------------------------

void compute_cube_root_from_factors(mpz_t result, const int *exponents,
                                    const mpz_t *cbrt_table, int prime_count,
                                    const mpz_t modulo)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    // reduce mod M only when accumulator grows beyond this threshold
    // avoids O(prime_count) expensive reductions — does ~O(prime_count/log) instead
    size_t reduce_threshold = 4 * mpz_sizeinbase(modulo, 2);

    // each thread accumulates its own partial product starting at 1
    mpz_t *partials = (mpz_t *)malloc((size_t)nthreads * sizeof(mpz_t));
    for (int t = 0; t < nthreads; t++)
        mpz_init_set_ui(partials[t], 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < prime_count; i++)
    {
        if (exponents[i] == 0)
            continue;

        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif

        if (exponents[i] == 1)
        {
            // most common case — multiply precomputed cube root directly
            mpz_mul(partials[t], partials[t], cbrt_table[i]);
        }
        else
        {
            // cbrt(p^k) = cbrt(p)^k
            // mpz_powm_ui takes an unsigned long exponent — avoids the
            // arbitrary-precision exponent path of mpz_powm, which would be
            // wasteful since our exponents are tiny (at most ~20)
            mpz_t powered;
            mpz_init(powered);
            mpz_powm_ui(powered, cbrt_table[i],
                        (unsigned long)exponents[i], modulo);
            mpz_mul(partials[t], partials[t], powered);
            mpz_clear(powered);
        }

        // deferred modular reduction
        if (mpz_sizeinbase(partials[t], 2) > reduce_threshold)
            mpz_mod(partials[t], partials[t], modulo);
    }

    // combine partial results — only nthreads multiplications
    mpz_set_ui(result, 1);
    for (int t = 0; t < nthreads; t++)
    {
        mpz_mod(partials[t], partials[t], modulo);
        mpz_mul(result, result, partials[t]);
        mpz_mod(result, result, modulo);
        mpz_clear(partials[t]);
    }
    free(partials);
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod,
                      const mpz_t modulo)
{
    mpz_t cubed;
    mpz_init(cubed);
    // check: does candidate^3 mod M equal the original challenge mod M?
    mpz_powm_ui(cubed, candidate_cuberoot, 3, modulo);
    if (mpz_cmp(cubed, challenge_mod) == 0)
        printf("SUCCESS: Cube root is correct!\n");
    else
        printf("FAILURE: Cube root is incorrect\n");
    mpz_clear(cubed);
}

// ---------------------------------------------------------------------------
// Factoring helper
// ---------------------------------------------------------------------------

int count_exponent(const mpz_t n, unsigned long prime)
{
    // fast early exit: no GMP allocation if prime doesn't divide n
    if (mpz_tdiv_ui(n, prime) != 0)
        return 0;

    mpz_t temp;
    mpz_init_set(temp, n); // work on a private copy

    int exponent = 0;
    while (mpz_divisible_ui_p(temp, prime))
    {
        mpz_divexact_ui(temp, temp, prime);
        exponent++;
    }
    mpz_clear(temp);
    return exponent;
}

// ---------------------------------------------------------------------------
// Benchmark/reporting helpers
// ---------------------------------------------------------------------------

double compute_stddev_from_values(const double *vals, int n, bool sample_stddev)
{
    if (!vals || n <= 0)
        return 0.0;
    if (n == 1)
        return 0.0;

    double sum = 0.0;
    double sumsq = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += vals[i];
        sumsq += vals[i] * vals[i];
    }

    return compute_stddev_from_moments(sum, sumsq, n, sample_stddev);
}

double compute_stddev_from_moments(double sum, double sumsq, int n,
                                   bool sample_stddev)
{
    if (n <= 0)
        return 0.0;
    if (n == 1)
        return 0.0;

    double denom = sample_stddev ? (double)(n - 1) : (double)n;
    double var = (sumsq - (sum * sum) / (double)n) / denom;
    if (var < 0.0)
    {
        // Clamp tiny negative values caused by floating-point roundoff.
        var = 0.0;
    }
    return sqrt(var);
}

void print_timing_stat_moments(const char *label,
                               double sum, double sumsq, int n,
                               int label_width, bool sample_stddev)
{
    double mean = (n > 0) ? (sum / n) : 0.0;
    double stddev = compute_stddev_from_moments(sum, sumsq, n, sample_stddev);
    printf("    %-*s avg=%9.3f ms   stddev=%7.3f ms\n",
           label_width, label, mean, stddev);
}

void print_timing_total(const char *label, double mean,
                        int label_width, const char *extra_text)
{
    if (extra_text && extra_text[0] != '\0')
    {
        printf("    %-*s avg=%9.3f ms  %s\n",
               label_width, label, mean, extra_text);
    }
    else
    {
        printf("    %-*s avg=%9.3f ms\n", label_width, label, mean);
    }
}