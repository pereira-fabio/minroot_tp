#include "k_common.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void read_mpz_from_file(mpz_t num, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    if (mpz_inp_str(num, f, 10) == 0)
    {
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

    for (unsigned long i = 2; i * i <= limit; i++)
    {
        if (sieve[i])
        {
            for (unsigned long j = i * i; j <= limit; j += i)
            {
                sieve[j] = 0;
            }
        }
    }

    *count = 0;
    for (unsigned long i = 2; i <= limit; i++)
    {
        if (sieve[i])
        {
            (*count)++;
        }
    }

    unsigned long *primes = (unsigned long *)malloc((size_t)(*count) * sizeof(unsigned long));
    if (!primes)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(sieve);
        exit(1);
    }

    int idx = 0;
    for (unsigned long i = 2; i <= limit; i++)
    {
        if (sieve[i])
        {
            primes[idx++] = i;
        }
    }

    free(sieve);
    return primes;
}

bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo)
{
    mpz_t phi, inverse, three;
    mpz_inits(phi, inverse, three, NULL);

    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);

    if (mpz_invert(inverse, three, phi) == 0)
    {
        mpz_set_ui(result, 0);
        mpz_clears(phi, inverse, three, NULL);
        return false;
    }

    mpz_powm(result, value, inverse, modulo);
    mpz_clears(phi, inverse, three, NULL);
    return true;
}

void precompute_cbrt_table(mpz_t *cbrt_table, const unsigned long *primes, int prime_count, const mpz_t modulo)
{
    // compute e = inverse(3, modulo - 1) once — reused by all threads
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);

    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi); // e = 3^-1 mod (modulo-1)

// e and modulo are read-only from here — safe to share across threads
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < prime_count; i++)
    {
        mpz_t prime_mpz; // declared inside loop = private per thread
        mpz_init_set_ui(prime_mpz, primes[i]);
        mpz_init(cbrt_table[i]);
        mpz_powm(cbrt_table[i], prime_mpz, e, modulo);
        mpz_clear(prime_mpz);
    }

    mpz_clears(phi, e, three, NULL);
}

void compute_cube_root_from_factors(mpz_t result, const int *exponents,
                                    const mpz_t *cbrt_table, int prime_count,
                                    const mpz_t modulo)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    // threshold: reduce mod when accumulator exceeds 4× modulus bit size
    size_t mod_bits = mpz_sizeinbase(modulo, 2);
    size_t reduce_threshold = 4 * mod_bits;

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
            mpz_mul(partials[t], partials[t], cbrt_table[i]);
        }
        else
        {
            mpz_t powered;
            mpz_init(powered);
            mpz_powm_ui(powered, cbrt_table[i], exponents[i], modulo);
            mpz_mul(partials[t], partials[t], powered);
            mpz_clear(powered);
        }

        // only reduce when accumulator grows large — much cheaper than reducing every step
        if (mpz_sizeinbase(partials[t], 2) > reduce_threshold)
            mpz_mod(partials[t], partials[t], modulo);
    }

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

void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod, const mpz_t modulo)
{
    mpz_t cubed;
    mpz_init(cubed);

    mpz_powm_ui(cubed, candidate_cuberoot, 3, modulo);

    if (mpz_cmp(cubed, challenge_mod) == 0)
    {
        printf("SUCCESS: Cube root is correct!\n");
    }
    else
    {
        printf("FAILURE: Cube root is incorrect\n");
    }

    mpz_clear(cubed);
}

int count_exponent(const mpz_t n, unsigned long prime)
{
    // fast early exit — no allocation needed if prime doesn't divide n
    if (mpz_tdiv_ui(n, prime) != 0)
        return 0;

    mpz_t temp;
    mpz_init_set(temp, n);

    int exponent = 0;
    while (mpz_divisible_ui_p(temp, prime))
    {
        mpz_divexact_ui(temp, temp, prime);
        exponent++;
    }

    mpz_clear(temp);
    return exponent;
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
        return false; // file doesn't exist yet — caller must compute it

    int stored_count;
    if (fscanf(f, "%d\n", &stored_count) != 1 || stored_count != prime_count)
    {
        fclose(f);
        return false; // stale or mismatched table — recompute
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