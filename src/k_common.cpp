#include "k_common.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    // compute e = inverse(3, modulo - 1) once — same exponent for every prime
    mpz_t phi, e, three, prime_mpz;
    mpz_inits(phi, e, three, prime_mpz, NULL);

    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);   // e = 3^-1 mod (p-1)

    for (int i = 0; i < prime_count; i++)
    {
        mpz_init(cbrt_table[i]);
        mpz_set_ui(prime_mpz, primes[i]);
        mpz_powm(cbrt_table[i], prime_mpz, e, modulo);   // cbrt_table[i] = primes[i]^e mod M
    }

    mpz_clears(phi, e, three, prime_mpz, NULL);
}

void compute_cube_root_from_factors(mpz_t result, const int *exponents, const mpz_t *cbrt_table, int prime_count, const mpz_t modulo)
{
    mpz_set_ui(result, 1);

    for (int i = 0; i < prime_count; i++)
    {
        if (exponents[i] == 0) continue;

        // multiply cbrt_table[i] in exponents[i] times
        for (int e = 0; e < exponents[i]; e++)
        {
            mpz_mul(result, result, cbrt_table[i]);
            mpz_mod(result, result, modulo);
        }
    }
}

void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod, const mpz_t modulo)
{
    mpz_t cubed;
    mpz_init(cubed);

    mpz_powm_ui(cubed, candidate_cuberoot, 3, modulo);

    printf("\nVerification:\n");
    gmp_printf("Candidate^3 mod modulo = %Zd\n", cubed);
    gmp_printf("Original challenge     = %Zd\n", challenge_mod);

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