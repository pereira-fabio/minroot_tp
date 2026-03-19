#include "ginza1_common.h"

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

int extract_k_from_filename(const char *path)
{
    const char *p = strstr(path, "challenge_");
    if (!p)
    {
        p = strstr(path, "modulo_");
    }

    if (!p)
    {
        return -1;
    }

    p += 10;
    int k = 0;
    while (*p && isdigit((unsigned char)*p))
    {
        k = k * 10 + (*p - '0');
        p++;
    }

    return (k > 0) ? k : -1;
}

bool parse_unsigned_long_arg(const char *text, unsigned long *value)
{
    if (!text || !*text)
    {
        return false;
    }

    char *end_ptr = NULL;
    unsigned long parsed = strtoul(text, &end_ptr, 10);
    if (*end_ptr != '\0')
    {
        return false;
    }

    *value = parsed;
    return true;
}

double wall_time_seconds()
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
#endif
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

bool compute_cube_root_direct_powm(mpz_t result, const mpz_t challenge, const mpz_t modulo)
{
    // Direct route: r = challenge^(3^{-1} mod (modulo-1)) mod modulo.
    return compute_cube_root_mod_prime(result, challenge, modulo);
}

void compute_cube_root_from_factors(mpz_t result, const factor_list_t *factors, const mpz_t modulo)
{
    mpz_t challenge_mod;
    mpz_init_set_ui(challenge_mod, 1);

    for (int i = 0; i < factors->count; i++)
    {
        mpz_mul(challenge_mod, challenge_mod, factors->items[i]);
        mpz_mod(challenge_mod, challenge_mod, modulo);
    }

    if (!compute_cube_root_mod_prime(result, challenge_mod, modulo))
    {
        fprintf(stderr, "Error: Cannot compute modular inverse of 3 modulo (modulo - 1)\n");
        mpz_set_ui(result, 0);
    }

    mpz_clear(challenge_mod);
}

void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge, const mpz_t modulo)
{
    mpz_t cubed;
    mpz_init(cubed);

    mpz_powm_ui(cubed, candidate_cuberoot, 3, modulo);

    printf("\nVerification:\n");
    gmp_printf("Candidate^3 mod modulo = %Zd\n", cubed);
    gmp_printf("Original challenge       = %Zd\n", challenge);

    if (mpz_cmp(cubed, challenge) == 0)
    {
        printf("SUCCESS: Cube root is correct!\n");
    }
    else
    {
        printf("FAILURE: Cube root is incorrect\n");
    }

    mpz_clear(cubed);
}

void print_factorization_summary(const mpz_t value, const factor_list_t *factors)
{
    printf("\n========================================\n");
    printf("Factoring challenge number:\n");
    gmp_printf("Challenge = %Zd\n", value);
    printf("========================================\n");
    printf("Prime factors found:\n");

    mpz_t product;
    mpz_init_set_ui(product, 1);

    for (int i = 0; i < factors->count; i++)
    {
        printf("  Factor %d: ", i + 1);
        gmp_printf("%Zd\n", factors->items[i]);
        mpz_mul(product, product, factors->items[i]);
    }

    printf("========================================\n");
    if (mpz_cmp(product, value) == 0)
    {
        printf("SUCCESS: Fully factored challenge into %d primes!\n", factors->count);
    }
    else
    {
        printf("WARNING: Product of factors does not match challenge\n");
    }

    mpz_clear(product);
}
