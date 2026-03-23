#include "k_common.h"

#include <climits>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char *argv[])
{

    enum solver_method_t
    {
        METHOD_FACTOR,
        METHOD_POWM,
    };

    mpz_t exponent; 
    mpz_init_set_ui(exponent, 3);

    const char *challenge_file = "data/challenges/k16/challenge_2_16_2048.txt";
    const char *modulo_file = "data/challenges/k16/modulo_16_2048.txt";
    solver_method_t method = METHOD_FACTOR;

    printf("Challenge file: %s\n", challenge_file);
    printf("Modulo file: %s\n", modulo_file);
    printf("Method: %s\n", (method == METHOD_POWM) ? "powm" : "factor");

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);

    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);

    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--method") == 0)
        {
            if (argi + 1 >= argc)
            {
                fprintf(stderr, "Error: Missing value for --method (use factor or powm)\n");
                return 1;
            }

            if (strcmp(argv[argi + 1], "factor") == 0)
            {
                method = METHOD_FACTOR;
            }
            else if (strcmp(argv[argi + 1], "powm") == 0)
            {
                method = METHOD_POWM;
            }
            else
            {
                fprintf(stderr, "Error: Invalid --method value '%s' (use factor or powm)\n", argv[argi + 1]);
                return 1;
            }

            argi++;
            continue;
        }
    }
    
    mpz_t result;
    mpz_init(result);
    
#ifdef _OPENMP
    printf("OpenMP enabled with %d threads available.\n", omp_get_max_threads());
#endif

    if (method == METHOD_POWM)
    {
        printf("\nComputing cube root using direct modular exponentiation...\n");
        auto start_time = std::chrono::high_resolution_clock::now();
        
        compute_cube_root_mod_prime(result, challenge, modulo);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        printf("Execution time: %.3f ms\n", duration.count());
    }
    else
    {
        printf("\nComputing cube root using factorization...\n");
        printf("Generating all primes up to 16 bits...\n");
        int prime_count;
        unsigned long *primes = generate_primes_up_to(UINT16_MAX, &prime_count);
        printf("Generated %d primes.\n", prime_count);

        factor_list_t factors;
        factor_list_init(&factors);

        mpz_t remaining;
        mpz_init_set(remaining, challenge);

        auto factor_start = std::chrono::high_resolution_clock::now();

        int *exponents = (int *)calloc(prime_count, sizeof(int));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < prime_count; i++)
        {
            exponents[i] = count_exponent(challenge, primes[i]);
        }

        int total_factors = 0;
        for (int i = 0; i < prime_count; i++)
        {
            if (exponents[i] > 0)
            {
                unsigned long p = primes[i];
                for (int e = 0; e < exponents[i]; e++)
                {
                    mpz_t f;
                    mpz_init_set_ui(f, p);
                    factor_list_append(&factors, f);
                    mpz_clear(f);
                    mpz_divexact_ui(remaining, remaining, p);
                    total_factors++;
                }
            }
        }
        free(exponents);
        
        auto factor_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> factor_duration = factor_end - factor_start;

        if (mpz_cmp_ui(remaining, 1) == 0)
        {
            printf("SUCCESS: Fully factored challenge into %d primes in %.3f ms!\n", total_factors, factor_duration.count());
            
            auto root_start = std::chrono::high_resolution_clock::now();
            compute_cube_root_from_factors(result, &factors, modulo);
            auto root_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> root_duration = root_end - root_start;
            
            printf("Cube root recovery time: %.3f ms\n", root_duration.count());
        }
        else
        {
            printf("WARNING: Not fully factored. Remaining: ");
            gmp_printf("%Zd\n", remaining);
        }

        mpz_clear(remaining);
        factor_list_clear(&factors);
        free(primes);
    }

    mpz_t challenge_mod;
    mpz_init(challenge_mod);
    mpz_mod(challenge_mod, challenge, modulo);

    verify_cube_root(result, challenge_mod, modulo);

    mpz_clears(challenge, modulo, exponent, result, challenge_mod, NULL);
    
    return 0;
}