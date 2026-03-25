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

    const char *challenge_file = "data/challenges/k16/challenge_2_16_2048.txt";
    const char *modulo_file    = "data/challenges/k16/modulo_16_2048.txt";
    solver_method_t method     = METHOD_FACTOR;
    int bench_count            = 1;  

    int positional_index = 0;
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
                method = METHOD_FACTOR;
            else if (strcmp(argv[argi + 1], "powm") == 0)
                method = METHOD_POWM;
            else
            {
                fprintf(stderr, "Error: Invalid --method value '%s' (use factor or powm)\n", argv[argi + 1]);
                return 1;
            }
            argi++;
            continue;
        }

        if (strcmp(argv[argi], "--bench") == 0)
        {
            if (argi + 1 >= argc)
            {
                fprintf(stderr, "Error: Missing value for --bench\n");
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
            fprintf(stderr, "Error: Unexpected argument '%s'\n", argv[argi]);
            return 1;
        }
        positional_index++;
    }

    printf("Challenge file: %s\n", challenge_file);
    printf("Modulo file: %s\n", modulo_file);
    printf("Method: %s\n", (method == METHOD_POWM) ? "powm" : "factor");
    if (bench_count > 1)
        printf("Benchmark mode: %d runs\n", bench_count);

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);

    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);

    mpz_t result;
    mpz_init(result);

    mpz_t *cbrt_table  = nullptr;
    int    prime_count = 0;
    unsigned long *primes = nullptr;

#ifdef _OPENMP
    printf("OpenMP enabled with %d threads available.\n", omp_get_max_threads());
    // warm up the thread pool once before any timing starts
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
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
        primes = generate_primes_up_to(UINT16_MAX, &prime_count);
        printf("Generated %d primes.\n", prime_count);

        cbrt_table = (mpz_t *)malloc((size_t)prime_count * sizeof(mpz_t));

        // derive table filename from modulo filename
        // e.g. "modulo_16_2048.txt" -> "modulo_16_2048.cbrt"
        char table_file[512];
        strncpy(table_file, modulo_file, sizeof(table_file) - 1);
        table_file[sizeof(table_file) - 1] = '\0';
        char *dot = strrchr(table_file, '.');
        if (dot) *dot = '\0';
        strncat(table_file, ".cbrt", sizeof(table_file) - strlen(table_file) - 1);

        printf("Looking for precomputed table at %s...\n", table_file);
        auto precomp_start = std::chrono::high_resolution_clock::now();

        if (!load_cbrt_table(cbrt_table, prime_count, table_file))
        {
            printf("Not found — computing and saving...\n");
            precompute_cbrt_table(cbrt_table, primes, prime_count, modulo);
            save_cbrt_table(cbrt_table, prime_count, table_file);
        }
        else
        {
            printf("Loaded from disk.\n");
        }

        auto precomp_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> precomp_duration = precomp_end - precomp_start;
        printf("Precomputation/load done in %.3f ms\n", precomp_duration.count());

        // export challenge bytes once — reused across all bench iterations
        size_t nbytes;
        unsigned char *challenge_bytes = (unsigned char *)mpz_export(
            NULL, &nbytes, 1, 1, 0, 0, challenge);

        // --- benchmark loop ---
        double total_factor_ms = 0.0;
        double total_root_ms   = 0.0;
        int last_total_factors = 0;

        for (int bench = 0; bench < bench_count; bench++)
        {
            int *exponents = (int *)calloc(prime_count, sizeof(int));

            auto factor_start = std::chrono::high_resolution_clock::now();

            // parallel trial division using native byte arithmetic —
            // no GMP calls in the hot path for the 99%+ of primes that don't divide
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < prime_count; i++)
            {
                unsigned long rem = 0;
                for (size_t b = 0; b < nbytes; b++)
                    rem = (rem * 256 + challenge_bytes[b]) % primes[i];

                if (rem != 0) { exponents[i] = 0; continue; }

                // prime divides — count exact exponent with GMP
                exponents[i] = count_exponent(challenge, primes[i]);
            }

            // verify full factorization and count factors
            mpz_t remaining;
            mpz_init_set(remaining, challenge);
            int total_factors = 0;
            for (int i = 0; i < prime_count; i++)
            {
                if (exponents[i] > 0)
                {
                    for (int e = 0; e < exponents[i]; e++)
                    {
                        mpz_divexact_ui(remaining, remaining, primes[i]);
                        total_factors++;
                    }
                    if (mpz_cmp_ui(remaining, 1) == 0) break;
                }
            }

            auto factor_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> factor_duration = factor_end - factor_start;
            total_factor_ms += factor_duration.count();
            last_total_factors = total_factors;

            if (mpz_cmp_ui(remaining, 1) == 0)
            {
                auto root_start = std::chrono::high_resolution_clock::now();
                compute_cube_root_from_factors(result, exponents, cbrt_table, prime_count, modulo);
                auto root_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> root_duration = root_end - root_start;
                total_root_ms += root_duration.count();
            }
            else if (bench == 0)
            {
                printf("WARNING: Not fully factored. Remaining: ");
                gmp_printf("%Zd\n", remaining);
            }

            mpz_clear(remaining);
            free(exponents);
        }
        // --- end benchmark loop ---

        free(challenge_bytes);

        if (bench_count == 1)
        {
            printf("SUCCESS: Fully factored challenge into %d primes in %.3f ms!\n",
                   last_total_factors, total_factor_ms);
            printf("Cube root recovery time: %.3f ms\n", total_root_ms);
        }
        else
        {
            printf("\nResults over %d runs:\n", bench_count);
            printf("  Average factoring time:      %.3f ms\n", total_factor_ms / bench_count);
            printf("  Average reconstruction time: %.3f ms\n", total_root_ms   / bench_count);
            printf("  Average total:               %.3f ms\n", (total_factor_ms + total_root_ms) / bench_count);
        }
    }

    mpz_t challenge_mod;
    mpz_init(challenge_mod);
    mpz_mod(challenge_mod, challenge, modulo);

    verify_cube_root(result, challenge_mod, modulo);

    mpz_clears(challenge, modulo, result, challenge_mod, NULL);
    if (cbrt_table)
    {
        for (int i = 0; i < prime_count; i++)
            mpz_clear(cbrt_table[i]);
        free(cbrt_table);
    }
    if (primes)
        free(primes);

    return 0;
}