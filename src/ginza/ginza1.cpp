#include <gmp.h>
#include <stdio.h>
#include <string.h>

#include "ginza1_common.h"
#include "ginza1_factor_list.h"
#include "ginza1_solver.h"

int main(int argc, char *argv[])
{
    enum solver_method_t
    {
        METHOD_FACTOR,
        METHOD_POWM,
    };

    const char *challenge_file = "data/challenges/challenge_16_1024.txt";
    const char *modulo_file = "data/challenges/modulo_16_1024.txt";
    solver_method_t method = METHOD_FACTOR;
    solver_options_t options = {
        .smoothness_bound = 100000UL,
        .attempts_per_worker = 2000UL,
    };

    int positional_index = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--bound") == 0)
        {
            if (argi + 1 >= argc || !parse_unsigned_long_arg(argv[argi + 1], &options.smoothness_bound))
            {
                fprintf(stderr, "Error: Invalid value for --bound\n");
                return 1;
            }
            argi++;
            continue;
        }

        if (strcmp(argv[argi], "--attempts") == 0)
        {
            if (argi + 1 >= argc || !parse_unsigned_long_arg(argv[argi + 1], &options.attempts_per_worker))
            {
                fprintf(stderr, "Error: Invalid value for --attempts\n");
                return 1;
            }
            argi++;
            continue;
        }

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

        if (positional_index == 0)
        {
            challenge_file = argv[argi];
        }
        else if (positional_index == 1)
        {
            modulo_file = argv[argi];
        }
        else
        {
            fprintf(stderr, "Error: Unexpected argument %s\n", argv[argi]);
            return 1;
        }
        positional_index++;
    }

    int k = extract_k_from_filename(challenge_file);

    if (k > 0)
    {
        printf("K=%d Time-Lock Puzzle Solver\n", k);
    }
    else
    {
        printf("Time-Lock Puzzle Solver\n");
    }
    printf("=============================\n");
    printf("Challenge file: %s\n", challenge_file);
    printf("Modulo file: %s\n", modulo_file);
    printf("Method: %s\n", (method == METHOD_POWM) ? "powm" : "factor");

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);

    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);

    printf("\nInput numbers:\n");
    gmp_printf("challenge = %Zd\n", challenge);
    gmp_printf("modulo = %Zd\n", modulo);

    if (mpz_cmp(modulo, challenge) <= 0)
    {
        printf("WARNING: Modulo is not larger than challenge!\n");
    }

    double start = wall_time_seconds();

    if (method == METHOD_POWM)
    {
        mpz_t candidate_cuberoot;
        mpz_init(candidate_cuberoot);

        if (!compute_cube_root_direct_powm(candidate_cuberoot, challenge, modulo))
        {
            fprintf(stderr, "Error: mpz_powm method cannot be applied because inverse of 3 modulo (modulo-1) does not exist\n");
            mpz_clear(candidate_cuberoot);
            mpz_clears(challenge, modulo, NULL);
            return 1;
        }

        verify_cube_root(candidate_cuberoot, challenge, modulo);
        mpz_clear(candidate_cuberoot);

        double end = wall_time_seconds();
        double elapsed_seconds = end - start;
        double elapsed_milliseconds = elapsed_seconds * 10000.0;
        printf("\nCompleted in %.2f seconds (%.3f ms)\n", elapsed_seconds, elapsed_milliseconds);

        mpz_clears(challenge, modulo, NULL);
        return 0;
    }

    factor_list_t factors;
    factor_list_init(&factors);

    switch (k)
    {
    case 16:
        solve_k16(challenge, &factors);
        break;
    case 32:
        solve_k32(challenge, &factors);
        break;
    case 64:
        solve_k64(challenge, &factors, &options);
        break;
    default:
        solve_unknown_k(k, challenge, &factors);
        break;
    }

    if (factors.count > 0)
    {
        mpz_t candidate_cuberoot;
        mpz_init(candidate_cuberoot);

        compute_cube_root_from_factors(candidate_cuberoot, &factors, modulo);
        verify_cube_root(candidate_cuberoot, challenge, modulo);

        mpz_clear(candidate_cuberoot);
    }

    double end = wall_time_seconds();
    double elapsed_seconds = end - start;
    double elapsed_milliseconds = elapsed_seconds * 10000.0;
    printf("\nFactorization completed in %.2f seconds (%.3f ms)\n", elapsed_seconds, elapsed_milliseconds);

    factor_list_clear(&factors);
    mpz_clears(challenge, modulo, NULL);

    return 0;
}
