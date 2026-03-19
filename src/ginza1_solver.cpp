#include "ginza1_solver.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ginza1_common.h"
#include "ginza1_factorization.h"

void solve_k16(const mpz_t challenge, factor_list_t *factors)
{
    printf("\n[K=16] Strategy: Trial division\n");
    printf("Generating all primes up to 65536...\n");

    int prime_count = 0;
    unsigned long *primes = generate_primes_up_to(65536UL, &prime_count);
    printf("Found %d primes\n", prime_count);

#ifdef _OPENMP
    printf("OpenMP enabled with up to %d threads\n", omp_get_max_threads());
#else
    printf("OpenMP not enabled; compile with -fopenmp to parallelize K=16\n");
#endif

    factor_challenge_trial_division(challenge, primes, prime_count, factors);
    free(primes);
}

void solve_k32(const mpz_t challenge, factor_list_t *factors)
{
    printf("\n[K=32] Strategy: Pollard Rho\n");
    factor_challenge_pollard_with_progress(challenge, factors);
    print_factorization_summary(challenge, factors);
}

void solve_k64(const mpz_t challenge, factor_list_t *factors, const solver_options_t *options)
{
    printf("\n[K=64] Strategy: Pollard Rho\n");
    printf("Configured smoothness bound: %lu\n", options->smoothness_bound);
    printf("Configured attempts per worker: %lu\n", options->attempts_per_worker);
    printf("Note: Pollard Rho is used for practical runtime on K=64 inputs.\n");

    factor_challenge_pollard_with_progress(challenge, factors);
    print_factorization_summary(challenge, factors);
}

void solve_unknown_k(int k, const mpz_t challenge, factor_list_t *factors)
{
    if (k > 0)
    {
        printf("\n[K=%d] Unsupported explicit branch; falling back to Pollard Rho\n", k);
    }
    else
    {
        printf("\n[K=unknown] Falling back to Pollard Rho\n");
    }

    factor_challenge_pollard_with_progress(challenge, factors);
    print_factorization_summary(challenge, factors);
}
