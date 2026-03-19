#include "ginza1_factorization.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int count_prime_exponent(const mpz_t challenge, unsigned long prime)
{
    mpz_t temp;
    mpz_init_set(temp, challenge);

    int exponent = 0;
    while (mpz_divisible_ui_p(temp, prime))
    {
        mpz_divexact_ui(temp, temp, prime);
        exponent++;
    }

    mpz_clear(temp);
    return exponent;
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

bool is_b_smooth(const mpz_t n, const unsigned long *primes, int prime_count, int *exponents)
{
    mpz_t temp;
    mpz_init_set(temp, n);

    for (int i = 0; i < prime_count; i++)
    {
        while (mpz_divisible_ui_p(temp, primes[i]))
        {
            mpz_divexact_ui(temp, temp, primes[i]);
            exponents[i]++;
        }

        if (mpz_cmp_ui(temp, 1) == 0)
        {
            mpz_clear(temp);
            return true;
        }
    }

    bool smooth = (mpz_cmp_ui(temp, 1) == 0);
    mpz_clear(temp);
    return smooth;
}

void factor_challenge_trial_division(const mpz_t challenge, const unsigned long *primes, int prime_count, factor_list_t *factors)
{
    mpz_t temp;
    mpz_init_set(temp, challenge);

    int *exponents = (int *)calloc((size_t)prime_count, sizeof(int));
    int *smooth_exponents = (int *)calloc((size_t)prime_count, sizeof(int));
    if (!exponents || !smooth_exponents)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(exponents);
        free(smooth_exponents);
        mpz_clear(temp);
        exit(1);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < prime_count; i++)
    {
        exponents[i] = count_prime_exponent(challenge, primes[i]);
    }

    bool smooth = is_b_smooth(challenge, primes, prime_count, smooth_exponents);

    printf("\n========================================\n");
    printf("Factoring challenge number:\n");
    gmp_printf("Challenge = %Zd\n", challenge);
    printf("========================================\n");
    printf("Prime factors found:\n");

    int factor_count = 0;
    for (int i = 0; i < prime_count; i++)
    {
        for (int exponent = 0; exponent < exponents[i]; exponent++)
        {
            mpz_divexact_ui(temp, temp, primes[i]);
            factor_list_append_ui(factors, primes[i]);

            printf("  Factor %d: %lu", ++factor_count, primes[i]);
            gmp_printf(" (0x%lX)\n", primes[i]);

            if (mpz_cmp_ui(temp, 1) != 0)
            {
                printf("  Remaining: ");
                gmp_printf("%Zd...\n", temp);
            }
        }
    }

    printf("========================================\n");
    if (mpz_cmp_ui(temp, 1) == 0)
    {
        printf("SUCCESS: Fully factored challenge into %d primes!\n", factor_count);
    }
    else
    {
        printf("WARNING: Not fully factored. Remaining: ");
        gmp_printf("%Zd\n", temp);
    }

    if (!smooth)
    {
        printf("WARNING: Challenge is not B-smooth over the generated factor base\n");
    }

    free(smooth_exponents);
    free(exponents);
    mpz_clear(temp);
}

static void pollard_rho(mpz_t factor, const mpz_t n)
{
    static const unsigned long small_primes[] = {2UL, 3UL, 5UL, 7UL, 11UL, 13UL, 17UL, 19UL, 23UL, 29UL};
    for (int i = 0; i < 10; i++)
    {
        if (mpz_divisible_ui_p(n, small_primes[i]))
        {
            mpz_set_ui(factor, small_primes[i]);
            return;
        }
    }

    if (mpz_even_p(n))
    {
        mpz_set_ui(factor, 2);
        return;
    }

    int found = 0;

#ifdef _OPENMP
#pragma omp parallel shared(found, factor)
#endif
    {
        gmp_randstate_t state;
        gmp_randinit_default(state);

        unsigned long seed = (unsigned long)time(NULL);
#ifdef _OPENMP
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        seed ^= (unsigned long)(thread_id + 1U) * 0x9E3779B1UL;

        if (thread_id == 0 && num_threads > 1)
        {
            printf("  [OpenMP] Parallel Pollard's Rho spawned %d threads.\n", num_threads);
            fflush(stdout);
        }
#endif
        gmp_randseed_ui(state, seed);

        mpz_t x, y, c, d, tmp, abs_diff, prod;
        mpz_inits(x, y, c, d, tmp, abs_diff, prod, NULL);

        while (!found)
        {
            mpz_urandomm(x, state, n);
            if (mpz_cmp_ui(x, 2) < 0)
            {
                mpz_add_ui(x, x, 2);
            }
            mpz_set(y, x);

            mpz_urandomm(c, state, n);
            if (mpz_cmp_ui(c, 0) == 0)
            {
                mpz_set_ui(c, 1);
            }

            mpz_set_ui(d, 1);
            mpz_set_ui(prod, 1);

            int inner_iterations = 0;
            unsigned long long total_steps = 0;
            const int batch_size = 128;

            while (mpz_cmp_ui(d, 1) == 0)
            {
                if (++inner_iterations % batch_size == 0)
                {
                    mpz_gcd(d, prod, n);
                    if (mpz_cmp_ui(d, 1) != 0)
                    {
                        break;
                    }
                    mpz_set_ui(prod, 1);

                    int stop = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
                    stop = found;
                    if (stop)
                    {
                        break;
                    }

                    total_steps += batch_size;
                    inner_iterations = 0;

#ifdef _OPENMP
                    if (omp_get_thread_num() == 0)
#endif
                    {
                        if (total_steps % (50000000ULL) == 0)
                        {
                            printf("  [Progress] Thread 0 reached %llu million modular steps on current chunk...\n", total_steps / 1000000ULL);
                            fflush(stdout);
                        }
                    }
                }

                mpz_mul(tmp, x, x);
                mpz_add(tmp, tmp, c);
                mpz_mod(x, tmp, n);

                mpz_mul(tmp, y, y);
                mpz_add(tmp, tmp, c);
                mpz_mod(y, tmp, n);

                mpz_mul(tmp, y, y);
                mpz_add(tmp, tmp, c);
                mpz_mod(y, tmp, n);

                mpz_sub(abs_diff, x, y);
                mpz_abs(abs_diff, abs_diff);

                mpz_mul(prod, prod, abs_diff);
                mpz_mod(prod, prod, n);
            }

            if (mpz_cmp_ui(d, 1) == 0)
            {
                mpz_gcd(d, prod, n);
            }

            if (mpz_cmp_ui(d, 1) != 0 && mpz_cmp(d, n) != 0)
            {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    if (!found)
                    {
                        mpz_set(factor, d);
                        found = 1;
                    }
                }
                break;
            }
        }

        mpz_clears(x, y, c, d, tmp, abs_diff, prod, NULL);
        gmp_randclear(state);
    }
}

static void factor_recursive(const mpz_t n, factor_list_t *factors)
{
    if (mpz_cmp_ui(n, 1) == 0)
    {
        return;
    }

    if (mpz_probab_prime_p(n, 25) > 0)
    {
        gmp_printf("[*] Found prime factor: %Zd\n", n);
        fflush(stdout);
        factor_list_append(factors, n);
        return;
    }

    mpz_t divisor, quotient;
    mpz_inits(divisor, quotient, NULL);

    pollard_rho(divisor, n);
    mpz_divexact(quotient, n, divisor);

    gmp_printf("\n[+] Split composite into a smaller chunk! Resuming recursive search...\n");
    fflush(stdout);

    factor_recursive(divisor, factors);
    factor_recursive(quotient, factors);

    mpz_clears(divisor, quotient, NULL);
}

void factor_challenge_pollard_with_progress(const mpz_t challenge, factor_list_t *factors)
{
    printf("Factoring using Pollard's Rho...\n");
    factor_recursive(challenge, factors);
    printf("done\n");
    fflush(stdout);
}
