#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct
{
    mpz_t *items;
    int count;
    int capacity;
} factor_list_t;

bool is_b_smooth(mpz_t n, unsigned long *primes, int prime_count, int *exponents);
bool find_b_smooth_transform(mpz_t transformed_value, mpz_t blinding_factor, factor_list_t *factors, const mpz_t challenge, const mpz_t modulo, unsigned long *primes, int prime_count, unsigned long attempts_per_worker);
bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo);
void print_factorization_summary(const mpz_t value, factor_list_t *factors);
void compute_cube_root_from_factors(mpz_t result, factor_list_t *factors, mpz_t modulo);
void verify_cube_root(mpz_t candidate_cuberoot, mpz_t challenge, mpz_t modulo);
void factor_challenge_pollard_with_progress(const mpz_t challenge, factor_list_t *factors);
double wall_time_seconds();

// Function to read an mpz_t from a file
void read_mpz_from_file(mpz_t num, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // Try decimal text first, then fallback to GMP raw binary format.
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

// Generate all primes up to a given limit using Sieve of Eratosthenes
unsigned long *generate_primes_up_to(unsigned long limit, int *count)
{
    char *sieve = (char *)malloc(limit + 1);
    if (!sieve)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize sieve - assume all numbers are prime initially
    memset(sieve, 1, limit + 1);
    sieve[0] = sieve[1] = 0;

    // Sieve
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

    // Count primes
    *count = 0;
    for (unsigned long i = 2; i <= limit; i++)
    {
        if (sieve[i])
            (*count)++;
    }

    // Allocate array for primes
    unsigned long *primes = (unsigned long *)malloc(*count * sizeof(unsigned long));
    if (!primes)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(sieve);
        exit(1);
    }

    // Fill primes array
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

void factor_list_init(factor_list_t *list)
{
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

void factor_list_append(factor_list_t *list, const mpz_t value)
{
    if (list->count == list->capacity)
    {
        int new_capacity = (list->capacity == 0) ? 16 : list->capacity * 2;
        mpz_t *new_items = (mpz_t *)realloc(list->items, (size_t)new_capacity * sizeof(mpz_t));
        if (!new_items)
        {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }

    mpz_init_set(list->items[list->count], value);
    list->count++;
}

void factor_list_clear(factor_list_t *list)
{
    for (int i = 0; i < list->count; i++)
    {
        mpz_clear(list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
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

    p += 10; // skip "challenge_" or "modulo_"
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

void pollard_rho(mpz_t factor, const mpz_t n)
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
        
        // Print on initialization so we can see how many threads actually spawned
        if (thread_id == 0 && num_threads > 1) {
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
                    if (mpz_cmp_ui(d, 1) != 0) {
                        break;
                    }
                    mpz_set_ui(prod, 1);
                    
                    int stop = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
                    stop = found;
                    if (stop) break;
                    
                    // Note: move periodic logging to be thread-local rather than across threads
                    // so that the atomic operations don't slow down the main batching.
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
            
            if (mpz_cmp_ui(d, 1) == 0) {
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

void factor_recursive(const mpz_t n, factor_list_t *factors)
{
    if (mpz_cmp_ui(n, 1) == 0)
    {
        return;
    }

    if (mpz_probab_prime_p(n, 25) > 0) // to change to 50 for more certainty
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

void factor_challenge_pollard(const mpz_t challenge)
{
    printf("\n========================================\n");
    printf("Factoring challenge number:\n");
    gmp_printf("Challenge = %Zd\n", challenge);
    printf("========================================\n");
    printf("Prime factors found:\n");

    factor_list_t factors;
    factor_list_init(&factors);

    factor_recursive(challenge, &factors);

    for (int i = 0; i < factors.count; i++)
    {
        printf("  Factor %d: ", i + 1);
        gmp_printf("%Zd\n", factors.items[i]);
    }

    mpz_t product;
    mpz_init_set_ui(product, 1);
    for (int i = 0; i < factors.count; i++)
    {
        mpz_mul(product, product, factors.items[i]);
    }

    printf("========================================\n");
    if (mpz_cmp(product, challenge) == 0)
    {
        printf("SUCCESS: Fully factored challenge into %d primes!\n", factors.count);
    }
    else
    {
        printf("WARNING: Product of factors does not match challenge\n");
    }

    mpz_clear(product);
    factor_list_clear(&factors);
}

void factor_list_append_ui(factor_list_t *list, unsigned long value)
{
    mpz_t factor;
    mpz_init_set_ui(factor, value);
    factor_list_append(list, factor);
    mpz_clear(factor);
}

void factor_list_append_ui_power(factor_list_t *list, unsigned long value, int exponent)
{
    for (int i = 0; i < exponent; i++)
    {
        factor_list_append_ui(list, value);
    }
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

bool find_b_smooth_transform(mpz_t transformed_value, mpz_t blinding_factor, factor_list_t *factors, const mpz_t challenge, const mpz_t modulo, unsigned long *primes, int prime_count, unsigned long attempts_per_worker)
{
    const size_t exponent_bytes = (size_t)prime_count * sizeof(int);
    int *final_exponents = (int *)calloc((size_t)prime_count, sizeof(int));
    if (!final_exponents)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int found = 0;
    mpz_set_ui(transformed_value, 0);
    mpz_set_ui(blinding_factor, 0);

#ifdef _OPENMP
#pragma omp parallel shared(found, final_exponents, transformed_value, blinding_factor)
#endif
    {
        gmp_randstate_t state;
        gmp_randinit_default(state);

        unsigned long seed = (unsigned long)time(NULL);
#ifdef _OPENMP
        seed ^= (unsigned long)(omp_get_thread_num() + 1U) * 0x9E3779B1UL;
#endif
        gmp_randseed_ui(state, seed);

        mpz_t r, r_cubed, y;
        mpz_inits(r, r_cubed, y, NULL);

        int *local_exponents = (int *)calloc((size_t)prime_count, sizeof(int));
        if (!local_exponents)
        {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }

        for (unsigned long attempt = 0; attempt < attempts_per_worker; attempt++)
        {
            int stop = 0;
#ifdef _OPENMP
#pragma omp atomic read
#endif
            stop = found;

            if (stop)
            {
                break;
            }

            memset(local_exponents, 0, exponent_bytes);

            mpz_urandomm(r, state, modulo);
            if (mpz_cmp_ui(r, 0) == 0)
            {
                continue;
            }

            mpz_powm_ui(r_cubed, r, 3, modulo);
            mpz_mul(y, challenge, r_cubed);
            mpz_mod(y, y, modulo);

            if (mpz_cmp_ui(y, 0) == 0)
            {
                continue;
            }

            if (is_b_smooth(y, primes, prime_count, local_exponents))
            {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    if (!found)
                    {
                        found = 1;
                        mpz_set(transformed_value, y);
                        mpz_set(blinding_factor, r);
                        memcpy(final_exponents, local_exponents, exponent_bytes);
                    }
                }
                break;
            }
        }

        free(local_exponents);
        mpz_clears(r, r_cubed, y, NULL);
        gmp_randclear(state);
    }

    if (!found)
    {
        free(final_exponents);
        return false;
    }

    for (int i = 0; i < prime_count; i++)
    {
        if (final_exponents[i] > 0)
        {
            factor_list_append_ui_power(factors, primes[i], final_exponents[i]);
        }
    }

    free(final_exponents);
    return true;
}

void print_factorization_summary(const mpz_t value, factor_list_t *factors)
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

int count_prime_exponent(const mpz_t challenge, unsigned long prime)
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

// Trial division to find factors
void factor_challenge(mpz_t challenge, unsigned long *primes, int prime_count, factor_list_t *factors)
{
    mpz_t temp;
    mpz_init_set(temp, challenge);

    int *exponents = (int *)calloc((size_t)prime_count, sizeof(int));
    int *smooth_exponents = (int *)calloc((size_t)prime_count, sizeof(int));
    if (!exponents)
    {
        fprintf(stderr, "Memory allocation failed\n");
        mpz_clear(temp);
        exit(1);
    }

    if (!smooth_exponents)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(exponents);
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

    // Check result
    if (mpz_cmp_ui(temp, 1) == 0)
    {
        printf("SUCCESS: Fully factored challenge into %d primes!\n", factor_count);
    }
    else
    {
        printf("WARNING: Not fully factored. Remaining: ");
        gmp_printf("%Zd\n", temp);
        printf("This shouldn't happen for K=16 - check your prime list\n");
    }

    if (!smooth)
    {
        printf("WARNING: Challenge is not B-smooth over the generated factor base\n");
    }

    // Verify the product equals original
    printf("\nVerification:\n");
    printf("This should match the original challenge\n");

    free(smooth_exponents);
    free(exponents);
    mpz_clear(temp);
}

void factor_challenge_pollard_with_progress(const mpz_t challenge, factor_list_t *factors)
{
    printf("Factoring using Pollard's Rho...\n");

    factor_recursive(challenge, factors);

    printf("done\n");
    fflush(stdout);
}

bool is_b_smooth(mpz_t n, unsigned long *primes, int prime_count, int *exponents)
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
        
        // Early exit: if temp reaches 1, we are done
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

void compute_cube_root_from_factors(mpz_t result, factor_list_t *factors, mpz_t modulo)
{
    mpz_t challenge_mod;
    mpz_init(challenge_mod);

    mpz_set_ui(challenge_mod, 1);
    for (int i = 0; i < factors->count; i++)
    {
        mpz_mul(challenge_mod, challenge_mod, factors->items[i]);
        mpz_mod(challenge_mod, challenge_mod, modulo);
    }

    if (!compute_cube_root_mod_prime(result, challenge_mod, modulo))
    {
        fprintf(stderr, "Error: Cannot compute modular inverse of 3 modulo (modulo - 1)\n");
        mpz_set_ui(result, 0);
        mpz_clear(challenge_mod);
        return;
    }

    mpz_clear(challenge_mod);
}

void verify_cube_root(mpz_t candidate_cuberoot, mpz_t challenge, mpz_t modulo)
{
    mpz_t cubed;
    mpz_init(cubed);

    // Compute (candidate)^3 mod modulo
    mpz_powm_ui(cubed, candidate_cuberoot, 3, modulo);

    printf("\nVerification:\n");
    gmp_printf("Candidate^3 mod modulo = %Zd\n", cubed);
    gmp_printf("Original challenge       = %Zd\n", challenge);

    if (mpz_cmp(cubed, challenge) == 0)
    {
        printf("✓ SUCCESS: Cube root is correct!\n");
    }
    else
    {
        printf("✗ FAILURE: Cube root is incorrect\n");
    }

    mpz_clear(cubed);
}

// Main function
int main(int argc, char *argv[])
{
    // Default filenames - adjust these to match your actual files
    const char *challenge_file = "data/challenges/challenge_16_1024.txt";
    const char *modulo_file = "data/challenges/modulo_16_1024.txt";
    unsigned long smoothness_bound = 100000;
    unsigned long attempts_per_worker = 2000;

    int positional_index = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--bound") == 0)
        {
            if (argi + 1 >= argc || !parse_unsigned_long_arg(argv[argi + 1], &smoothness_bound))
            {
                fprintf(stderr, "Error: Invalid value for --bound\n");
                return 1;
            }
            argi++;
            continue;
        }

        if (strcmp(argv[argi], "--attempts") == 0)
        {
            if (argi + 1 >= argc || !parse_unsigned_long_arg(argv[argi + 1], &attempts_per_worker))
            {
                fprintf(stderr, "Error: Invalid value for --attempts\n");
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
    if (k == 64)
    {
        printf("Smoothness bound: %lu\n", smoothness_bound);
        printf("Attempts per worker: %lu\n", attempts_per_worker);
    }

    // Read the challenge and modulo
    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);

    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);

    // Display the numbers
    printf("\nInput numbers:\n");
    gmp_printf("challenge = %Zd\n", challenge);
    gmp_printf("modulo = %Zd\n", modulo);

    // Check that modulo > challenge (should be true)
    if (mpz_cmp(modulo, challenge) <= 0)
    {
        printf("WARNING: Modulo is not larger than challenge!\n");
    }

    // Time the factorization (wall-clock elapsed time)
    double start = wall_time_seconds();
    factor_list_t factors;
    factor_list_init(&factors);

    bool candidate_ready = false;
    mpz_t candidate_cuberoot;
    mpz_init(candidate_cuberoot);

    if (k == 16)
    {
        printf("\nUsing trial division for K=16\n");
        printf("Generating all primes up to 65536...\n");
        int prime_count;
        unsigned long *primes = generate_primes_up_to(65536, &prime_count);
        printf("Found %d primes\n", prime_count);
#ifdef _OPENMP
        printf("OpenMP enabled with up to %d threads\n", omp_get_max_threads());
#else
        printf("OpenMP not enabled; compile with -fopenmp to parallelize K=16\n");
#endif

        factor_challenge(challenge, primes, prime_count, &factors);
        free(primes);
    }
    else if (k == -1)
    {
        printf("\nUsing Pollard Rho factorization (K=64)\n");
        printf("Note: A B-smooth search on C * r^3 mod N for a 1024-bit number has vanishingly small probability (~10^-48)\n");
        printf("However, since the challenge C is natively a product of <= 64-bit primes, parallel Pollard Rho will break it natively within a few hours!\n");
        factor_challenge_pollard_with_progress(challenge, &factors);
        print_factorization_summary(challenge, &factors);
        
        if (factors.count > 0)
        {
            compute_cube_root_from_factors(candidate_cuberoot, &factors, modulo);
            candidate_ready = true;
        }
    }
    else
    {
        printf("\nUsing Pollard Rho factorization (K=%d)\n", k);
        factor_challenge_pollard_with_progress(challenge, &factors);
        print_factorization_summary(challenge, &factors);
    }

    if (candidate_ready)
    {
        verify_cube_root(candidate_cuberoot, challenge, modulo);
    }
    else if (factors.count > 0)
    {
        compute_cube_root_from_factors(candidate_cuberoot, &factors, modulo);
        verify_cube_root(candidate_cuberoot, challenge, modulo);
    }

    double end = wall_time_seconds();
    double time_spent = end - start;
    printf("\nFactorization completed in %.2f seconds\n", time_spent);

    // Clean up
    factor_list_clear(&factors);
    mpz_clear(candidate_cuberoot);
    mpz_clears(challenge, modulo, NULL);

    return 0;
}
