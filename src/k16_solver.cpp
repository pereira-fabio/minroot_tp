#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

// Structure to store factors
typedef struct {
    mpz_t *items;
    int count;
    int capacity;
} factor_list_t;

// Initialize factor list
void factor_list_init(factor_list_t *list) {
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

// Add a factor to the list
void factor_list_append(factor_list_t *list, mpz_t value) {
    if (list->count == list->capacity) {
        int new_capacity = (list->capacity == 0) ? 16 : list->capacity * 2;
        mpz_t *new_items = (mpz_t *)realloc(list->items, new_capacity * sizeof(mpz_t));
        if (!new_items) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }
    mpz_init_set(list->items[list->count], value);
    list->count++;
}

// Clear factor list (free memory)
void factor_list_clear(factor_list_t *list) {
    for (int i = 0; i < list->count; i++) {
        mpz_clear(list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

// Read a number from file (supports both decimal text and GMP raw format)
void read_mpz_from_file(mpz_t num, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }
    
    // Try decimal text first
    if (mpz_inp_str(num, f, 10) == 0) {
        // If that fails, try raw binary format
        rewind(f);
        if (mpz_inp_raw(num, f) == 0) {
            fprintf(stderr, "Error: Failed to read number from %s\n", filename);
            fclose(f);
            exit(1);
        }
    }
    
    fclose(f);
    printf("Read number from %s\n", filename);
}

// Generate all primes up to limit using Sieve of Eratosthenes
unsigned long* generate_primes_up_to(unsigned long limit, int *count) {
    char *sieve = (char *)malloc(limit + 1);
    if (!sieve) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    // Initialize sieve
    memset(sieve, 1, limit + 1);
    sieve[0] = sieve[1] = 0;
    
    // Sieve
    for (unsigned long i = 2; i * i <= limit; i++) {
        if (sieve[i]) {
            for (unsigned long j = i * i; j <= limit; j += i) {
                sieve[j] = 0;
            }
        }
    }
    
    // Count primes
    *count = 0;
    for (unsigned long i = 2; i <= limit; i++) {
        if (sieve[i]) (*count)++;
    }
    
    // Store primes
    unsigned long *primes = (unsigned long *)malloc(*count * sizeof(unsigned long));
    if (!primes) {
        fprintf(stderr, "Memory allocation failed\n");
        free(sieve);
        exit(1);
    }
    
    int idx = 0;
    for (unsigned long i = 2; i <= limit; i++) {
        if (sieve[i]) {
            primes[idx++] = i;
        }
    }
    
    free(sieve);
    return primes;
}

// Save the generated primes to a text file
void save_primes_to_file(unsigned long *primes, int count, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return;
    }
    
    for (int i = 0; i < count; i++) {
        fprintf(f, "%lu\n", primes[i]);
    }
    
    fclose(f);
    printf("Saved %d primes to %s\n", count, filename);
}

// Count how many times a prime divides a number
int count_exponent(mpz_t n, unsigned long prime) {
    mpz_t temp;
    mpz_init_set(temp, n);
    
    int exponent = 0;
    while (mpz_divisible_ui_p(temp, prime)) {
        mpz_divexact_ui(temp, temp, prime);
        exponent++;
    }
    
    mpz_clear(temp);
    return exponent;
}

// Factor the challenge using trial division (K=16 method)
void factor_challenge(mpz_t challenge, unsigned long *primes, int prime_count, factor_list_t *factors) {
    mpz_t remaining;
    mpz_init_set(remaining, challenge);
    
    printf("\n========================================\n");
    printf("Factoring challenge...\n");
    printf("========================================\n");
    
    int factor_count = 0;
    
    // Try each prime in our factor base
    for (int i = 0; i < prime_count; i++) {
        unsigned long prime = primes[i];
        
        // Count how many times this prime divides the challenge
        int exp = count_exponent(challenge, prime);
        
        // Add each occurrence as a separate factor
        for (int e = 0; e < exp; e++) {
            mpz_t factor;
            mpz_init_set_ui(factor, prime);
            factor_list_append(factors, factor);
            mpz_clear(factor);
            
            // Remove this factor from remaining
            mpz_divexact_ui(remaining, remaining, prime);
            factor_count++;
            
            printf("  Factor %d: %lu (0x%lX)\n", factor_count, prime, prime);
            
            if (mpz_cmp_ui(remaining, 1) != 0) {
                printf("    Remaining: ");
                gmp_printf("%Zd\n", remaining);
            }
        }
        
        // Early exit if fully factored
        if (mpz_cmp_ui(remaining, 1) == 0) {
            break;
        }
    }
    
    printf("========================================\n");
    
    // Check if we fully factored
    if (mpz_cmp_ui(remaining, 1) == 0) {
        printf("SUCCESS: Fully factored into %d primes!\n", factor_count);
    } else {
        printf("WARNING: Not fully factored. Remaining: ");
        gmp_printf("%Zd\n", remaining);
        printf("This shouldn't happen for K=16\n");
    }
    
    mpz_clear(remaining);
}

// Compute cube root from factors
void compute_cube_root_from_factors(mpz_t result, factor_list_t *factors, mpz_t modulo) {
    mpz_t phi, exponent, three, challenge_mod;
    mpz_inits(phi, exponent, three, challenge_mod, NULL);
    
    // Step 1: Compute challenge mod modulo from factors
    mpz_set_ui(challenge_mod, 1);
    for (int i = 0; i < factors->count; i++) {
        mpz_mul(challenge_mod, challenge_mod, factors->items[i]);
        mpz_mod(challenge_mod, challenge_mod, modulo);
    }
    
    // Step 2: Compute exponent = 3^(-1) mod (modulo - 1)
    mpz_sub_ui(phi, modulo, 1);  // phi = modulo - 1
    mpz_set_ui(three, 3);
    
    if (mpz_invert(exponent, three, phi) == 0) {
        fprintf(stderr, "Error: 3 has no inverse modulo (modulo-1)\n");
        mpz_set_ui(result, 0);
        mpz_clears(phi, exponent, three, challenge_mod, NULL);
        return;
    }
    
    printf("\nExponent = ");
    gmp_printf("%Zd\n", exponent);
    
    // Step 3: Compute cube root = challenge_mod^exponent mod modulo
    mpz_powm(result, challenge_mod, exponent, modulo);
    
    mpz_clears(phi, exponent, three, challenge_mod, NULL);
}

// Verify the cube root
void verify_cube_root(mpz_t cube_root, mpz_t original_challenge, mpz_t modulo) {
    mpz_t cubed, challenge_mod;
    mpz_inits(cubed, challenge_mod, NULL);
    
    // Compute cube_root^3 mod modulo
    mpz_powm_ui(cubed, cube_root, 3, modulo);
    
    // Compute original_challenge mod modulo
    mpz_mod(challenge_mod, original_challenge, modulo);
    
    printf("\n========================================\n");
    printf("Verification:\n");
    printf("========================================\n");
    gmp_printf("Cube root found: %Zd\n", cube_root);
    gmp_printf("\nCheck: (%Zd)^3 mod %Zd = %Zd\n", cube_root, modulo, cubed);
    gmp_printf("Original challenge mod %Zd = %Zd\n", modulo, challenge_mod);
    
    if (mpz_cmp(cubed, challenge_mod) == 0) {
        printf("\n✓ SUCCESS! Cube root is correct!\n");
    } else {
        printf("\n✗ FAILURE! Cube root is incorrect!\n");
    }
    
    mpz_clears(cubed, challenge_mod, NULL);
}


int main(int argc, char *argv[]) {
    // Default filenames (change these to match your files)
    const char *challenge_file = "challenge_16_1024.txt";
    const char *modulo_file = "modulo_16_1024.txt";
    
    // Allow command-line override
    if (argc >= 2) challenge_file = argv[1];
    if (argc >= 3) modulo_file = argv[2];
        
    printf("========================================\n");
    printf("K=16 Cube Root Solver (Time-Lock Puzzle)\n");
    printf("========================================\n");
    printf("Challenge file: %s\n", challenge_file);
    printf("Modulo file: %s\n", modulo_file);
    printf("\n");
    
    // Read challenge and modulo
    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);
    
    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);
    
    // Display numbers
    printf("\nChallenge: ");
    gmp_printf("%Zd...\n", challenge);  // Truncated for display
    printf("Modulo: ");
    gmp_printf("%Zd\n", modulo);
    
    // Check that modulo is prime (should be)
    if (mpz_probab_prime_p(modulo, 25) == 0) {
        printf("WARNING: Modulo is not prime!\n");
    }
    
    // Check that modulo > challenge
    if (mpz_cmp(modulo, challenge) <= 0) {
        printf("WARNING: Modulo is not larger than challenge!\n");
    }
    
    // Time the whole process
    clock_t start = clock();
       
    printf("\n=== Phase 1: Factoring Challenge ===\n");
    printf("Generating all primes up to 65536...\n");
    
    int prime_count;
    unsigned long *primes = generate_primes_up_to(USHRT_MAX, &prime_count);
    printf("Generated %d primes\n", prime_count);
    
    // Save primes to file
    //save_primes_to_file(primes, prime_count, "primes_k16.txt");
    
    factor_list_t factors;
    factor_list_init(&factors);
    
    factor_challenge(challenge, primes, prime_count, &factors);
    
    printf("\n=== Phase 2: Computing Cube Root ===\n");
    
    if (factors.count == 0) {
        printf("ERROR: No factors found!\n");
        factor_list_clear(&factors);
        free(primes);
        mpz_clears(challenge, modulo, NULL);
        return 1;
    }
    
    mpz_t cube_root;
    mpz_init(cube_root);
    
    compute_cube_root_from_factors(cube_root, &factors, modulo);
    
    printf("\n=== Phase 3: Verification ===\n");
    verify_cube_root(cube_root, challenge, modulo);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nTotal time: %.9f seconds\n", time_spent);
    
    // Cleanup
    mpz_clear(cube_root);
    factor_list_clear(&factors);
    free(primes);
    mpz_clears(challenge, modulo, NULL);
    
    return 0;
}