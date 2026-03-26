#pragma once

#include <gmp.h>

// --- prime generation ---
unsigned long *generate_primes_up_to(unsigned long limit, int *count);

// --- file I/O ---
void read_mpz_from_file(mpz_t num, const char *filename);

// --- cube root computation ---
bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo);
void compute_cube_root_from_factors(mpz_t result, const int *exponents,
                                    const mpz_t *cbrt_table, int prime_count,
                                    const mpz_t modulo);

// --- precomputed table ---
void precompute_cbrt_table(mpz_t *cbrt_table, const unsigned long *primes,
                           int prime_count, const mpz_t modulo);
void save_cbrt_table(const mpz_t *cbrt_table, int prime_count, const char *filename);
bool load_cbrt_table(mpz_t *cbrt_table, int prime_count, const char *filename);

// --- verification ---
void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod,
                      const mpz_t modulo);

// --- factoring helpers ---
int count_exponent(const mpz_t n, unsigned long prime);