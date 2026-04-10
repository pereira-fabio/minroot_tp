#pragma once

// =============================================================================
// k_common.h — Shared utilities for modular cube root solvers
//
// Mathematical context:
//   We compute challenge^(1/3) mod M, where M is prime and gcd(3, M-1) = 1.
//
//   By Fermat's little theorem:
//     x^(1/3) mod M  =  x^e mod M,   e = 3^-1 mod (M-1)
//
//   Homomorphism exploited for factorization-based solvers:
//     cbrt(a * b) = cbrt(a) * cbrt(b) mod M
//
//   So if challenge = p1^e1 * p2^e2 * ... * pN^eN, then:
//     cbrt(challenge) = cbrt(p1)^e1 * cbrt(p2)^e2 * ... * cbrt(pN)^eN mod M
//
//   This lets us replace one expensive powm on a large number with many
//   cheap powm calls on small primes — precomputed once per modulo file.
// =============================================================================

#include <gmp.h>

// ---------------------------------------------------------------------------
// Prime generation
// ---------------------------------------------------------------------------

// Generate all primes up to `limit` using the Sieve of Eratosthenes.
// Returns a heap-allocated array; sets *count. Caller must free().
unsigned long *generate_primes_up_to(unsigned long limit, int *count);

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

// Read a GMP integer from file. Tries decimal text first, then GMP binary
// (mpz_inp_raw) as fallback. Exits on failure.
void read_mpz_from_file(mpz_t num, const char *filename);

// ---------------------------------------------------------------------------
// Cube root — naive baseline (METHOD_POWM in k16)
// ---------------------------------------------------------------------------

// Compute result = value^e mod modulo, where e = 3^-1 mod (modulo-1).
// This is O(log modulo) squarings — correct but not parallelizable.
// Used as comparison baseline and for small factor cube roots in k16.
// Returns false if 3 has no inverse mod (modulo-1).
bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo);

// ---------------------------------------------------------------------------
// Cube root — factorization-based reconstruction (k16 METHOD_FACTOR)
// ---------------------------------------------------------------------------

// Reconstruct cbrt(challenge) from its factorization, using precomputed
// cube roots of each small prime:
//   result = prod_{i: exponents[i]>0} cbrt_table[i]^exponents[i]  mod M
//
// exponents[i]  = multiplicity of primes[i] in the challenge (0 if absent).
// cbrt_table[i] = primes[i]^e mod M, built once by precompute_cbrt_table().
// prime_count   = length of both arrays (6542 for k16).
//
// Complexity: O(prime_count) multiplications — no powm on large numbers.
// Inner loop parallelized with OpenMP; partial products merged at the end.
void compute_cube_root_from_factors(mpz_t result, const int *exponents,
                                    const mpz_t *cbrt_table, int prime_count,
                                    const mpz_t modulo);

// ---------------------------------------------------------------------------
// Precomputed cube root table (k16)
// ---------------------------------------------------------------------------

// Fill cbrt_table[i] = primes[i]^e mod modulo for all 6542 k16 primes.
// e = 3^-1 mod (modulo-1) is computed once; loop is OpenMP-parallelized.
// cbrt_table must be pre-allocated as mpz_t[prime_count].
void precompute_cbrt_table(mpz_t *cbrt_table, const unsigned long *primes,
                           int prime_count, const mpz_t modulo);

// Persist the table to disk (one decimal per line, header = prime_count).
// Filename convention: modulo_16_2048.txt -> modulo_16_2048.cbrt
// load_cbrt_table returns false if file is absent or prime_count mismatches.
void save_cbrt_table(const mpz_t *cbrt_table, int prime_count, const char *filename);
bool load_cbrt_table(mpz_t *cbrt_table, int prime_count, const char *filename);

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

// Verify that candidate^3 mod modulo == challenge_mod.
// Prints SUCCESS or FAILURE.
void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod,
                      const mpz_t modulo);

// ---------------------------------------------------------------------------
// Factoring helper
// ---------------------------------------------------------------------------

// Return the exact exponent of `prime` in n (i.e. largest k s.t. prime^k | n).
// Fast path: mpz_tdiv_ui checks divisibility without any GMP allocation;
// returns 0 immediately for the ~99.9% of primes that don't divide n.
int count_exponent(const mpz_t n, unsigned long prime);

// ---------------------------------------------------------------------------
// Benchmark/reporting helpers
// ---------------------------------------------------------------------------

// Compute standard deviation from raw timing samples.
// sample_stddev=true uses Bessel correction (n-1), false uses population (n).
double compute_stddev_from_values(const double *vals, int n, bool sample_stddev);

// Compute standard deviation from sum and sum-of-squares accumulators.
// sample_stddev=true uses Bessel correction (n-1), false uses population (n).
double compute_stddev_from_moments(double sum, double sumsq, int n,
                                   bool sample_stddev);

// Print one aligned timing line in the common format:
//   "    <label> avg=... ms   stddev=... ms"
void print_timing_stat_moments(const char *label,
                               double sum, double sumsq, int n,
                               int label_width, bool sample_stddev);

// Print one aligned average-only timing line in the common format:
//   "    <label> avg=... ms"
// If extra_text is non-null/non-empty, it is appended after the avg.
void print_timing_total(const char *label, double mean,
                        int label_width, const char *extra_text);