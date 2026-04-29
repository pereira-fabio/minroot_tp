#pragma once

// =============================================================================
// k_factorization.h — Factorization infrastructure for all k-solvers
//
// Centralises logic that was previously duplicated across k16.cpp, k32.cpp,
// and k32_mpi.cpp.  Each solver only needs to include this header and link
// against k_factorization.cpp to get the full factorization stack.
//
//   k16 path:   save_prime_list / load_prime_list
//   k32 path:   generate_small_primes, prime table I/O,
//               trial_divide_sieve, trial_divide_table,
//               cbrt_cache_t + helpers, compute_cube_root_from_found
// =============================================================================

#include "k_common.h"

#include <stdint.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define KF_SEG_SIZE (1 << 19)                      // 512 KiB sieve segment
#define KF_PRIME_TABLE_MAGIC 0x4B33325052494D53ULL // "K32PRIMS"
#define KF_WRITE_BATCH 65536                       // primes per fwrite batch

// ---------------------------------------------------------------------------
// k32 — small prime generator
//
// Returns all primes <= 65536 = sqrt(2^32) via Sieve of Eratosthenes.
// Used to mark composites inside 32-bit sieve segments.
// Caller must free() the returned array.
// ---------------------------------------------------------------------------

unsigned *kf_generate_small_primes(int *count);

// ---------------------------------------------------------------------------
// k32 — prime table file I/O
//
// Binary file format:
//   [uint64_t magic = K32_PRIME_TABLE_MAGIC]
//   [uint64_t count]
//   [uint32_t prime[count]]   sorted ascending, range [65537, 2^32]
//
// generate_and_save_prime_table:
//   Streams all ~203 M primes to disk in K32_WRITE_BATCH chunks.
//   One-time cost (~10 s); file is reused for all subsequent runs.
//
// load_prime_table:
//   Loads the full table into a malloc'd uint32_t array (~813 MB).
//   Used by single-node k32 --method table.
//
// load_prime_slice:
//   Binary-searches the file to load only [range_lo, range_hi].
//   Used by k32_mpi so each rank reads only its own sub-range.
// ---------------------------------------------------------------------------

bool kf_generate_and_save_prime_table(const char *path,
                                      int sp_count, const unsigned *sp,
                                      const char *log_prefix);
uint32_t *kf_load_prime_table(uint64_t *out_count, const char *path);
uint32_t *kf_load_prime_slice(uint64_t *slice_count, const char *path,
                              uint32_t range_lo, uint32_t range_hi);

// ---------------------------------------------------------------------------
// k32 — OpenMP-parallel trial division
//
// Both functions fill `found` with (prime, exponent) pairs.
// Thread-private factor lists are merged at the end — no locking in hot loop.
//
// trial_divide_sieve:
//   On-the-fly segmented sieve; 512 KiB working set stays in L2 cache.
//
// trial_divide_table:
//   Iterates a preloaded sorted uint32_t prime array (static OMP schedule).
// ---------------------------------------------------------------------------

void kf_trial_divide_sieve(const mpz_t n,
                           uint64_t range_lo, uint64_t range_hi,
                           int sp_count, const unsigned *sp,
                           kfactor_list_t *found,
                           int64_t *thread_tested = nullptr,
                           int *thread_found = nullptr,
                           int nthreads_out = 0);

void kf_trial_divide_table(const mpz_t n,
                           const uint32_t *primes, int64_t prime_count,
                           kfactor_list_t *found,
                           int64_t *thread_tested = nullptr,
                           int *thread_found = nullptr,
                           int nthreads_out = 0);

// k16 — Horner byte-level trial division
void kf_trial_divide_k16(const unsigned char *challenge_bytes, size_t nbytes,
                         const mpz_t challenge,
                         const unsigned long *primes, int prime_count,
                         int *exponents);

// ---------------------------------------------------------------------------
// k32 — cbrt cache
//
// Persists cbrt(p) mod M for each 32-bit prime found during factorisation.
// Avoids recomputing mpz_powm on repeated runs with the same modulo file
// (each powm on a 7000-bit modulus costs ~90 ms).
//
// File format (text):
//   line 1 : count
//   line i : <prime_decimal> <cbrt_decimal>
// Filename convention: modulo_32_2048.txt -> modulo_32_2048.cbrt
// ---------------------------------------------------------------------------

typedef struct
{
    unsigned long prime;
    mpz_t cbrt;
} cbrt_entry_t;

typedef struct
{
    cbrt_entry_t *items;
    int count;
    int capacity;
} cbrt_cache_t;

void kf_cbrt_cache_init(cbrt_cache_t *cache);
void kf_cbrt_cache_clear(cbrt_cache_t *cache);
// Returns index of prime in cache, or -1 if not found.
int kf_cbrt_cache_find(const cbrt_cache_t *cache, unsigned long prime);
void kf_cbrt_cache_append(cbrt_cache_t *cache, unsigned long prime,
                          const mpz_t val);
bool kf_cbrt_cache_load(cbrt_cache_t *cache, const char *path);
bool kf_cbrt_cache_save(const cbrt_cache_t *cache, const char *path);

// ---------------------------------------------------------------------------
// k32 — reconstruction from found factors
//
//   result = ∏ cbrt(p_i)^e_i  mod M
//
// Looks up cbrt(p) in the cache; computes and caches on miss.
// Sets *cache_dirty = true whenever a new entry is added.
// ---------------------------------------------------------------------------

void kf_compute_cbrt_from_factors(mpz_t result,
                                  const kfactor_list_t *found,
                                  const mpz_t modulo,
                                  cbrt_cache_t *cache,
                                  bool *cache_dirty);

// ---------------------------------------------------------------------------
// Blinding pair table (Algorithm 2 precomputation, Biryukov et al. §2)
//
// The paper separates "Precomputation" (compute r^a and r^{-1} for R random r)
// from the "Online Phase" (pick one r per group, blind, factor, unblind).
// This table implements that separation: all expensive modular arithmetic is
// done once before the bench loop starts, so the inner loop only indexes in.
//
// Each entry holds three values for a single random r ∈ [2, M-1]:
//   r      — the blinding factor itself
//   r3     — r^3 mod M  (used in: y = challenge * r^3 mod M)
//   r_inv  — r^{-1} mod M (used in: result = cbrt(y) * r^{-1} mod M)
//
// Both r3 and r_inv are precomputed so the online loop contains no powm and
// no extended-GCD call — only one mpz_mul + mpz_mod per phase.
//
// Recommended table size: R = max_attempts - 1 (one entry per non-trivial
// attempt).  The paper uses R = 2^{20}–2^{24}; for benchmarking 255 suffices.
// ---------------------------------------------------------------------------

typedef struct
{
    mpz_t r;     // Random blinding factor in [2, M-1]
    mpz_t r3;    // r^3 mod M  — precomputed blinding multiplier
    mpz_t r_inv; // r^{-1} mod M — precomputed unblinding multiplier
} blinding_pair_t;

// Allocate and fill a table of `count` independently drawn blinding pairs.
// Each pair satisfies gcd(r, M) = 1 (guaranteed since M is prime and r < M).
// Seeds GMP's MT RNG from time(NULL) XOR address entropy.
// Caller must free with kf_blinding_table_free.
blinding_pair_t *kf_blinding_table_alloc(int count, const mpz_t modulo);

// Release all mpz_t values and free the backing array.
void kf_blinding_table_free(blinding_pair_t *table, int count);

//
// Implements the smoothness-based cube root attack:
//   for attempt in 0..max_attempts:
//     r  = 1                     (attempt == 0, fast path for smooth vectors)
//         OR table[attempt-1].r  (if precomputed table provided)
//         OR rand [2,M)          (fallback: on-the-fly)
//     y  = challenge * r^3 mod M
//     factor y with sieve (prime_table == NULL) or preloaded prime table
//     if y == ∏ p_i^{e_i}  (B-smooth or almost-smooth via Pollard):
//         cbrt_y = ∏ cbrt(p_i)^{e_i}  mod M
//         result = cbrt_y * r^{-1} mod M        (unblinding)
//         return true
//   return false  (max_attempts exhausted)
//
// Parameters:
//   prime_table / prime_table_count : NULL/0 → sieve; non-NULL → table scan.
//   cache / cache_dirty             : caller manages init, load and save.
//   attempts_out                    : if non-NULL receives the attempt count.
//   table / table_size              : precomputed pairs from
//                                     kf_blinding_table_alloc(); pass NULL
//                                     to generate r on-the-fly (old behaviour).
// ---------------------------------------------------------------------------

bool kf_factorize_with_blinding(
    mpz_t result,
    const mpz_t challenge,
    const mpz_t modulo,
    int sp_count, const unsigned *sp,
    const uint32_t *prime_table, int64_t prime_table_count,
    cbrt_cache_t *cache, bool *cache_dirty,
    int max_attempts,
    int *attempts_out,
    const blinding_pair_t *table, // precomputed pairs; NULL = on-the-fly
    int table_size);              // number of valid entries in table

// ---------------------------------------------------------------------------
// k64 — Pollard's rho factorization
// ---------------------------------------------------------------------------

bool kf_is_prime(const mpz_t n);
bool kf_pollard_rho_brent(mpz_t factor, const mpz_t n, unsigned long c_val);
void kf_factor_recursive(const mpz_t n, kfactor_list_t *factors,
                         unsigned long c_start);
void kf_factor_challenge(const mpz_t challenge, kfactor_list_t *out);

// ---------------------------------------------------------------------------
// k16 — Precompute and reconstruct
// ---------------------------------------------------------------------------

void kf_precompute_cbrt_table_k16(mpz_t *cbrt_table, const unsigned long *primes,
                                  int prime_count, const mpz_t modulo);
void kf_save_cbrt_table_k16(const mpz_t *cbrt_table, int prime_count, const char *filename);
bool kf_load_cbrt_table_k16(mpz_t *cbrt_table, int prime_count, const char *filename);
void kf_compute_cube_root_from_factors_k16(mpz_t result, const int *exponents,
                                           const mpz_t *cbrt_table, int prime_count,
                                           const mpz_t modulo);