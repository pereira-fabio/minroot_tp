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
                           kfactor_list_t *found);

void kf_trial_divide_table(const mpz_t n,
                           const uint32_t *primes, int64_t prime_count,
                           kfactor_list_t *found);

// k16 — Horner byte-level trial division
void kf_trial_divide_k16(const unsigned char *challenge_bytes, size_t nbytes,
                         const mpz_t challenge,
                         const unsigned long *primes, int prime_count,
                         int *exponents);

// ---------------------------------------------------------------------------
// k64 — Pollard's rho factorization
// ---------------------------------------------------------------------------

bool kf_is_prime(const mpz_t n);
bool kf_pollard_rho_brent(mpz_t factor, const mpz_t n, unsigned long c_val);
void kf_factor_recursive(const mpz_t n, kfactor_list_t *factors,
                         unsigned long c_start);
void kf_factor_challenge(const mpz_t challenge, kfactor_list_t *out);