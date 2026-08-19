#pragma once

// =============================================================================
// factorization.h — Factorization infrastructure for all k-solvers
//
// Centralises logic that was previously duplicated across k16.cpp, k32.cpp,
// and alg2_k32.cpp.  Each solver only needs to include this header and link
// against factorization.cpp to get the full factorization stack.
//
//   k16 path:   save_prime_list / load_prime_list
//   k32 path:   generate_small_primes, prime table I/O,
//               trial_divide_sieve, trial_divide_table,
//               cbrt_cache_t + helpers, compute_cube_root_from_found
// =============================================================================

#include "common.h"

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

// ---------------------------------------------------------------------------
// k35 upper range — uint64_t prime table for (2^32, 2^35]
//
// The k35 smoothness bound B = 2^35 covers primes that do not fit in uint32_t
// (those above 2^32).  We store the upper range (2^32, 2^35] in a separate
// binary file k35_upper_primes.bin using uint64_t entries.
//
// The lower range [65537, 2^32] is handled by the existing k32_primes.bin
// infrastructure (uint32_t).  k35 table method loads both slices.
//
// File format (k35_upper_primes.bin):
//   [uint64_t magic]  = K35U_PRIME_TABLE_MAGIC
//   [uint64_t count]  = total number of primes in file
//   [uint64_t * count] = primes in (2^32, 2^35], sorted ascending
// ---------------------------------------------------------------------------

#define K35U_PRIME_TABLE_MAGIC 0x4B333555505249DULL // "K35UPRI"
#define K35U_DEFAULT_PATH "data/k35_upper_primes.bin"

// Generate and write k35_upper_primes.bin (primes in (2^32, 2^35]).
// Requires small primes sp[] up to sqrt(2^35) ~ 2^17.5 (use 2^18).
bool kf_generate_and_save_prime_table_u64(const char *path,
                                          uint64_t range_lo,
                                          uint64_t range_hi,
                                          int sp_count,
                                          const unsigned *sp,
                                          const char *log_prefix);

// Load the slice of k35_upper_primes.bin covering [lo, hi].
// Returns malloc'd uint64_t array; caller must free.  *count set on return.
uint64_t *kf_load_prime_slice_u64(int64_t *count, const char *path,
                                  uint64_t lo, uint64_t hi);

// Trial-divide n by a preloaded uint64_t prime slice (OMP parallel).
void kf_trial_divide_table_u64(const mpz_t n,
                               const uint64_t *primes, int64_t prime_count,
                               kfactor_list_t *found,
                               int64_t *thread_tested = nullptr,
                               int *thread_found = nullptr,
                               int nthreads_out = 0);

// ---------------------------------------------------------------------------
// Delta/gap-encoded prime table (advisor M2 — see wiki delta-prime-table)
//
// A sorted prime list compresses ~4-8x by storing gaps, not values: every
// prime gap below 2^35 is <= 456, so (gap/2) <= 228 < 256 fits in ONE byte.
// The layout keeps O(1) random *block* access, so the schedule(static) trial-
// division loop and the per-rank / per-GPU range split are all preserved.
//
// File format (.dlt):
//   [u64 magic][u64 count][u32 block_size B][u32 width]      (width = 4 or 8)
//   then ceil(count/B) blocks; block g holds len = min(B, count - g*B) primes:
//     [anchor : width bytes = p[g*B]] [u8 gaphalf[len-1]]    p[i]=p[i-1]+2*gaphalf
//   Full blocks are fixed size (width + B-1 bytes), so block g starts at byte
//   HDR(24) + g*(width+B-1): O(1) seek. Each block re-anchors its first prime,
//   so blocks decode independently (embarrassingly parallel; one per warp).
//
//   width 4  <- k32 flat table  [65537, 2^32]   (813 MB  -> ~204 MB, ~4x)
//   width 8  <- k35 upper table (2^32, 2^35]    (9.6 GB  -> ~1.2 GB, ~8x)
// ---------------------------------------------------------------------------

#define KF_DELTA_MAGIC_U32 0x4B3332444C544131ULL // "K32DLTA1"
#define KF_DELTA_MAGIC_U64 0x4B3335444C544131ULL // "K35DLTA1"
#define KF_DELTA_BLOCK 1024                      // primes per block (0.4% overhead)
#define KF_DELTA_HDR 24                          // header bytes

// A loaded (sub-)range of a delta table, ready for trial division.
typedef struct
{
    uint8_t *bytes;              // owned: concatenated blocks for the loaded slice
    uint64_t total_count;        // primes in the whole file (for last-block length)
    uint64_t first_block_index;  // global index of the first block in `bytes`
    uint32_t n_blocks;           // number of blocks held in `bytes`
    uint32_t block_size;         // primes per full block (B)
    int width;                   // anchor width in bytes: 4 (k32) or 8 (k35 upper)
    uint64_t range_lo, range_hi; // inclusive prime-value bounds this slice serves
} kf_delta_slice_t;

// One-time conversion: flat prime table -> delta table. Returns false (and
// removes the output) on I/O error or if any gap/2 exceeds 255 (which would
// break the 1-byte scheme — a hard correctness guard, never expected to fire).
bool kf_convert_prime_table_to_delta(const char *flat_path, const char *delta_path,
                                     const char *log_prefix); // u32 (k32)
bool kf_convert_prime_table_to_delta_u64(const char *flat_path, const char *delta_path,
                                         const char *log_prefix); // u64 (k35 upper)

// Load the delta blocks covering [range_lo, range_hi] (inclusive) into *out.
// Mirrors kf_load_prime_slice: exact prime-value boundaries (trial division
// filters the over-read at each end), so adjacent ranks never overlap.
// Returns false on error; on success free with kf_delta_slice_free.
bool kf_load_delta_slice(kf_delta_slice_t *out, const char *path,
                         uint64_t range_lo, uint64_t range_hi);

void kf_delta_slice_free(kf_delta_slice_t *s);

// Trial-divide n by a delta slice (OMP over blocks; each block is decoded into
// a stack buffer, then the same inner mpz loop as kf_trial_divide_table runs).
// Handles both widths. Output is identical to the flat table method (verified
// by tests/delta_table_test).
void kf_trial_divide_delta(const mpz_t n, const kf_delta_slice_t *slice,
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
// Every lookup goes through kf_cbrt_cache_find, which is a LINEAR scan, so a
// lookup costs O(cache->count) comparisons. Counting probes as well as hits
// exposes that cost: a challenge with n distinct prime factors performs ~n
// lookups against an n-entry cache, i.e. O(n^2) comparisons per reconstruction.
//
// Counters are per-thread, indexed by omp_get_thread_num, and each block is
// padded to 64 bytes so the OpenMP reconstruction loop does not false-share.
// Accounting is one increment per LOOKUP (not per comparison), so the counters
// do not perturb the scan they measure.
// ---------------------------------------------------------------------------

#define KF_CBRT_MAX_THREADS 256

typedef struct
{
    long hits;    // lookups that found the prime already cached
    long misses;  // lookups that fell through to a full mpz_powm
    long probes;  // cache entries compared (the linear-scan cost)
    long _pad[5]; // pad to 64 B — one counter block per cache line
} kf_cbrt_stat_t;

void kf_cbrt_stats_reset(void);
// Aggregate across all thread slots. Any pointer may be NULL.
void kf_cbrt_stats_total(long *hits, long *misses, long *probes);
// Number of thread slots that recorded at least one lookup.
int kf_cbrt_stats_threads_used(void);
const kf_cbrt_stat_t *kf_cbrt_stats_raw(void);
// Per-thread table + summary, plus greppable "CBRTSTAT|" lines for the
// analysis pipeline. `tag` labels the run (e.g. "k32 24576b e3").
void kf_cbrt_stats_report(const char *tag);

// ---------------------------------------------------------------------------
// Reconstruction phase profile. kf_compute_cbrt_from_factors_omp splits into a
// parallel phase (per-factor cube root: cache lookup, or mpz_powm on a miss)
// and a strictly serial phase (product mod M + cache append). Separating them
// shows whether reconstruction is bound by exponentiation or by the serial
// accumulation — the question that decides whether a device port could help.
// ---------------------------------------------------------------------------

typedef struct
{
    double parallel_ms; // phase 1: per-factor cbrt across threads
    double serial_ms;   // phase 2: serial product mod M + cache append
    long factors;       // total factors processed
    long calls;         // number of reconstruction calls accumulated
} kf_recon_prof_t;

void kf_recon_prof_reset(void);
const kf_recon_prof_t *kf_recon_prof_get(void);

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
                                  bool *cache_dirty,
                                  unsigned long exponent);

// OpenMP-parallel variant of the above, for single-process (IRIS GPU) solvers
// where reconstruction is not distributed across MPI ranks. Computes the
// per-factor cube roots concurrently; output is identical to the serial version.
// Cache reads happen in the parallel region, cache writes serially afterward.
void kf_compute_cbrt_from_factors_omp(mpz_t result,
                                      const kfactor_list_t *found,
                                      const mpz_t modulo,
                                      cbrt_cache_t *cache,
                                      bool *cache_dirty,
                                      unsigned long exponent);

// ---------------------------------------------------------------------------
// Blinding pair table (Algorithm 2 precomputation, Biryukov et al. §2)
//
// The paper separates "Precomputation" (compute r^a and r^{-1} for R random r)
// from the "Online Phase" (pick one r per group, blind, factor, unblind).
// This table implements that separation: all expensive modular arithmetic is
// done once before the bench loop starts, so the inner loop only indexes in.
//
// Each entry holds three values for a single random r ∈ [2, M-1]:
//   r       — the blinding factor itself
//   r_blind — r^e mod M  (used in: y = challenge * r^e mod M)
//             e is the root exponent (3 for cube root, 5 for 5th root)
//   r_inv   — r^{-1} mod M (used in: result = nth_root(y) * r^{-1} mod M)
//
// Both r_blind and r_inv are precomputed so the online loop contains no powm
// and no extended-GCD call — only one mpz_mul + mpz_mod per phase.
//
// Recommended table size: R = max_attempts - 1 (one entry per non-trivial
// attempt).  The paper uses R = 2^{20}–2^{24}; for benchmarking 255 suffices.
// ---------------------------------------------------------------------------

typedef struct
{
    mpz_t r;       // Random blinding factor in [2, M-1]
    mpz_t r_blind; // r^e mod M  — precomputed blinding multiplier
    mpz_t r_inv;   // r^{-1} mod M — precomputed unblinding multiplier
} blinding_pair_t;

// Allocate and fill a table of `count` independently drawn blinding pairs.
// Each pair satisfies gcd(r, M) = 1 (guaranteed since M is prime and r < M).
// Seeds GMP's MT RNG from time(NULL) XOR address entropy.
// Caller must free with kf_blinding_table_free.
blinding_pair_t *kf_blinding_table_alloc(int count, const mpz_t modulo,
                                         unsigned long exponent);

// Release all mpz_t values and free the backing array.
void kf_blinding_table_free(blinding_pair_t *table, int count);

// ---------------------------------------------------------------------------
// k64 — Pollard's rho factorization
// ---------------------------------------------------------------------------

bool kf_is_prime(const mpz_t n);
bool kf_pollard_rho_brent(mpz_t factor, const mpz_t n, unsigned long c_val);
void kf_factor_recursive(const mpz_t n, kfactor_list_t *factors,
                         unsigned long c_start);
void kf_factor_challenge(const mpz_t challenge, kfactor_list_t *out);

// ---------------------------------------------------------------------------
// k64 — B-smooth test via Pollard's rho
//
// Attempts to fully factor n using Pollard's rho (OMP parallel).
// Returns true if n is completely factored and every prime factor fits in
// 64 bits (i.e. n is 2^64-smooth). Returns false if a factor > 2^64 is
// found or factoring fails within the iteration limit.
//
// On success, factors contains the complete (prime, exponent) factorization.
// On failure, factors contains whatever partial factorization was found.
//
// c_start: first c value for Pollard's rho (rank-specific seed for racing).
// ---------------------------------------------------------------------------
bool kf_try_factor_b64smooth(const mpz_t n,
                             kfactor_list_t *factors,
                             unsigned long c_start);

// ---------------------------------------------------------------------------
// k16 — Precompute and reconstruct
// ---------------------------------------------------------------------------

void kf_precompute_cbrt_table_k16(mpz_t *cbrt_table, const unsigned long *primes,
                                  int prime_count, const mpz_t modulo,
                                  unsigned long exponent);
void kf_save_cbrt_table_k16(const mpz_t *cbrt_table, int prime_count, const char *filename);
bool kf_load_cbrt_table_k16(mpz_t *cbrt_table, int prime_count, const char *filename);
void kf_compute_cube_root_from_factors_k16(mpz_t result, const int *exponents,
                                           const mpz_t *cbrt_table, int prime_count,
                                           const mpz_t modulo);