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
//   Iterates a preloaded sorted uint32_t prime array (dynamic OMP schedule).
//
// Cofactor sharing. Every factor divided out of y makes later tests cheaper
// (mpz_tdiv_ui walks every limb), but a thread only sees the factors it or
// its rank's other threads found. Both functions therefore scan in
// KF_SHARE_PHASES phases; between phases the threads refresh their copy of
// the shrinking cofactor, and when `share` is non-null the master thread
// hands it this rank's factors found since the last checkpoint and divides
// out every factor it returns (other ranks' finds). A smooth y then shrinks
// for everyone as the scan advances: measured 124.7 -> 57.3 ms on a 768-
// factor y with one shared snapshot vs fixed snapshots.
//
// `share` is a collective in the MPI solvers (ksc_share_found): every rank
// must call the function the same number of times, each making exactly
// KF_SHARE_PHASES - 1 calls to it, whatever its slice holds. Pass it only
// where all ranks trial-divide in lockstep. `found` only ever lists this
// rank's own factors; shared ones are only divided out of the cofactor.
// ---------------------------------------------------------------------------

#define KF_SHARE_PHASES 8

// mine: this rank's factors since the last call; all: every rank's (appended).
typedef void (*kf_share_fn)(const kfactor_list_t *mine, kfactor_list_t *all);

void kf_trial_divide_sieve(const mpz_t n,
                           uint64_t range_lo, uint64_t range_hi,
                           int sp_count, const unsigned *sp,
                           kfactor_list_t *found,
                           int64_t *thread_tested = nullptr,
                           int *thread_found = nullptr,
                           int nthreads_out = 0,
                           kf_share_fn share = nullptr);

void kf_trial_divide_table(const mpz_t n,
                           const uint32_t *primes, int64_t prime_count,
                           kfactor_list_t *found,
                           int64_t *thread_tested = nullptr,
                           int *thread_found = nullptr,
                           int nthreads_out = 0,
                           kf_share_fn share = nullptr);

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
    // Open-addressing index prime -> items[] position, kept in step by
    // kf_cbrt_cache_append. slot holds index+1; 0 marks an empty slot.
    int *slots;
    int slot_mask; // slot capacity - 1 (a power of two), or -1 when empty
} cbrt_cache_t;

void kf_cbrt_cache_init(cbrt_cache_t *cache);
void kf_cbrt_cache_clear(cbrt_cache_t *cache);
// Returns index of prime in cache, or -1 if not found. O(1) expected via the
// hash index; safe to call concurrently as long as nothing appends meanwhile.
int kf_cbrt_cache_find(const cbrt_cache_t *cache, unsigned long prime);
void kf_cbrt_cache_append(cbrt_cache_t *cache, unsigned long prime,
                          const mpz_t val);
bool kf_cbrt_cache_load(cbrt_cache_t *cache, const char *path);
bool kf_cbrt_cache_save(const cbrt_cache_t *cache, const char *path);

// ---------------------------------------------------------------------------
// Every lookup goes through kf_cbrt_cache_find. It was a LINEAR scan, so a
// challenge with n factors against a C-entry cache cost O(n*C) comparisons,
// and the cache persists and grows across challenges. It is now a hash index;
// `probes` counts hash slots inspected, so probes/lookup near 1 confirms O(1).
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
// per-factor phase (cube root: cache lookup, or mpz_powm on a miss; then the
// e_i-th power) and a combine phase (product mod M + cache append). Separating
// them shows whether reconstruction is bound by exponentiation or by the
// accumulation. The combine was strictly serial before kf_product_mod_tree;
// the field keeps its name so RECONPROF lines stay comparable.
// ---------------------------------------------------------------------------

typedef struct
{
    double parallel_ms; // phase 1: per-factor cbrt across threads
    double serial_ms;   // phase 2: product mod M (tree) + cache append
    long factors;       // total factors processed
    long calls;         // number of reconstruction calls accumulated
} kf_recon_prof_t;

void kf_recon_prof_reset(void);
const kf_recon_prof_t *kf_recon_prof_get(void);

// ---------------------------------------------------------------------------
// Cost-balanced split of a trial-division range [lo, hi] into `parts`
// contiguous pieces; piece `idx` is returned as [*out_lo, *out_hi] (empty when
// *out_lo > *out_hi). `limbs` is the size of the number being divided (64-bit
// limbs) and `sieve` says whether the range is sieved (sieve method) or read
// from a prime table. Equal-width pieces are unbalanced: low ranges hold more
// primes (density 1/ln x), primes below 2^31 are tested in pairs at under half
// the cost each, and the sieve adds a fixed cost per integer. The model, per
// integer x (constants measured single-threaded, see factorization.cpp):
//
//     cost(x) = sieve * KF_SIEVE_NS + w(x) * u(limbs) / ln x
//     u(limbs) = KF_TD_BASE_NS + KF_TD_LIMB_NS * limbs     one prime test
//     w(x) = KF_PAIRED_COST below 2^31, 1 above
//
// Every caller computes the same boundaries independently.
// ---------------------------------------------------------------------------

#define KF_SIEVE_NS 2.1
#define KF_TD_BASE_NS 5.0
#define KF_TD_LIMB_NS 0.55
#define KF_PAIRED_COST 0.45
void kf_cost_split(uint64_t lo, uint64_t hi, int parts, int idx, int limbs, bool sieve,
                   uint64_t *out_lo, uint64_t *out_hi);

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
// per-factor cube roots concurrently and combines them with
// kf_product_mod_tree; output is identical to the serial version.
// Cache reads happen in the parallel region, cache writes serially afterward.
void kf_compute_cbrt_from_factors_omp(mpz_t result,
                                      const kfactor_list_t *found,
                                      const mpz_t modulo,
                                      cbrt_cache_t *cache,
                                      bool *cache_dirty,
                                      unsigned long exponent);

// result = vals[0] * ... * vals[n-1] mod M (1 when n == 0), computed as
// per-thread chunk products combined pairwise on up to nthreads OpenMP
// threads: n/T + ceil(log2 T) dependent mulmods instead of n-1. Identical to
// the serial product; falls back to it without OpenMP or for small n.
void kf_product_mod_tree(mpz_t result, mpz_t *vals, int n,
                         const mpz_t modulo, int nthreads);

// True when `mulmods` |M|-bit modular multiplications are enough work to be
// worth waking an OpenMP team for (~1 ms serial). Below it, thread start-up and
// barrier stalls on a busy machine cost more than they save.
bool kf_parallel_worth(double mulmods, const mpz_t modulo);

// Where kf_compute_cbrt_from_factors_omp sends its final product. The default
// is kf_product_mod_tree; the CUDA solvers can install the device tree
// (gpu_recon.cuh, --recon device). nullptr restores the default.
typedef void (*kf_product_fn)(mpz_t result, mpz_t *vals, int n,
                              const mpz_t modulo, int nthreads);
void kf_set_product_backend(kf_product_fn fn);

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