#pragma once

// =============================================================================
// solver_common.h — Shared MPI+OpenMP infrastructure for cube root solvers
//
// Centralises logic duplicated between k16_mpi.cpp and k32_mpi.cpp:
//   - Challenge/modulo broadcast
//   - JSON results initialisation
//   - MPI_Gather of partial cbrt products + combine/unblind
//   - Per-rank prime stats gather
//   - Timing report + powm baseline
//   - Precomputation guarantees (primes + cbrt tables/caches)
//
// Both solvers include this header and link solver_common.cpp.
// =============================================================================

#include "common.h"
#include "factorization.h"
#include <mpi.h>

// ---------------------------------------------------------------------------
// Data directory — all precomputed files live here
// ---------------------------------------------------------------------------
#define KSC_DATA_DIR "data"
#define KSC_CBRT_DIR "data/cbrt" // cbrt tables live here, separate from primes/challenges

// ---------------------------------------------------------------------------
// e-th root exponent — compute e_inv = exponent^{-1} mod (modulo-1)
//
// Extracted from the repetitive mpz_invert boilerplate in every solve_one.
// e_inv_out must be pre-initialised with mpz_init; caller clears with mpz_clear.
// ---------------------------------------------------------------------------
void ksc_compute_e_inv(mpz_t e_inv_out, const mpz_t modulo, unsigned long exponent);

// ---------------------------------------------------------------------------
// cbrt cache path construction
//
// Builds the canonical path:  KSC_CBRT_DIR/<modulo_stem><suffix>.<rank>
// (or without the rank suffix when rank < 0, e.g. for the k16 shared table).
//
// buf must be at least KSC_CBRT_PATH_MAX bytes.  Truncation is silently
// avoided: modulo_file stems are typically < 64 bytes.
// ---------------------------------------------------------------------------
#define KSC_CBRT_PATH_MAX 600

void ksc_make_cbrt_cache_path(char *buf, size_t bufsz,
                              const char *cbrt_dir,
                              const char *modulo_file,
                              const char *suffix,
                              int rank);

// ---------------------------------------------------------------------------
//
// Builds  <cbrt_dir>/cbrt_<size>.cbrt  — the ONE full cache per (variant, size)
// produced by analysis/merge_cbrt_cache.py.  <size> is parsed from the modulo
// filename (modulo_<...>_<size>.txt or modulo_<...>_<size>_e<N>.txt).  The
// solvers load this (all ranks, read-only) in preference to the per-rank build:
// it contains every prime any challenge needs, so reconstruction lookups all
// hit and no per-rank build / cold run is required.  (Variants have different
// moduli — the prime generator races — so the cache is per-variant, not shared
// across variants.)
// ---------------------------------------------------------------------------
void ksc_shared_cbrt_path(char *buf, size_t bufsz,
                          const char *cbrt_dir, const char *modulo_file);

// ---------------------------------------------------------------------------
// Challenge header — print challenge/modulo/rank info on rank 0
// ---------------------------------------------------------------------------
void ksc_print_challenge_header(const mpz_t challenge, const mpz_t modulo,
                                int rank, int nranks);

// ---------------------------------------------------------------------------
// Challenge / modulo broadcast
// Rank 0 reads files; all ranks receive via Bcast.
// Returns true on success, calls MPI_Abort + returns false on failure.
// ---------------------------------------------------------------------------
// Does an input file exist? Rank 0 checks and the answer is broadcast, so every
// rank takes the same branch -- a collective decision, since a rank that carried
// on alone would hang the others at the next barrier.
//
// Used by --all to skip a size whose files are absent. Without it a missing
// input reaches ksc_broadcast_challenge_modulo, which calls MPI_Abort and takes
// the whole sweep down on the first gap.
bool ksc_input_exists(const char *path, int rank);

bool ksc_broadcast_challenge_modulo(
    mpz_t challenge, mpz_t modulo,
    const char *challenge_file, const char *modulo_file,
    int rank);

// ---------------------------------------------------------------------------
// JSON results — initialise all fields common to both solvers
// ---------------------------------------------------------------------------
void ksc_init_json_results(
    benchmark_results_t *r,
    int smoothness_bits,
    const char *method_name,
    int mpi_ranks, int threads_per_rank,
    const mpz_t challenge, int rank);

// ---------------------------------------------------------------------------
// Log-depth product reduction — called by EVERY rank
//
// result (rank 0 only) = product of every rank's `local`, reduced mod *modulo
// when modulo is non-null, exact otherwise. Uses MPI_Reduce with a
// user-defined op, so the multiplications form a ceil(log2 nranks)-deep tree
// instead of nranks-1 serial steps on rank 0. Values travel as nbytes
// zero-padded big-endian buffers; an exact product that overflows nbytes
// saturates to all 0xFF bytes (never equal to, or a divisor of, a value < M).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// kf_share_fn for the MPI solvers: every rank's newly found (prime, exponent)
// pairs to every rank (MPI_Allgather of counts, then MPI_Allgatherv). A
// collective: pass it to kf_trial_divide_* only where all ranks trial-divide
// in lockstep. Called from the OpenMP master thread (MPI_THREAD_FUNNELED).
// ---------------------------------------------------------------------------
void ksc_share_found(const kfactor_list_t *mine, kfactor_list_t *all);


void ksc_reduce_product(mpz_t result,
                        const mpz_t local,
                        mpz_srcptr modulo,
                        size_t nbytes,
                        int rank);

// ---------------------------------------------------------------------------
// Partial cbrt combine
//
// Tree-reduces every rank's local_cbrt into result mod M on rank 0
// (ksc_reduce_product), then unblinds: if attempt > 0, multiplies by
// blind_table[attempt-1].r_inv.
//
// t_gat_ms_out is always 0: there is no separate gather step any more.
// t_com_ms_out is the whole reduce (communication + mulmods) plus the unblind,
// on rank 0 (others: ignored).
// ---------------------------------------------------------------------------
void ksc_gather_and_combine_cbrt(
    mpz_t result,
    const mpz_t local_cbrt,
    const mpz_t modulo,
    size_t cbrt_bytes,
    int rank, int nranks,
    int attempt,
    const blinding_pair_t *blind_table, int blind_table_size,
    double *t_gat_ms_out,
    double *t_com_ms_out);

// ---------------------------------------------------------------------------
// Local cube root reconstruction — called by EVERY rank
//
// Computes  partial = ∏ cbrt(p_i)^e_i  mod M  for this rank's local factors.
// Phase 1 (sequential): look up cbrt(p) per factor; misses (mpz_powm) in parallel.
// Phase 2 (parallel):   exponentiate each cbrt by its exponent (OMP, when worth it).
// Phase 3 (parallel):   product tree mod M (kf_product_mod_tree).
// ---------------------------------------------------------------------------
void ksc_compute_local_cbrt(
    mpz_t result,
    const kfactor_list_t *factors,
    const mpz_t e,
    const mpz_t modulo,
    cbrt_cache_t *cache,
    bool *cache_dirty,
    double *hit_ms_out = nullptr,
    double *miss_ms_out = nullptr,
    int *hits_out = nullptr,
    int *misses_out = nullptr,
    bool record = true);

// ---------------------------------------------------------------------------
// Per-rank prime stats (call once, on first warm run bench == 1)
// ---------------------------------------------------------------------------
void ksc_gather_prime_stats(
    benchmark_results_t *r,
    int local_found,
    int64_t local_tested,
    int rank, int nranks);

// ---------------------------------------------------------------------------
// Timing report — rank 0 prints, all accumulators already gathered
// ---------------------------------------------------------------------------
void ksc_print_timing_report(
    const bench_accum_t *fac_acc,
    const bench_accum_t *loc_acc,
    const bench_accum_t *gat_acc,
    const bench_accum_t *com_acc,
    const bench_accum_t *recon_acc);

// ---------------------------------------------------------------------------
// Powm baseline + speedup (rank 0 only)
// Runs KSC_POWM_BASELINE_REPS sequential powm calls, stores the mean in
// r->powm_baseline_ms, prints speedup vs reconstruction-only and end-to-end.
// ---------------------------------------------------------------------------
void ksc_run_and_report_powm_baseline(
    benchmark_results_t *r,
    const mpz_t challenge,
    const mpz_t modulo,
    const bench_accum_t *recon_acc,
    const bench_accum_t *fac_acc,
    int rank,
    unsigned long exponent);

// ---------------------------------------------------------------------------
// k16 precomputation
//
// ksc_ensure_k16_primes:
//   Loads data/k16_primes.bin if present; otherwise generates via Eratosthenes
//   and saves. Called on every rank independently (file is identical everywhere).
//   Returns heap-allocated array; caller must free().
//
// ksc_ensure_k16_cbrt:
//   Rank 0 loads or computes cbrt table from data/modulo_16_<bits>.cbrt,
//   then broadcasts every entry to all ranks. After return, cbrt_table is
//   valid on ALL ranks with no mpz_powm needed in the bench loop.
//   cbrt_table must be pre-allocated as mpz_t[prime_count] with mpz_init().
// ---------------------------------------------------------------------------
unsigned long *ksc_ensure_k16_primes(int *prime_count_out);

void ksc_ensure_k16_cbrt(
    mpz_t *cbrt_table,
    const unsigned long *primes,
    int prime_count,
    const mpz_t modulo,
    const char *modulo_file,
    const char *cbrt_dir,
    int rank,
    unsigned long exponent);

// ---------------------------------------------------------------------------
// k32 precomputation
//
// ksc_ensure_k32_primes:
//   Ensures data/k32_primes.bin exists (rank 0 generates if absent).
//   All ranks barrier after. Returns path string (static storage).
// ---------------------------------------------------------------------------
const char *ksc_ensure_k32_primes(int sp_count, const unsigned *sp, int rank);

// ---------------------------------------------------------------------------
// cbrt cache precomputation — unified for k32, k35, alg3, alg4
//
// Loads the cache from cache_path if it exists. Otherwise runs one untimed
// factoring pass over [my_lo, my_hi] to discover all prime factors in the
// challenge, computes cbrt for each, and saves the result.
//
// prime_slice != NULL → use table method (kf_trial_divide_table)
// prime_slice == NULL → use sieve method (kf_trial_divide_sieve)
//
// check_small_primes = true → rank 0 also checks sp[] entries below my_lo
//   before the main sieve/table pass. Used by k35: its range starts at 65537
//   but challenges may have factors from smaller primes as well.
// ---------------------------------------------------------------------------
void ksc_ensure_cbrt_cache(
    cbrt_cache_t *cache,
    bool *cache_dirty,
    const char *cache_path,
    const mpz_t challenge,
    const mpz_t modulo,
    const mpz_t e_inv,
    uint64_t my_lo, uint64_t my_hi,
    int sp_count, const unsigned *sp,
    const uint32_t *prime_slice, int64_t slice_count,
    bool check_small_primes,
    int rank);

// ---------------------------------------------------------------------------
// k35 upper-range prime table management
// ---------------------------------------------------------------------------

// Ensures data/k35_upper_primes.bin exists (rank 0 generates if missing).
// All ranks synchronise via MPI_Barrier before returning.
// sp/sp_count must cover primes up to sqrt(2^35) ~ 185363 (use 2^18 limit).
const char *ksc_ensure_k35_primes(int sp_count, const unsigned *sp, int rank);

// ---------------------------------------------------------------------------
// Algorithm 2 — B-smooth completeness check (k32, k35)
//
// Gathers each rank's partial factor product, combines on rank 0, and checks
// combined_product == blinded (fully B-smooth, cofactor = 1).
// Returns 1 if complete, 0 if the blinded challenge is not fully B-smooth.
// No cofactor is broadcast — Algorithm 2 does not have one.
// ---------------------------------------------------------------------------
int ksc_alg2_broadcast_complete(
    const mpz_t blinded,
    const mpz_t local_prod,
    size_t cbrt_bytes,
    int rank, int nranks);

// ---------------------------------------------------------------------------
// Algorithm 3 — almost-smooth completeness check + cofactor broadcast
//
// All ranks call this function after trial division.  Each passes its local
// factor product (local_prod = ∏ p_i^{e_i} for this rank's found primes).
//
// Internally performs MPI_Gather of all local products, then rank 0:
//   1. Multiplies them into combined_prod
//   2. Computes cofactor = blinded / combined_prod
//   3. Classifies: cofactor==1 → B-smooth; cofactor is prime ≤ 65 bits →
//      almost-smooth (one medium factor q0 ∈ (2^32, 2^65]); else → retry
//
// Broadcasts `complete` (1/0) and the cofactor bytes to all ranks.
// cofactor_out is set on ALL ranks on return (1 if B-smooth, q0 if almost-smooth).
// Returns 1 if the challenge is fully accounted for (proceed), 0 if retry.
// ---------------------------------------------------------------------------
int ksc_alg3_broadcast_cofactor(
    mpz_t cofactor_out,
    const mpz_t blinded,
    const mpz_t local_prod,
    size_t cbrt_bytes,
    int rank, int nranks,
    double *primality_ms = nullptr);

// ---------------------------------------------------------------------------
// Algorithm 4 — 2^64-smooth completeness check + medium factor broadcast
//
// Extension of ksc_alg3_broadcast_cofactor: accepts composite cofactors ≤ 2^64
// by factoring them with Pollard's rho (rank 0 only).
//
// medium_factors_out receives all medium prime factors on ALL ranks on success.
// Returns 1 if the challenge is fully accounted for, 0 if retry.
// ---------------------------------------------------------------------------
int ksc_alg4_broadcast_cofactor(
    kfactor_list_t *medium_factors_out,
    const mpz_t blinded,
    const mpz_t local_prod,
    size_t cbrt_bytes,
    int rank, int nranks);

// ---------------------------------------------------------------------------
// ksc_fold_cofactor_cbrt — fold cbrt(cofactor) into local_cbrt (rank 0 only)
//
// Called after ksc_compute_local_cbrt when the blinded challenge is
// almost-smooth (cofactor > 1).  Checks the cbrt cache first; computes
// mpz_powm only on the first encounter (cache miss).
// No-op if cofactor == 1 (B-smooth challenge).
// ---------------------------------------------------------------------------
void ksc_fold_cofactor_cbrt(
    mpz_t local_cbrt,
    const mpz_t cofactor,
    const mpz_t e_inv,
    const mpz_t modulo,
    cbrt_cache_t *cache,
    bool *cache_dirty);

// ---------------------------------------------------------------------------
// ksc_recon_ctx_t — per-challenge context for ksc_reconstruct_y
//
// Groups the 13 parameters that are fixed for an entire challenge (modulo,
// cbrt bounds, cache, prime range, blinding table, MPI topology).  Construct
// once per solve_one, pass as a const pointer to each ksc_reconstruct_y call.
// ---------------------------------------------------------------------------
typedef struct
{
    const mpz_t *modulo;
    size_t cbrt_bytes;
    const mpz_t *e_inv;
    cbrt_cache_t *cache;
    bool *cache_dirty;
    uint64_t my_lo, my_hi;
    int sp_count;
    const unsigned *sp;
    blinding_pair_t *blind_table;
    int blind_table_size;
    int rank, nranks;
    run_method_t method;   // METHOD_TABLE → use prime_slice in Phase 2 sieve
    uint32_t *prime_slice; // preloaded slice for table method (may be NULL)
    int64_t slice_count;
} ksc_recon_ctx_t;

// ---------------------------------------------------------------------------
// ksc_reconstruct_y — Phase 2 sieve + cbrt for a known candidate y
//
// Sieves y in [my_lo, my_hi], checks (almost-)smoothness via
// ksc_alg3_broadcast_cofactor, computes distributed cbrt, and accumulates
// timing.  Used by alg7 after Phase 1 selects the smooth candidate.
//
// Returns 1 if y is (almost-)smooth and cbrt was gathered into result_out.
// Returns 0 if cofactor is too large; caller records failure stats + continues.
//
// cofactor_out: cofactor value on all ranks (1 = B-smooth).
// fac2_dt_out:  time for sieve + cofactor check (ms), valid on all ranks.
// recon_dt_out: fac2_dt + cbrt + gather + combine (ms), valid on rank 0 only.
// ---------------------------------------------------------------------------
int ksc_reconstruct_y(
    mpz_t result_out, mpz_t cofactor_out,
    const mpz_t y, int attempt, int bench,
    bench_accum_t *loc_acc, bench_accum_t *gat_acc, bench_accum_t *com_acc,
    double *fac2_dt_out, double *recon_dt_out,
    benchmark_results_t *json_results,
    const ksc_recon_ctx_t *ctx);

// ---------------------------------------------------------------------------
// Strategy pattern — unifies the solve_one bench loop across k32, k35, alg3.
// k16 and alg4 are excluded: k16 uses a dense broadcast table and has no
// retry loop; alg4 has a fundamentally different outer pre-filter structure.
// ---------------------------------------------------------------------------

// Factor hook: called by every rank inside the retry loop.
// Appends found prime factors to factors_out.
// bench == -1 signals warmup (no JSON stats recorded); bench >= 0 is a
// timed run.
typedef void (*ksc_factor_fn_t)(
    kfactor_list_t *factors_out,
    const mpz_t blinded,
    int bench,
    benchmark_results_t *json_results,
    void *ctx,
    int rank, int nranks);

// Fold hook: called on rank 0 after ksc_compute_local_cbrt, inside the
// local-cbrt timing envelope.  Folds cbrt(cofactor) into local_cbrt.
// Set to NULL for k32/k35 (B-smooth only, no medium prime to fold).
typedef void (*ksc_fold_fn_t)(
    mpz_t local_cbrt,
    const mpz_t cofactor,
    const mpz_t e_inv,
    const mpz_t modulo,
    cbrt_cache_t *cache,
    bool *cache_dirty,
    benchmark_results_t *json_results,
    int bench,
    void *ctx);

typedef struct
{
    ksc_factor_fn_t factor;    // required
    ksc_fold_fn_t fold;        // NULL for k32/k35
    bool accept_almost_smooth; // true for alg3; false for k32/k35
    void *ctx;                 // algorithm-specific state
} ksc_factor_strategy_t;

// ---------------------------------------------------------------------------
// ksc_run_bench_loop — shared warmup (2 iterations) + benchmark loop.
//
// Handles: blinding + broadcast, factoring (via strategy), completeness check
// (via ksc_alg3_broadcast_cofactor), local cbrt, gather+combine, accumulation.
// On return, result_out holds the cube root found on the last successful
// bench iteration.
//
// Callers must initialise all accumulators to {0,0} before calling.
// json_results must be pre-initialised with ksc_init_json_results.
// ---------------------------------------------------------------------------
void ksc_run_bench_loop(
    mpz_t result_out,
    run_method_t method,
    const mpz_t challenge,
    const mpz_t modulo,
    size_t cbrt_bytes,
    const mpz_t e_inv,
    cbrt_cache_t *cache, bool *cache_dirty,
    blinding_pair_t *blind_table, int blind_table_size,
    uint64_t my_lo, uint64_t my_hi,
    bench_accum_t *fac_acc, bench_accum_t *loc_acc,
    bench_accum_t *gat_acc, bench_accum_t *com_acc, bench_accum_t *recon_acc,
    benchmark_results_t *json_results,
    const ksc_factor_strategy_t *strategy,
    int rank, int nranks);
