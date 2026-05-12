#pragma once

// =============================================================================
// k_solver_common.h — Shared MPI+OpenMP infrastructure for cube root solvers
//
// Centralises logic duplicated between k16_mpi.cpp and k32_mpi.cpp:
//   - Challenge/modulo broadcast
//   - JSON results initialisation
//   - MPI_Gather of partial cbrt products + combine/unblind
//   - Per-rank prime stats gather
//   - Timing report + powm baseline
//   - Precomputation guarantees (primes + cbrt tables/caches)
//
// Both solvers include this header and link k_solver_common.cpp.
// =============================================================================

#include "k_common.h"
#include "k_factorization.h"
#include <mpi.h>

// ---------------------------------------------------------------------------
// Data directory — all precomputed files live here
// ---------------------------------------------------------------------------
#define KSC_DATA_DIR "data"
#define KSC_CBRT_DIR "data/cbrt" // cbrt tables live here, separate from primes/challenges

// ---------------------------------------------------------------------------
// Challenge / modulo broadcast
// Rank 0 reads files; all ranks receive via Bcast.
// Returns true on success, calls MPI_Abort + returns false on failure.
// ---------------------------------------------------------------------------
bool ksc_broadcast_challenge_modulo(
    mpz_t challenge, mpz_t modulo,
    const char *challenge_file, const char *modulo_file,
    int rank);

// ---------------------------------------------------------------------------
// JSON results — initialise all fields common to both solvers
// ---------------------------------------------------------------------------
void ksc_init_json_results(
    benchmark_results_t *r,
    int k_size, int bench_count,
    const char *method_name,
    int mpi_ranks, int threads_per_rank,
    const mpz_t challenge, int rank);

// ---------------------------------------------------------------------------
// Partial cbrt gather + combine
//
// Every rank exports local_cbrt to a fixed-size big-endian buffer.
// Rank 0 gathers all nranks buffers, multiplies them into result, then
// unblinds: if attempt > 0, multiplies by blind_table[attempt-1].r_inv.
//
// t_gat_ms_out, t_com_ms_out: elapsed times on rank 0 (others: ignored).
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
// Phase 1 (sequential): resolve cbrt(p) per factor via cache or mpz_powm.
// Phase 2 (parallel):   exponentiate each cbrt by its exponent (OMP).
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
    int bench_count,
    const bench_accum_t *fac_acc,
    const bench_accum_t *loc_acc,
    const bench_accum_t *gat_acc,
    const bench_accum_t *com_acc,
    const bench_accum_t *recon_acc);

// ---------------------------------------------------------------------------
// Powm baseline + speedup (rank 0 only)
// Runs bench_count sequential powm calls, stores in r->powm_baseline_ms,
// prints speedup vs reconstruction-only and end-to-end.
// ---------------------------------------------------------------------------
void ksc_run_and_report_powm_baseline(
    benchmark_results_t *r,
    const mpz_t challenge,
    const mpz_t modulo,
    int bench_count,
    const bench_accum_t *recon_acc,
    const bench_accum_t *fac_acc,
    int rank);

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
    int rank);

// ---------------------------------------------------------------------------
// k32 precomputation
//
// ksc_ensure_k32_primes:
//   Ensures data/k32_primes.bin exists (rank 0 generates if absent).
//   All ranks barrier after. Returns path string (static storage).
//
// ksc_ensure_k32_cbrt:
//   Performs one UNTIMED factoring pass to populate the per-rank cbrt cache,
//   then saves it. If the cache file already exists it is loaded directly.
//   Guarantees that every prime found in the challenge has a precomputed
//   cbrt entry before the first timed run.
// ---------------------------------------------------------------------------
const char *ksc_ensure_k32_primes(int sp_count, const unsigned *sp, int rank);

void ksc_ensure_k32_cbrt(
    cbrt_cache_t *cache,
    bool *cache_dirty,
    const char *cache_path,
    const mpz_t challenge,
    const mpz_t modulo,
    const mpz_t e_inv,
    uint64_t my_lo, uint64_t my_hi,
    int sp_count, const unsigned *sp,
    const uint32_t *prime_slice, int64_t slice_count,
    int rank, int nranks,
    int method_uses_table);

// ---------------------------------------------------------------------------
// k64 precomputation
//
// ksc_ensure_k64_cbrt:
//   Rank 0 calls kf_factor_challenge (trial division + OMP Pollard's rho) on
//   the challenge, computes cbrt for every factor, and saves a shared cache
//   to data/cbrt/modulo_64_<bits>.cbrt (no rank suffix — all ranks use the
//   same file since the factor set is determined by the challenge, not the rank).
//
//   After return, the cache is loaded on ALL ranks via MPI_Bcast of the
//   serialised entries, so every rank can look up cbrt for any factor without
//   an mpz_powm call during the bench loop.
//
// factor_buf_out / factor_count_out:
//   Caller-allocated arrays (size K64_MAX_FACTORS) that receive the complete
//   factor list after the precompute pass so the bench loop can reuse it for
//   partition-based cbrt assignment without re-factoring.
// ---------------------------------------------------------------------------

#define K64_MAX_FACTORS 256 // generous upper bound on distinct prime factors

void ksc_ensure_k64_cbrt(
    cbrt_cache_t *cache,
    const mpz_t challenge,
    const mpz_t modulo,
    const char *modulo_file,
    int rank);

// ---------------------------------------------------------------------------
// k35 precomputation (same structure as k32 but range [65537, 2^35])
// ---------------------------------------------------------------------------
void ksc_ensure_k35_cbrt(
    cbrt_cache_t *cache,
    bool *cache_dirty,
    const char *cache_path,
    const mpz_t challenge,
    const mpz_t modulo,
    const mpz_t e_inv,
    uint64_t my_lo, uint64_t my_hi,
    int sp_count, const unsigned *sp,
    int rank, int nranks);

// ---------------------------------------------------------------------------
// k35 upper-range prime table management
// ---------------------------------------------------------------------------

// Ensures data/k35_upper_primes.bin exists (rank 0 generates if missing).
// All ranks synchronise via MPI_Barrier before returning.
// sp/sp_count must cover primes up to sqrt(2^35) ~ 185363 (use 2^18 limit).
const char *ksc_ensure_k35_primes(int sp_count, const unsigned *sp, int rank);

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
    int rank, int nranks);