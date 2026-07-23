#pragma once
// =============================================================================
// prefilter.h — Simple pre-filtering (paper Fig. 6 / Algorithm 5)
//
// Advisor todo 2026-06-29; |Re| sizing method fixed 2026-07-10 (benchmark the
// pre-filter alone, size by the B - S budget). Design plan:
//   wiki/planning/simple-prefilter-plan-2026-07-07.md
//
// Key realization: the existing blinding table IS the paper's Re/R1 table.
//   blinding_pair_t.r_blind = r^e  mod M   ← Re[i]
//   blinding_pair_t.r_inv   = r^-1 mod M   ← R1[i]
// So the pre-filter adds NO new table. It only changes how a candidate is
// *selected*: instead of the sequential blind-retry (attempt 0 = raw challenge,
// attempt k = challenge * r_blind[k-1], one at a time), it forms ALL |Re|
// candidates y_i = challenge * r_blind[i] mod M at once and picks the SMALLEST
// (Dickman rho: smaller value => higher B-smoothness probability), then solves
// that single candidate and unblinds with r_inv[j].
//
// This header exposes just the selection primitive (Phase 0). Wiring it into the
// k32 / alg3 bench loop via a pre_select hook is Phase 1 (separate change).
// =============================================================================

#include <gmp.h>
#include "factorization.h" // blinding_pair_t

// ---------------------------------------------------------------------------
// prefilter_argmin_local — serial, no MPI. Unit-testable core.
//
// Scans table indices [lo, hi) and returns, in y_best/idx_best, the candidate
// y_i = challenge * table[i].r_blind mod M with the SMALLEST value.
//
//   y_best  : pre-initialised mpz_t; set to the winning y_i on return. If the
//             range is empty (lo >= hi) it is set to `modulo` (a strict upper
//             bound on every real y_i, used as a +inf sentinel) and *idx_best
//             is set to -1.
//   idx_best: winning table index (in [lo, hi)), or -1 for an empty range.
//
// Comparison is exact (full mpz), so this defines the ground truth the parallel
// version must match.
// ---------------------------------------------------------------------------
void prefilter_argmin_local(mpz_t y_best, int *idx_best,
                            const mpz_t challenge,
                            const blinding_pair_t *table,
                            int lo, int hi,
                            const mpz_t modulo);

// ---------------------------------------------------------------------------
// prefilter_simple_select — MPI global argmin over the whole blinding table.
//
// Each rank owns table indices [lo, hi) (caller partitions [0, table_size)).
// Every rank computes its local argmin (prefilter_argmin_local), then rank 0
// gathers the per-rank best values + indices, picks the global minimum EXACTLY
// (fixed-size big-endian buffers, one mpz per rank — cheap for nranks <= 128
// and provably matches a single-rank brute force), and broadcasts the winner.
//
// On return (identical on ALL ranks):
//   y_out : mpz_t (pre-initialised) = the winning candidate y_j.
//   *j_out: winning table index j in [0, table_size), or -1 if no rank owned a
//           valid entry (only possible if table_size == 0).
//
// Unblinding is the caller's job: root(y_j) * table[j].r_inv mod M recovers the
// root of the ORIGINAL challenge.
//
// Deliberately exact (gather-to-rank-0) rather than the MINLOC sort-key sketched
// in the plan: for a first implementation correctness beats the micro-optimised
// collective, and the key-vs-exact tradeoff was flagged as the main risk.
// ---------------------------------------------------------------------------
int prefilter_simple_select(mpz_t y_out, int *j_out,
                            const mpz_t challenge,
                            const blinding_pair_t *table, int table_size,
                            int lo, int hi,
                            const mpz_t modulo,
                            int rank, int nranks);

// Edit: 2026-07-12
