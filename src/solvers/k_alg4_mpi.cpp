// =============================================================================
// k_alg4_mpi.cpp — Algorithm 4 solver: 2^64-smooth cube root recovery
//
// Extends Algorithm 3 (k_alg3_mpi) by tolerating composite cofactors ≤ 2^64
// in the blinded challenge, per Biryukov et al. CRYPTO 2024.
//
// Key differences from alg3_mpi:
//   OLD (alg3): cofactor must be prime ≤ 2^65 (one medium prime q0)
//   NEW (alg4): cofactor ≤ 2^64, either prime or composite.
//               If composite, Pollard's rho factors it (rank 0 only).
//               All medium prime cbrts are folded into local_cbrt on rank 0.
//
//   PRE-FILTER (the core alg4 contribution per §3.2 of the paper):
//     Before each full MPI sieve, rank 0 screens ALG4_PREFILTER_K blinding
//     candidates using only primes ≤ B0=2^16 (reusing the sp[] array).
//     The candidate whose blinded value has the largest B0-smooth part
//     (= most bits removed by small-prime trial division) is selected and
//     broadcast for the expensive full B=2^32 sieve.
//
//     Effect: each outer sieve attempt operates on a pre-screened y that is
//     more likely to be (2^32, 2^65)-almost-smooth, reducing the expected
//     number of full sieve calls and the failure rate at large challenge sizes.
//
// Pre-filter parameters (§3.2, scaled to our setup):
//   B0           = 2^16  (sp[] already covers primes ≤ 65536)
//   PREFILTER_K  = 32    (candidates screened per outer attempt at rank 0)
//   MAX_OUTER    = 8     (outer sieve attempts; total table = 32×8 = 256)
//
//   blinding_attempts in JSON records total candidates evaluated:
//     attempt=0 (unblinded)  → blinding_attempts = 1
//     outer=1 succeeds       → blinding_attempts = 1 + 32 = 33
//     outer=k succeeds       → blinding_attempts = 1 + k*32
//     all fail               → blinding_attempts = 257  (same as alg3)
//
// Isolation from Algorithm 3:
//   Challenges:  data/challenges/alg4/
//   cbrt caches: data/cbrt/<stem>.cbrt_alg4.<rank>
//   Results:     results/algo4/alg4_mpi_N*/
//   Prime table: reuses data/k32_primes.bin read-only (sieve method only)
//
// Communication pattern: identical to alg3_mpi.
//
// Usage:
//   alg4_mpi [challenge] [modulo] [--method sieve|powm] [--bench N]
//   alg4_mpi --all [--data DIR] [--bench N]
// =============================================================================

#include "k_common.h"
#include "k_factorization.h"
#include "k_solver_common.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

typedef enum
{
    METHOD_SIEVE,
    METHOD_POWM,
} run_method_t;

// Pre-filter parameters (§3.2 Biryukov et al.)
// B0 = 2^16 — reuses sp[] (primes ≤ 65536) generated at startup; no new
// prime generation needed.  The paper uses B0=2^20 with π(2^20)≈72k primes;
// we use B0=2^16 with π(2^16)≈6542 primes so the pre-filter runs in <10ms
// even for the largest (24576-bit) challenges, keeping other ranks unblocked.
#define ALG4_PREFILTER_K 32                                  // candidates screened per outer sieve attempt
#define ALG4_MAX_OUTER 8                                     // outer attempts (BLIND_TABLE_SIZE / PREFILTER_K)
#define BLIND_TABLE_SIZE (ALG4_PREFILTER_K * ALG4_MAX_OUTER) // 256, same as before

#define print_stat(lbl, sum, sumsq, n) \
    print_timing_stat_moments(lbl, sum, sumsq, n, 30, true)

// ---------------------------------------------------------------------------
// b0_smooth_score
//
// Returns the number of bits removed from y when all prime factors ≤ B0 are
// divided out.  Higher score = more smooth = better pre-filter candidate.
//
// Uses sp[] (primes ≤ 65536) as the B0 factor base.  Operates on a copy of
// y; the original is not modified.  Early exit once remainder == 1.
// ---------------------------------------------------------------------------

static int b0_smooth_score(const mpz_t y, const unsigned *sp, int sp_count)
{
    int orig_bits = (int)mpz_sizeinbase(y, 2);

    mpz_t rem;
    mpz_init_set(rem, y);

    for (int i = 0; i < sp_count && mpz_cmp_ui(rem, 1) > 0; i++)
    {
        unsigned long p = (unsigned long)sp[i];
        // Fast check: mpz_tdiv_ui does not modify rem.
        while (mpz_divisible_ui_p(rem, p))
            mpz_divexact_ui(rem, rem, p);
    }

    // If remainder == 1 the value is fully B0-smooth: all bits were removed.
    int rem_bits = (mpz_cmp_ui(rem, 1) == 0) ? 0 : (int)mpz_sizeinbase(rem, 2);
    mpz_clear(rem);
    return orig_bits - rem_bits;
}

// ---------------------------------------------------------------------------
// prefilter_candidates  (rank 0 only)
//
// Tries ALG4_PREFILTER_K blinding candidates from the table starting at
// table_start, computing y = challenge * r^3 mod M for each and scoring
// it with b0_smooth_score().
//
// On return:
//   best_y       — set to the highest-scoring blinded challenge value.
//   return value — absolute table index of the winner.
//
// The winning index is needed later to retrieve r_inv for unblinding.
// ---------------------------------------------------------------------------

static int prefilter_candidates(
    mpz_t best_y,
    const mpz_t challenge,
    const mpz_t modulo,
    const blinding_pair_t *table,
    int table_start,
    const unsigned *sp, int sp_count)
{
    int best_score = -1;
    int best_local = 0;

    mpz_t y_try;
    mpz_init(y_try);

    for (int k = 0; k < ALG4_PREFILTER_K; k++)
    {
        int idx = table_start + k;

        // y = challenge * r^3 mod M  (r^3 is precomputed in the table)
        mpz_mul(y_try, challenge, table[idx].r3);
        mpz_mod(y_try, y_try, modulo);

        int score = b0_smooth_score(y_try, sp, sp_count);

        if (score > best_score)
        {
            best_score = score;
            best_local = k;
            mpz_set(best_y, y_try);
        }
    }

    mpz_clear(y_try);
    return table_start + best_local; // absolute table index
}

// ---------------------------------------------------------------------------
// solve_one
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count, int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method,
    const char *json_output_file = nullptr)
{
    if (rank == 0)
    {
        printf("\n=== %s ===\n", challenge_file);
        fflush(stdout);
    }

    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    if (!ksc_broadcast_challenge_modulo(challenge, modulo,
                                        challenge_file, modulo_file, rank))
    {
        mpz_clears(challenge, modulo, result, NULL);
        return 0;
    }

    if (rank == 0)
    {
        printf("Challenge: %zu bits\n", mpz_sizeinbase(challenge, 2));
        printf("Modulo:    %zu bits\n", mpz_sizeinbase(modulo, 2));
        printf("Ranks: %d  Threads/rank: %d  Total workers: %d\n",
               nranks,
#ifdef _OPENMP
               omp_get_max_threads(), nranks * omp_get_max_threads());
#else
               1, nranks);
#endif
        printf("Pre-filter: B0=2^16, K=%d candidates/outer, max %d outer attempts\n",
               ALG4_PREFILTER_K, ALG4_MAX_OUTER);
        fflush(stdout);
    }

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    // Compute e_inv = 3^{-1} mod (M-1) — the cube root exponent.
    mpz_t phi, e_inv, three_mpz;
    mpz_inits(phi, e_inv, three_mpz, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three_mpz, 3);
    mpz_invert(e_inv, three_mpz, phi);
    mpz_clears(phi, three_mpz, NULL);

    // ALG4: cache suffix .cbrt_alg4.<rank> — isolated from alg3 caches.
    char cbrt_cache_path[600];
    {
        const char *bname = modulo_file ? modulo_file : "modulo";
        const char *slash = strrchr(bname, '/');
        const char *fname = slash ? slash + 1 : bname;
        char stem[512];
        strncpy(stem, fname, sizeof(stem) - 1);
        stem[sizeof(stem) - 1] = '\0';
        size_t blen = strlen(stem);
        if (blen >= 4 && strcmp(stem + blen - 4, ".txt") == 0)
            stem[blen - 4] = '\0';
        snprintf(cbrt_cache_path, sizeof(cbrt_cache_path),
                 "%s/%s.cbrt_alg4.%d", KSC_CBRT_DIR, stem, rank);
    }

    cbrt_cache_t local_cbrt_cache;
    kf_cbrt_cache_init(&local_cbrt_cache);
    bool cbrt_cache_dirty = false;

    ksc_ensure_k32_cbrt(
        &local_cbrt_cache, &cbrt_cache_dirty, cbrt_cache_path,
        challenge, modulo, e_inv,
        my_lo, my_hi, sp_count, sp,
        NULL, 0,
        rank, nranks, 0 /* method_uses_table = sieve */);

    // Precompute cbrt(q0) for the medium prime(s) before the bench loop.
    //
    // Without this, bench run 1 pays a full powm (~1800ms for large M) per
    // medium prime on rank 0.  With this block all warm bench runs are instant
    // cache hits.
    //
    // We only precompute when the precompute itself is FAST:
    //   (a) cofactor is prime  → one direct powm, always do it.
    //   (b) cofactor ≤ 2^64   → both factors ≤ 2^32, kf_factor_recursive is
    //                            instant (trial division, no Pollard needed).
    //
    // We deliberately SKIP composites > 2^64 (alg4_multi p1*p2 with 63-bit
    // factors).  kf_pollard_rho_brent has no inner iteration limit so it
    // can run for √(2^63) ≈ 3 billion steps per c-value, hanging all other
    // MPI ranks at MPI_Barrier for minutes.  For those challenges the first
    // bench run will hit a cache miss (~1800ms × 2), cache the result, and
    // all 49 subsequent runs will be instant hits — an acceptable trade-off.
    if (rank == 0)
    {
        // Sieve the FULL [65537, 2^32] range on rank 0 to get the smooth part.
        kfactor_list_t pre_factors;
        kfl_init(&pre_factors);
        kf_trial_divide_sieve(challenge, 65537ULL, (uint64_t)UINT32_MAX,
                              sp_count, sp, &pre_factors);

        mpz_t combined;
        mpz_init_set_ui(combined, 1);
        for (int i = 0; i < pre_factors.count; i++)
        {
            unsigned long p = mpz_get_ui(pre_factors.items[i].prime);
            for (int ex = 0; ex < pre_factors.items[i].exponent; ex++)
                mpz_mul_ui(combined, combined, p);
        }
        kfl_free(&pre_factors);

        mpz_t cofactor, rem;
        mpz_inits(cofactor, rem, NULL);
        mpz_tdiv_qr(cofactor, rem, challenge, combined);
        mpz_clear(combined);

        if (mpz_sgn(rem) == 0 && mpz_cmp_ui(cofactor, 1) > 0)
        {
            bool precomputed = false;

            if (mpz_probab_prime_p(cofactor, 15))
            {
                // (a) Single medium prime — precompute directly.
                unsigned long q_ul = mpz_fits_ulong_p(cofactor)
                                         ? mpz_get_ui(cofactor)
                                         : 0UL;
                if (q_ul > 0UL &&
                    kf_cbrt_cache_find(&local_cbrt_cache, q_ul) < 0)
                {
                    mpz_t cbrt_q;
                    mpz_init(cbrt_q);
                    mpz_powm(cbrt_q, cofactor, e_inv, modulo);
                    kf_cbrt_cache_append(&local_cbrt_cache, q_ul, cbrt_q);
                    cbrt_cache_dirty = true;
                    mpz_clear(cbrt_q);
                    printf("[precompute] cached cbrt(q0) (%zu bits)\n",
                           mpz_sizeinbase(cofactor, 2));
                    fflush(stdout);
                    precomputed = true;
                }
            }
            else if (mpz_sizeinbase(cofactor, 2) <= 64)
            {
                // (b) Composite ≤ 2^64: both factors ≤ 2^32, fast to factor.
                kfactor_list_t poly;
                kfl_init(&poly);
                if (kf_try_factor_b64smooth(cofactor, &poly, 1))
                {
                    for (int i = 0; i < poly.count; i++)
                    {
                        unsigned long q_ul =
                            mpz_fits_ulong_p(poly.items[i].prime)
                                ? mpz_get_ui(poly.items[i].prime)
                                : 0UL;
                        if (q_ul > 0UL &&
                            kf_cbrt_cache_find(&local_cbrt_cache, q_ul) < 0)
                        {
                            mpz_t cbrt_q;
                            mpz_init(cbrt_q);
                            mpz_powm(cbrt_q, poly.items[i].prime,
                                     e_inv, modulo);
                            kf_cbrt_cache_append(&local_cbrt_cache,
                                                 q_ul, cbrt_q);
                            cbrt_cache_dirty = true;
                            mpz_clear(cbrt_q);
                        }
                    }
                    precomputed = true;
                    printf("[precompute] cached cbrt for %d medium prime(s) "
                           "(composite %zu bits)\n",
                           poly.count, mpz_sizeinbase(cofactor, 2));
                    fflush(stdout);
                }
                kfl_free(&poly);
            }
            else
            {
                // Composite > 2^64 (alg4_multi with large p1*p2).
                // Skipping precompute — Pollard's rho would be too slow here.
                // First bench run will cache-miss and compute cbrt on-the-fly.
                printf("[precompute] cofactor %zu bits is composite > 2^64 "
                       "— skipping precompute, first bench run will cache.\n",
                       mpz_sizeinbase(cofactor, 2));
                fflush(stdout);
                (void)precomputed;
            }
        }
        mpz_clears(cofactor, rem, NULL);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Blinding table: BLIND_TABLE_SIZE = PREFILTER_K × MAX_OUTER entries.
    // Rank 0 uses these for both pre-filtering (K per outer attempt) and
    // unblinding (r_inv of the winning entry).
    // Non-zero ranks never touch the table — they just receive the chosen y.
    blinding_pair_t *blind_table = NULL;
    if (rank == 0)
    {
        blind_table = kf_blinding_table_alloc(BLIND_TABLE_SIZE, modulo);
        if (!blind_table)
        {
            fprintf(stderr, "[rank 0] Failed to allocate blinding table\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    mpz_t blinded_challenge;
    mpz_init(blinded_challenge);

    kfactor_list_t medium_factors;
    kfl_init(&medium_factors);

    bench_accum_t fac_acc = {0, 0};
    bench_accum_t loc_acc = {0, 0};
    bench_accum_t gat_acc = {0, 0};
    bench_accum_t com_acc = {0, 0};
    bench_accum_t recon_acc = {0, 0};
    double cur_max_loc = 0.0;

    int n_threads_json = 1;
#ifdef _OPENMP
    n_threads_json = omp_get_max_threads();
#endif
    benchmark_results_t json_results;
    const char *mname = (method == METHOD_POWM) ? "powm" : "sieve";
    // BUG FIX: k_size = 32, not 64.  Alg4 sieves [65537, 2^32] — same range
    // as alg3.  The k_size field in JSON records the sieve upper bound in bits.
    ksc_init_json_results(&json_results, 32, bench_count, mname,
                          nranks, n_threads_json, challenge, rank);

    // ---- 2 warmup runs ----
    // Warmup exercises the sieve + local cbrt path to prime CPU caches and
    // fill the cbrt cache with any factors present in the unblinded challenge.
    if (method != METHOD_POWM)
    {
        for (int wu = 0; wu < 2; wu++)
        {
            kfactor_list_t wu_factors;
            kfl_init(&wu_factors);

            size_t wu_nb = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
            unsigned char *wu_buf = (unsigned char *)calloc(wu_nb, 1);

            if (rank == 0)
            {
                size_t wr = 0;
                unsigned char *tmp = (unsigned char *)
                    mpz_export(NULL, &wr, 1, 1, 0, 0, challenge);
                if (tmp && wr > 0 && wr <= wu_nb)
                {
                    memcpy(wu_buf + wu_nb - wr, tmp, wr);
                    free(tmp);
                }
            }
            MPI_Bcast(wu_buf, (int)wu_nb, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

            mpz_t wu_ch;
            mpz_init(wu_ch);
            mpz_import(wu_ch, wu_nb, 1, 1, 0, 0, wu_buf);
            free(wu_buf);

            kf_trial_divide_sieve(wu_ch, my_lo, my_hi, sp_count, sp, &wu_factors);

            mpz_t wu_cbrt;
            mpz_init(wu_cbrt);
            bool wu_dirty = false;

            auto wu_t0 = std::chrono::high_resolution_clock::now();
            ksc_compute_local_cbrt(wu_cbrt, &wu_factors, e_inv, modulo,
                                   &local_cbrt_cache, &wu_dirty);
            auto wu_t1 = std::chrono::high_resolution_clock::now();

            if (wu_dirty)
                cbrt_cache_dirty = true;
            if (rank == 0)
                json_results.warmup_total_ms.push_back(
                    std::chrono::duration<double, std::milli>(wu_t1 - wu_t0).count());

            mpz_clears(wu_ch, wu_cbrt, NULL);
            kfl_free(&wu_factors);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // ---- Benchmark loop ----

    for (int bench = 0; bench < bench_count && method != METHOD_POWM; bench++)
    {
        kfactor_list_t local_factors;
        kfl_init(&local_factors);
        double iter_gat = 0.0, iter_com = 0.0;

        kfl_free(&medium_factors);
        kfl_init(&medium_factors);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        // winning_table_idx: which blinding table entry was chosen.
        //   -1  → unblinded (outer attempt 0, y = challenge)
        //   ≥ 0 → index into blind_table[], used for unblinding via r_inv.
        // Broadcast to all ranks so non-zero ranks know the attempt count
        // for JSON (they don't need r_inv themselves).
        int winning_table_idx = -1;
        bool bench_succeeded = false;
        int outer_attempts_used = 0;

        size_t blinded_nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
        unsigned char *blinded_buf = (unsigned char *)calloc(blinded_nbytes, 1);

        // ---- Outer attempt loop ----
        //
        // Attempt 0: unblinded (y = challenge directly, no pre-filter).
        //            If the challenge itself is (B, B')-almost-smooth this
        //            succeeds on the first try — the common case.
        //
        // Attempts 1..MAX_OUTER: pre-filter ALG4_PREFILTER_K candidates at
        //            rank 0, broadcast the best one, then do a full sieve.

        for (int outer = 0; outer <= ALG4_MAX_OUTER && !bench_succeeded; outer++)
        {
            outer_attempts_used = outer;

            // -- Rank 0: select blinded challenge for this outer attempt --
            if (rank == 0)
            {
                if (outer == 0)
                {
                    // No blinding: use challenge as-is.
                    mpz_set(blinded_challenge, challenge);
                    winning_table_idx = -1;
                }
                else
                {
                    // Pre-filter: screen PREFILTER_K candidates from the
                    // blinding table and pick the one with the largest
                    // B0=2^16-smooth part.  This selects the candidate most
                    // likely to pass the full B=2^32 smoothness test.
                    int table_start = (outer - 1) * ALG4_PREFILTER_K;
                    winning_table_idx = prefilter_candidates(
                        blinded_challenge, challenge, modulo,
                        blind_table, table_start,
                        sp, sp_count);

                    if (bench == 0)
                    {
                        // Log pre-filter outcome on the diagnostic run only.
                        int score = b0_smooth_score(blinded_challenge, sp, sp_count);
                        printf("Pre-filter outer %d: best B0-score=%d bits "
                               "(table[%d])\n",
                               outer, score, winning_table_idx);
                        fflush(stdout);
                    }
                }

                // Serialise blinded_challenge to fixed-length buffer.
                memset(blinded_buf, 0, blinded_nbytes);
                size_t written = 0;
                unsigned char *tmp = (unsigned char *)mpz_export(
                    NULL, &written, 1, 1, 0, 0, blinded_challenge);
                if (tmp && written > 0 && written <= blinded_nbytes)
                {
                    memcpy(blinded_buf + blinded_nbytes - written, tmp, written);
                    free(tmp);
                }
            }

            // Broadcast the chosen blinded challenge to all ranks.
            MPI_Bcast(blinded_buf, (int)blinded_nbytes, MPI_UNSIGNED_CHAR,
                      0, MPI_COMM_WORLD);
            // Also broadcast winning_table_idx so all ranks can compute the
            // correct blinding_attempts count for JSON.
            MPI_Bcast(&winning_table_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank != 0)
                mpz_import(blinded_challenge, blinded_nbytes, 1, 1, 0, 0,
                           blinded_buf);

            // -- All ranks: full sieve on the chosen blinded challenge --
            kfl_free(&local_factors);
            kfl_init(&local_factors);
            kf_trial_divide_sieve(blinded_challenge, my_lo, my_hi,
                                  sp_count, sp, &local_factors);

            // -- Rank 0: catch small primes (≤ 65536) missed by the sieve --
            //
            // The sieve covers [65537, 2^32].  Challenges generated with small
            // random seeds can include prime factors ≤ 65536 (from the smooth
            // base) that are below the sieve start.  This is especially likely
            // for 256-bit alg4_multi challenges where only ~4 primes are needed.
            //
            // Fix: rank 0 additionally trial-divides by all primes in sp[]
            // (the Eratosthenes base, covering [2, 65536]).  This is O(6542)
            // mpz_tdiv_ui calls — negligible cost even for large challenges.
            // Other ranks don't need this: by MPI partitioning, rank 0 is
            // responsible for its sieve slice AND the sub-65537 range.
            if (rank == 0)
            {
                for (int i = 0; i < sp_count; i++)
                {
                    unsigned long p = (unsigned long)sp[i];
                    if (mpz_divisible_ui_p(blinded_challenge, p))
                    {
                        // count_exponent tallies the exact power without
                        // modifying blinded_challenge.
                        int e = count_exponent(blinded_challenge, p);
                        if (e > 0)
                            kfl_append_ul(&local_factors, p, e);
                    }
                }
            }

            // -- ALG4 cofactor check --
            // Each rank computes the product of its found prime powers,
            // then ksc_alg4_broadcast_cofactor gathers all products to rank 0,
            // computes the cofactor, and (if ≤ 2^64) factors it with Pollard's
            // rho.  Broadcasts complete=1 and the medium prime(s) to all ranks.
            mpz_t local_prod;
            mpz_init_set_ui(local_prod, 1);
            for (int i = 0; i < local_factors.count; i++)
            {
                unsigned long p = mpz_get_ui(local_factors.items[i].prime);
                for (int ex = 0; ex < local_factors.items[i].exponent; ex++)
                    mpz_mul_ui(local_prod, local_prod, p);
            }

            kfl_free(&medium_factors);
            kfl_init(&medium_factors);

            int complete = ksc_alg4_broadcast_cofactor(
                &medium_factors, blinded_challenge, local_prod, cbrt_bytes,
                rank, nranks);
            mpz_clear(local_prod);

            if (rank == 0 && bench == 0)
            {
                int nmed = medium_factors.count;
                const char *label = complete
                                        ? (nmed == 0   ? "B-smooth — proceeding"
                                           : nmed == 1 ? "almost-smooth — proceeding"
                                                       : "multi-medium — proceeding")
                                        : "not 2^64-smooth — trying next group";
                printf("Outer attempt %d: %s\n", outer, label);
                fflush(stdout);
            }

            if (complete)
                bench_succeeded = true;

        } // end outer attempt loop

        free(blinded_buf);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        // Record factoring time.  Includes pre-filter time at rank 0 because
        // the pre-filter is part of the factoring phase — other ranks are idle
        // during it, but that idle time IS part of the wall-clock cost.
        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(
                            t_fac1 - t_fac0)
                            .count();
            accum_record(&fac_acc, dt);

            if (bench > 0 && bench_succeeded)
                json_results.factor_times.push_back(dt);

            if (bench == 0)
            {
                printf("Factors found by rank 0: %d distinct primes "
                       "(after %d outer attempt(s), ~%d total candidates)\n",
                       local_factors.count,
                       outer_attempts_used + 1,
                       1 + outer_attempts_used * ALG4_PREFILTER_K);
                fflush(stdout);
            }

            if (bench > 0 && bench_succeeded)
            {
                // blinding_attempts: total candidates evaluated, matching the
                // alg3 convention (1 = first try worked, 257 = all exhausted).
                //   outer=0 (unblinded) → 1
                //   outer=k → 1 + k * PREFILTER_K
                int total_candidates = 1 + outer_attempts_used * ALG4_PREFILTER_K;
                json_results.blinding_attempts.push_back(total_candidates);

                // Hit classification.
                int nmed = medium_factors.count;
                int htype = (nmed == 0) ? 1 : (nmed == 1) ? 2
                                                          : 3;
                json_results.hit_type.push_back(htype);

                if (nmed == 0)
                {
                    json_results.cofactor_bits.push_back(1);
                }
                else
                {
                    mpz_t prod;
                    mpz_init_set_ui(prod, 1);
                    for (int i = 0; i < nmed; i++)
                        mpz_mul(prod, prod, medium_factors.items[i].prime);
                    json_results.cofactor_bits.push_back(
                        (int)mpz_sizeinbase(prod, 2));
                    mpz_clear(prod);
                }
                json_results.medium_factor_count.push_back(nmed);
            }
        }

        // ---- DISTRIBUTED RECONSTRUCTION ----

        auto t_loc0 = std::chrono::high_resolution_clock::now();

        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double cbrt_hit_ms = 0.0, cbrt_miss_ms = 0.0;
        int cbrt_hits = 0, cbrt_misses = 0;
        bool record_cbrt = (bench > 0);
        ksc_compute_local_cbrt(local_cbrt, &local_factors, e_inv, modulo,
                               &local_cbrt_cache, &cbrt_cache_dirty,
                               &cbrt_hit_ms, &cbrt_miss_ms,
                               &cbrt_hits, &cbrt_misses, record_cbrt);

        // ALG4: rank 0 folds cbrt for each medium prime into its local partial
        // product.  Medium primes are > 2^32 so they are not covered by any
        // rank's sieve slice — only rank 0 handles them.
        //
        // IMPORTANT: cbrt(q0) = mpz_powm(q0, e_inv, M) costs ~1 full powm
        // (~1800ms for 24576-bit M) because the cost is dominated by |e_inv|,
        // not |q0|.  We therefore check the cbrt_cache first.  For a fixed
        // challenge, q0 is always the same (outer=0 unblinded path), so the
        // cache is populated on the first warm run and all subsequent runs are
        // instant cache hits.
        //
        // q0 fits in unsigned long on all 64-bit Linux targets (q0 ≤ 2^63
        // for alg4/alg4_multi), so the existing cache infrastructure works
        // directly without any schema change.
        if (rank == 0 && medium_factors.count > 0)
        {
            for (int i = 0; i < medium_factors.count; i++)
            {
                mpz_t cbrt_qi;
                mpz_init(cbrt_qi);

                // Cache lookup: q0 must fit in unsigned long (≤ 63 bits).
                unsigned long q_ul =
                    mpz_fits_ulong_p(medium_factors.items[i].prime)
                        ? mpz_get_ui(medium_factors.items[i].prime)
                        : 0UL;

                int cache_idx = (q_ul > 0UL)
                                    ? kf_cbrt_cache_find(&local_cbrt_cache, q_ul)
                                    : -1;

                if (cache_idx >= 0)
                {
                    // Cache hit: instant.
                    mpz_set(cbrt_qi, local_cbrt_cache.items[cache_idx].cbrt);
                }
                else
                {
                    // Cache miss: one full powm (~1800ms for large M).
                    // Happens only on the first encounter of this q0.
                    mpz_powm(cbrt_qi, medium_factors.items[i].prime,
                             e_inv, modulo);

                    if (q_ul > 0UL)
                    {
                        kf_cbrt_cache_append(&local_cbrt_cache, q_ul, cbrt_qi);
                        cbrt_cache_dirty = true;
                    }
                }

                mpz_mul(local_cbrt, local_cbrt, cbrt_qi);
                mpz_mod(local_cbrt, local_cbrt, modulo);
                mpz_clear(cbrt_qi);
            }
        }

        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(
                              t_loc1 - t_loc0)
                              .count();

        // Gather prime stats once (first warm run).
        if (bench == 1)
            ksc_gather_prime_stats(&json_results,
                                   local_factors.count,
                                   (int64_t)(my_hi - my_lo + 1),
                                   rank, nranks);

        if (rank == 0 && bench > 0 && bench_succeeded)
        {
            json_results.cbrt_hit_ms.push_back(cbrt_hit_ms);
            json_results.cbrt_miss_ms.push_back(cbrt_miss_ms);
            json_results.cbrt_hits.push_back(cbrt_hits);
            json_results.cbrt_misses.push_back(cbrt_misses);
        }

        // Use the maximum local_cbrt time across all ranks as the
        // reconstruction bottleneck (determines when gather can start).
        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            accum_record(&loc_acc, max_local_dt);
            cur_max_loc = max_local_dt;
            if (bench > 0 && bench_succeeded)
                json_results.local_cbrt_times.push_back(max_local_dt);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Gather partial cbrt products from all ranks, combine, and unblind.
        //
        // KEY: pass (winning_table_idx + 1) as the `attempt` parameter.
        // ksc_gather_and_combine_cbrt uses blind_table[attempt-1].r_inv, so:
        //   winning_table_idx == -1  →  attempt == 0  →  no unblinding  ✓
        //   winning_table_idx == k   →  attempt == k+1  →  blind_table[k].r_inv  ✓
        double iter_gat2 = 0.0, iter_com2 = 0.0;
        ksc_gather_and_combine_cbrt(result, local_cbrt, modulo, cbrt_bytes,
                                    rank, nranks,
                                    winning_table_idx + 1, // see note above
                                    blind_table, BLIND_TABLE_SIZE,
                                    &iter_gat2, &iter_com2);
        mpz_clear(local_cbrt);

        if (rank == 0)
        {
            accum_record(&gat_acc, iter_gat2);
            iter_gat = iter_gat2;
            if (bench > 0 && bench_succeeded)
                json_results.gather_times.push_back(iter_gat2);

            accum_record(&com_acc, iter_com2);
            iter_com = iter_com2;
            if (bench > 0 && bench_succeeded)
                json_results.combine_times.push_back(iter_com2);

            double recon_dt = cur_max_loc + iter_gat + iter_com;
            accum_record(&recon_acc, recon_dt);
            if (bench > 0 && bench_succeeded)
            {
                json_results.recon_times.push_back(recon_dt);
                if (!json_results.factor_times.empty())
                    json_results.total_times.push_back(
                        json_results.factor_times.back() + recon_dt);
            }
        }

        kfl_free(&local_factors);

    } // end bench loop

    // ---- Timing report ----
    if (rank == 0)
    {
        printf("\n");
        if (bench_count == 1)
        {
            printf("Factoring:         %.3f ms\n", fac_acc.sum);
            printf("Local cbrt (max):  %.3f ms\n", loc_acc.sum);
            printf("Gather:            %.3f ms\n", gat_acc.sum);
            printf("Combine:           %.3f ms\n", com_acc.sum);
            printf("Reconstruction:    %.3f ms\n",
                   loc_acc.sum + gat_acc.sum + com_acc.sum);
            printf("Total:             %.3f ms\n",
                   fac_acc.sum + loc_acc.sum + gat_acc.sum + com_acc.sum);
        }
        else
        {
            printf("Results over %d runs:\n", bench_count);
            print_stat("Factoring:", fac_acc.sum, fac_acc.sumsq, bench_count);
            print_stat("Local cbrt (max):", loc_acc.sum, loc_acc.sumsq, bench_count);
            print_stat("Gather:", gat_acc.sum, gat_acc.sumsq, bench_count);
            print_stat("Combine (rank 0):", com_acc.sum, com_acc.sumsq, bench_count);
            print_stat("Reconstruction total:", recon_acc.sum, recon_acc.sumsq, bench_count);
            printf("    %-30s avg=%9.3f ms\n", "Total:",
                   (fac_acc.sum + recon_acc.sum) / bench_count);
        }

        ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                         bench_count, &recon_acc, &fac_acc, rank);

        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_cube_root(result, challenge_mod, modulo);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    // ---- Cleanup ----
    mpz_clear(blinded_challenge);
    kfl_free(&medium_factors);
    if (rank == 0)
        kf_blinding_table_free(blind_table, BLIND_TABLE_SIZE);

    if (cbrt_cache_dirty)
        kf_cbrt_cache_save(&local_cbrt_cache, cbrt_cache_path);
    kf_cbrt_cache_clear(&local_cbrt_cache);

    mpz_clear(e_inv);
    mpz_clears(challenge, modulo, result, NULL);

    if (rank == 0 && json_output_file && json_output_file[0] != '\0')
        write_benchmark_json(json_output_file, json_results);

    return 1;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
    {
        fprintf(stderr, "Error: MPI_THREAD_FUNNELED not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char *challenge_file = NULL;
    const char *modulo_file = NULL;
    const char *data_dir = "data/challenges/alg4";
    const char *json_file = NULL;
    int bench_count = 1;
    int run_all = 0;
    run_method_t method = METHOD_SIEVE;

    int positional = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--bench") == 0 && argi + 1 < argc)
        {
            bench_count = atoi(argv[++argi]);
        }
        else if (strcmp(argv[argi], "--all") == 0)
        {
            run_all = 1;
        }
        else if (strcmp(argv[argi], "--data") == 0 && argi + 1 < argc)
        {
            data_dir = argv[++argi];
        }
        else if (strcmp(argv[argi], "--method") == 0 && argi + 1 < argc)
        {
            const char *m = argv[++argi];
            if (strcmp(m, "sieve") == 0)
                method = METHOD_SIEVE;
            else if (strcmp(m, "powm") == 0)
                method = METHOD_POWM;
            else
            {
                if (rank == 0)
                    fprintf(stderr,
                            "Unknown --method '%s' (alg4_mpi supports "
                            "sieve and powm)\n",
                            m);
                MPI_Finalize();
                return 1;
            }
        }
        else if (strcmp(argv[argi], "--json") == 0 && argi + 1 < argc)
        {
            json_file = argv[++argi];
        }
        else if (argv[argi][0] != '-')
        {
            if (positional == 0)
            {
                challenge_file = argv[argi];
                positional++;
            }
            else if (positional == 1)
            {
                modulo_file = argv[argi];
                positional++;
            }
            else
            {
                if (rank == 0)
                    fprintf(stderr, "Unexpected argument '%s'\n", argv[argi]);
                MPI_Finalize();
                return 1;
            }
        }
        else
        {
            if (rank == 0)
                fprintf(stderr, "Unknown option '%s'\n", argv[argi]);
            MPI_Finalize();
            return 1;
        }
    }

    if (!run_all && (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: alg4_mpi <challenge> <modulo> "
                    "[--method sieve|powm] [--bench N]\n"
                    "       alg4_mpi --all [--data DIR] [--bench N]\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("=== alg4_mpi (Algorithm 4 — 2^64-smooth + pre-filter) ===\n");
        printf("MPI ranks:      %d\n", nranks);
#ifdef _OPENMP
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
#endif
        printf("Method:         %s\n",
               method == METHOD_SIEVE ? "sieve" : "powm");
        printf("Pre-filter K:   %d candidates / outer attempt\n",
               ALG4_PREFILTER_K);
        printf("Max outer:      %d attempts (%d total table entries)\n",
               ALG4_MAX_OUTER, BLIND_TABLE_SIZE);
        if (bench_count > 1)
            printf("Benchmark runs: %d\n", bench_count);
        if (run_all)
            printf("Mode:           --all\n");
        fflush(stdout);
    }

    // Generate small primes ≤ 65536 (used for both sieve marking and B0
    // pre-filter scoring; no separate B0 prime generation needed).
    int sp_count;
    unsigned *sp = kf_generate_small_primes(&sp_count);

    // Partition sieve range [65537, 2^32] evenly across MPI ranks.
    const uint64_t RANGE_LO = 65537ULL;
    const uint64_t RANGE_HI = (uint64_t)UINT32_MAX;
    const uint64_t range_size = RANGE_HI - RANGE_LO + 1;

    uint64_t my_lo = RANGE_LO + (range_size * (uint64_t)rank) / (uint64_t)nranks;
    uint64_t my_hi = RANGE_LO + (range_size * (uint64_t)(rank + 1)) / (uint64_t)nranks - 1;
    if (rank == nranks - 1)
        my_hi = RANGE_HI;

    if (rank == 0)
    {
        printf("Prime range per rank: ~%.1fM primes\n\n",
               (double)range_size / nranks / 1e6);
        fflush(stdout);
    }

    // Warm OMP thread pool.
#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    if (run_all)
    {
        const char *sizes[] = {
            "256", "512", "1024", "2048", "4096",
            "6144", "8192", "12288", "16384", "20480", "24576"};
        int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
        for (int s = 0; s < nsizes; s++)
        {
            char cf[512], mf[512];
            snprintf(cf, sizeof(cf),
                     "%s/challenge_0_alg4_%s.txt", data_dir, sizes[s]);
            snprintf(mf, sizeof(mf),
                     "%s/modulo_alg4_%s.txt", data_dir, sizes[s]);
            solve_one(rank, nranks, cf, mf, bench_count,
                      sp_count, sp, my_lo, my_hi, method, json_file);
        }
    }
    else
    {
        solve_one(rank, nranks, challenge_file, modulo_file, bench_count,
                  sp_count, sp, my_lo, my_hi, method, json_file);
    }

    free(sp);
    MPI_Finalize();
    return 0;
}