// =============================================================================
// alg4.cpp — Algorithm 4 solver: 2^64-smooth cube root recovery
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
//   PRE-FILTER (the paper's Algorithm 4, §3.2 — threshold-discard):
//     Before paying for a full MPI sieve, rank 0 screens blinding candidates
//     against the B0 = 2^20 prime base (the paper's value).  For each candidate
//     it divides out every prime ≤ B0 and inspects the residue: if the residue
//     exceeds the threshold t = 2^76 the candidate is DISCARDED (it cannot be
//     (B, B')-almost-smooth), otherwise it is broadcast for the full B = 2^32
//     sieve.  Rank 0 scans the blinding table, discarding failing candidates,
//     until one passes (sieved) or the table is exhausted (run fails).
//
//     Effect: most candidates are rejected cheaply, so each expensive full
//     sieve operates on a y that is plausibly (2^32, 2^65)-almost-smooth —
//     reducing the expected number of full sieve passes (paper: R 2^20 → 2^12).
//
// Pre-filter parameters (§3.2):
//   B0  = 2^20  (the paper's value; π(2^20) ≈ 82,025 primes; alg4_b0_primes())
//   t   = 2^76  (discard threshold on the B0-residue; ALG4_T_THRESHOLD_BITS)
//   blinding table = 256 candidates scanned before the run fails
//
//   blinding_attempts in JSON = candidates evaluated = 1 (unblinded round 0) +
//   the number of blinding-table entries examined (including discards):
//     round 0 hits directly → blinding_attempts = 1  (the crafted-benchmark case)
//     table exhausted        → blinding_attempts = 1 + 256
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
//   alg4_mpi [challenge] [modulo] [--method sieve|powm]
//   alg4_mpi --all [--data DIR]
// =============================================================================

#include "common.h"
#include "factorization.h"
#include "delta_table.h" // experiments/delta — L2-residency experiment
#include "solver_common.h"
#include "sizes.h"

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

// run_method_t defined in common.h

#define ALG4_T_THRESHOLD_BITS 76 // discard if B0-residue > 2^76 (paper §3.2)
#define BLIND_TABLE_SIZE 256     // blinding candidates available to the pre-filter scan

// kprint_stat() defined in common.h

#define ALG4_B0 (1UL << 20) // pre-filter smoothness bound (paper §3.2)

static const unsigned long *alg4_b0_primes(int *count_out)
{
    static unsigned long *primes = nullptr;
    static int n = 0;
    if (!primes)
        primes = generate_primes_up_to(ALG4_B0, &n); // one-time, rank-0 sequential
    *count_out = n;
    return primes;
}

// ---------------------------------------------------------------------------
// b0_residue_bits
//
// Returns the bit-length of the residue left after dividing every prime ≤ B0
// out of y (0 if y is fully B0-smooth).  This is the quantity the paper's
// pre-filter (§3.2) thresholds against: a candidate whose B0-residue exceeds
// t = 2^76 cannot be (B, B')-almost-smooth, so it is discarded without paying
// for the expensive full B=2^32 sieve.
//
// Uses the B0 = 2^20 prime base.  Operates on a copy of y; the original is not
// modified.  Early exit once the remainder reaches 1.
// ---------------------------------------------------------------------------

static int b0_residue_bits(const mpz_t y,
                           const unsigned long *b0_primes, int b0_count)
{
    mpz_t rem;
    mpz_init_set(rem, y);

    for (int i = 0; i < b0_count && mpz_cmp_ui(rem, 1) > 0; i++)
    {
        unsigned long p = b0_primes[i];
        while (mpz_divisible_ui_p(rem, p))
            mpz_divexact_ui(rem, rem, p);
    }

    int rem_bits = (mpz_cmp_ui(rem, 1) == 0) ? 0 : (int)mpz_sizeinbase(rem, 2);
    mpz_clear(rem);
    return rem_bits;
}

// ---------------------------------------------------------------------------
// run_one — run benchmark for one challenge file, return results (no JSON write)
// ---------------------------------------------------------------------------

static benchmark_results_t run_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method, const char *primes_file,
    unsigned long exponent,
    const char *cbrt_dir = KSC_CBRT_DIR,
    bool skip_powm = false)
{
    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    if (!ksc_broadcast_challenge_modulo(challenge, modulo,
                                        challenge_file, modulo_file, rank))
    {
        mpz_clears(challenge, modulo, result, NULL);
        return benchmark_results_t{}; // unreachable — ksc_broadcast aborts
    }

    if (rank == 0)
    {
        const char *mname = method == METHOD_TABLE  ? "table"
                            : method == METHOD_POWM ? "powm"
                                                    : "sieve";
        printf("\n=== %s  [%s | %dr×%dt | %zub] ===\n",
               challenge_file, mname, nranks,
#ifdef _OPENMP
               omp_get_max_threads(),
#else
               1,
#endif
               mpz_sizeinbase(challenge, 2));
        fflush(stdout);
    }

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    // e_inv = e^-1 mod (M-1) — same on every rank
    mpz_t e_inv;
    mpz_init(e_inv);
    ksc_compute_e_inv(e_inv, modulo, exponent);

    // Load prime slice for table method (once, outside bench loop).  The table
    // file may be flat (k32_primes.bin) or delta (k32_primes.dlt) — auto-detect
    // by magic (same shared k32 table as alg2_k32; see wiki delta-prime-table).
    uint32_t *prime_slice = NULL;
    int64_t slice_count = 0;
    kf_delta_slice_t delta_slice;
    memset(&delta_slice, 0, sizeof(delta_slice));
    bool have_delta = false;
    if (method == METHOD_TABLE)
    {
        bool is_delta = false;
        FILE *pf = fopen(primes_file, "rb");
        if (pf)
        {
            uint64_t m = 0;
            if (fread(&m, 8, 1, pf) == 1 &&
                (m == KF_DELTA_MAGIC_U32 || m == KF_DELTA_MAGIC_U64))
                is_delta = true;
            fclose(pf);
        }
        if (is_delta)
        {
            if (!kf_load_delta_slice(&delta_slice, primes_file, my_lo, my_hi))
            {
                fprintf(stderr, "[rank %d] Failed to load delta slice from %s\n",
                        rank, primes_file);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            have_delta = true;
        }
        else
        {
            uint64_t sc = 0;
            prime_slice = kf_load_prime_slice(&sc, primes_file,
                                              (uint32_t)my_lo, (uint32_t)my_hi);
            slice_count = (int64_t)sc;
            if (!prime_slice)
            {
                fprintf(stderr, "[rank %d] Failed to load prime slice from %s\n",
                        rank, primes_file);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    char cbrt_cache_path[KSC_CBRT_PATH_MAX];
    ksc_make_cbrt_cache_path(cbrt_cache_path, sizeof(cbrt_cache_path),
                             cbrt_dir, modulo_file, ".cbrt_alg4", rank); // per-rank fallback

    cbrt_cache_t local_cbrt_cache;
    kf_cbrt_cache_init(&local_cbrt_cache);
    bool cbrt_cache_dirty = false;
    bool used_shared = false;

    if (method != METHOD_POWM)
    {
        char shared_cbrt_path[KSC_CBRT_PATH_MAX];
        ksc_shared_cbrt_path(shared_cbrt_path, sizeof(shared_cbrt_path),
                             cbrt_dir, modulo_file);
        used_shared = kf_cbrt_cache_load(&local_cbrt_cache, shared_cbrt_path);
        if (rank == 0)
            printf(used_shared ? "cbrt: loaded merged %s (%d entries)\n"
                               : "cbrt: no merged cache at %s — per-rank build\n",
                   shared_cbrt_path, used_shared ? local_cbrt_cache.count : 0);
        if (!used_shared)
            ksc_ensure_cbrt_cache(
                &local_cbrt_cache, &cbrt_cache_dirty, cbrt_cache_path,
                challenge, modulo, e_inv,
                my_lo, my_hi, sp_count, sp,
                (method == METHOD_TABLE) ? prime_slice : NULL,
                (method == METHOD_TABLE) ? slice_count : 0,
                false, rank);
    }

    // Precompute cbrt(q0) for a single medium-prime cofactor before the bench
    // loop, so warm runs on the common (single prime ≤ 2^64) path are instant
    // cache hits instead of paying ~1800ms powm on run 1.
    if (!used_shared)
    {
        kfactor_list_t pre_factors;
        kfl_init(&pre_factors);
        if (have_delta)
            kf_trial_divide_delta(challenge, &delta_slice, &pre_factors);
        else if (method == METHOD_TABLE)
            kf_trial_divide_table(challenge, prime_slice, slice_count, &pre_factors);
        else
            kf_trial_divide_sieve(challenge, my_lo, my_hi, sp_count, sp, &pre_factors);

        mpz_t local_prod;
        mpz_init_set_ui(local_prod, 1);
        for (int i = 0; i < pre_factors.count; i++)
        {
            unsigned long p = mpz_get_ui(pre_factors.items[i].prime);
            for (int ex = 0; ex < pre_factors.items[i].exponent; ex++)
                mpz_mul_ui(local_prod, local_prod, p);
        }
        kfl_free(&pre_factors);

        mpz_t cofactor;
        mpz_init_set_ui(cofactor, 1);
        int complete = ksc_alg3_broadcast_cofactor(
            cofactor, challenge, local_prod, cbrt_bytes, rank, nranks);
        mpz_clear(local_prod);

        if (rank == 0 && complete && mpz_cmp_ui(cofactor, 1) > 0 &&
            mpz_fits_ulong_p(cofactor))
        {
            unsigned long q_ul = mpz_get_ui(cofactor);
            if (kf_cbrt_cache_find(&local_cbrt_cache, q_ul) < 0)
            {
                mpz_t cbrt_q;
                mpz_init(cbrt_q);
                mpz_powm(cbrt_q, cofactor, e_inv, modulo);
                kf_cbrt_cache_append(&local_cbrt_cache, q_ul, cbrt_q);
                cbrt_cache_dirty = true;
                mpz_clear(cbrt_q);
            }
        }
        mpz_clear(cofactor);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Blinding table: BLIND_TABLE_SIZE = PREFILTER_K × MAX_OUTER entries.
    // Rank 0 uses these for both pre-filtering (K per outer attempt) and
    // unblinding (r_inv of the winning entry).
    // Non-zero ranks never touch the table — they just receive the chosen y.
    blinding_pair_t *blind_table = NULL;
    if (rank == 0)
    {
        blind_table = kf_blinding_table_alloc(BLIND_TABLE_SIZE, modulo, exponent);
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
    const char *mname = (method == METHOD_TABLE)  ? "table"
                        : (method == METHOD_POWM) ? "powm"
                                                  : "sieve";
    // BUG FIX: k_size = 32, not 64.  Alg4 sieves [65537, 2^32] — same range
    // as alg3.  The k_size field in JSON records the sieve upper bound in bits.
    ksc_init_json_results(&json_results, 32, mname,
                          nranks, n_threads_json, challenge, rank);
    json_results.config.algorithm = "alg4";
    json_results.config.exponent = exponent;
    json_results.config.b0 = ALG4_B0;
    json_results.config.t_threshold_bits = ALG4_T_THRESHOLD_BITS;

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

            if (have_delta)
                kf_trial_divide_delta(wu_ch, &delta_slice, &wu_factors);
            else if (method == METHOD_TABLE)
                kf_trial_divide_table(wu_ch, prime_slice, slice_count,
                                      &wu_factors, nullptr, nullptr, 0);
            else
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
                json_results.timing.warmup_total_ms.push_back(
                    std::chrono::duration<double, std::milli>(wu_t1 - wu_t0).count());

            mpz_clears(wu_ch, wu_cbrt, NULL);
            kfl_free(&wu_factors);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // ---- Benchmark loop ----

    // Single timed run per challenge (the `bench < 1` loop runs once; the old
    // per-challenge repeat / --bench is removed).  Multiple challenges are run
    // by the caller and merged.
    for (int bench = 0; bench < 1 && method != METHOD_POWM; bench++)
    {
        kfactor_list_t local_factors;
        kfl_init(&local_factors);
        double iter_gat = 0.0, iter_com = 0.0;

        kfl_free(&medium_factors);
        kfl_init(&medium_factors);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        // winning_table_idx: which blinding table entry was sieved.
        //   -1  → unblinded (round 0, y = challenge)
        //   ≥ 0 → index into blind_table[], used for unblinding via r_inv.
        int winning_table_idx = -1;
        bool bench_succeeded = false;
        int table_cursor = 0; // blinding-table entries examined (incl. discards)

        size_t blinded_nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
        unsigned char *blinded_buf = (unsigned char *)calloc(blinded_nbytes, 1);

        // ---- Pre-filter loop (paper §3.2: B0 threshold-discard) ----
        //
        // Round 0: unblinded (y = challenge directly, no pre-filter).  If the
        //          challenge itself is (B, B')-almost-smooth this succeeds on
        //          the first try — the common case for crafted benchmarks.
        //
        // Rounds 1..: scan the blinding table.  For each candidate, compute the
        //          residue after dividing out all primes ≤ B0 = 2^20; if it
        //          exceeds t = 2^76 the candidate is DISCARDED with no full
        //          sieve.  The first candidate whose residue ≤ t is sieved.
        //          `do_sieve` is broadcast each round so every rank performs the
        //          same number of (collective) full-sieve rounds; when the table
        //          is exhausted with nothing passing, all ranks stop together.

        for (int round = 0; !bench_succeeded; round++)
        {
            int do_sieve = 1; // 0 → table exhausted this round, stop (uniform)

            // -- Rank 0: choose the candidate to sieve (or signal exhaustion) --
            if (rank == 0)
            {
                if (round == 0)
                {
                    // No blinding: use challenge as-is.
                    mpz_set(blinded_challenge, challenge);
                    winning_table_idx = -1;
                }
                else
                {
                    // Threshold-discard: scan from the cursor, skipping any
                    // candidate whose B0-residue exceeds t = 2^76, until one
                    // passes (sieved) or the table is exhausted (run fails).
                    winning_table_idx = -1;
                    int b0c = 0;
                    const unsigned long *b0p = alg4_b0_primes(&b0c);
                    mpz_t y_try;
                    mpz_init(y_try);
                    while (table_cursor < BLIND_TABLE_SIZE)
                    {
                        int idx = table_cursor++;
                        mpz_mul(y_try, challenge, blind_table[idx].r_blind);
                        mpz_mod(y_try, y_try, modulo);
                        if (b0_residue_bits(y_try, b0p, b0c) <= ALG4_T_THRESHOLD_BITS)
                        {
                            mpz_set(blinded_challenge, y_try); // passes the filter
                            winning_table_idx = idx;
                            break;
                        }
                        // residue > t → discard (no full sieve)
                    }
                    mpz_clear(y_try);
                    if (winning_table_idx < 0)
                        do_sieve = 0; // table exhausted, none passed → run fails
                }

                if (do_sieve)
                {
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
            }

            // All ranks learn whether a full sieve happens this round (uniform,
            // so the collective sieve/gather below is reached by all or none).
            MPI_Bcast(&do_sieve, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!do_sieve)
                break; // table exhausted → bench_succeeded stays false

            // Broadcast the chosen blinded challenge + its table index.
            MPI_Bcast(blinded_buf, (int)blinded_nbytes, MPI_UNSIGNED_CHAR,
                      0, MPI_COMM_WORLD);
            MPI_Bcast(&winning_table_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank != 0)
                mpz_import(blinded_challenge, blinded_nbytes, 1, 1, 0, 0,
                           blinded_buf);

            // -- All ranks: full sieve/table on the chosen blinded challenge --
            kfl_free(&local_factors);
            kfl_init(&local_factors);
            if (have_delta)
                kf_trial_divide_delta(blinded_challenge, &delta_slice, &local_factors);
            else if (method == METHOD_TABLE)
                kf_trial_divide_table(blinded_challenge, prime_slice, slice_count,
                                      &local_factors, nullptr, nullptr, 0);
            else
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

            if (complete)
                bench_succeeded = true;

        } // end pre-filter loop

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

            if (bench_succeeded)
                json_results.timing.factor_times.push_back(dt);

            if (bench_succeeded)
            {
                // blinding_attempts: total candidates evaluated = 1 (unblinded
                // round 0) + the number of blinding-table entries examined by
                // the threshold scan (including the ones discarded for residue
                // > t).  1 means round 0 hit directly (the crafted-benchmark
                // case); the maximum is 1 + BLIND_TABLE_SIZE.
                int total_candidates = 1 + table_cursor;
                json_results.timing.blinding_attempts.push_back(total_candidates);

                // Hit classification.
                int nmed = medium_factors.count;
                int htype = (nmed == 0) ? 1 : (nmed == 1) ? 2
                                                          : 3;
                json_results.hits.hit_type.push_back(htype);

                if (nmed == 0)
                {
                    json_results.hits.cofactor_bits.push_back(1);
                }
                else
                {
                    mpz_t prod;
                    mpz_init_set_ui(prod, 1);
                    for (int i = 0; i < nmed; i++)
                        mpz_mul(prod, prod, medium_factors.items[i].prime);
                    json_results.hits.cofactor_bits.push_back(
                        (int)mpz_sizeinbase(prod, 2));
                    mpz_clear(prod);
                }
                json_results.hits.medium_factor_count.push_back(nmed);
            }
        }

        // ---- DISTRIBUTED RECONSTRUCTION ----

        auto t_loc0 = std::chrono::high_resolution_clock::now();

        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double cbrt_hit_ms = 0.0, cbrt_miss_ms = 0.0;
        int cbrt_hits = 0, cbrt_misses = 0;
        bool record_cbrt = true;
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
        //
        // NOTE: hit/miss accounting here feeds the same cbrt_hits/cbrt_misses
        // counters reported in JSON so type-3 cache behaviour is visible.
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
                    if (record_cbrt)
                        cbrt_hits++;
                }
                else
                {
                    // Cache miss: one full powm (~1800ms for large M).
                    // Happens only on the first encounter of this medium prime.
                    // After this point the result is cached so subsequent runs
                    // (including other bench iterations) pay only a lookup cost.
                    auto _mt0 = std::chrono::high_resolution_clock::now();
                    mpz_powm(cbrt_qi, medium_factors.items[i].prime,
                             e_inv, modulo);
                    auto _mt1 = std::chrono::high_resolution_clock::now();

                    if (record_cbrt)
                    {
                        cbrt_miss_ms += std::chrono::duration<double, std::milli>(
                                            _mt1 - _mt0)
                                            .count();
                        cbrt_misses++;
                    }

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
        if (bench == 0)
            ksc_gather_prime_stats(&json_results,
                                   local_factors.count,
                                   (int64_t)(my_hi - my_lo + 1),
                                   rank, nranks);

        if (rank == 0 && bench_succeeded)
        {
            json_results.timing.cbrt_hit_ms.push_back(cbrt_hit_ms);
            json_results.timing.cbrt_miss_ms.push_back(cbrt_miss_ms);
            json_results.timing.cbrt_hits.push_back(cbrt_hits);
            json_results.timing.cbrt_misses.push_back(cbrt_misses);
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
            if (bench_succeeded)
                json_results.timing.local_cbrt_times.push_back(max_local_dt);
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
            if (bench_succeeded)
                json_results.timing.gather_times.push_back(iter_gat2);

            accum_record(&com_acc, iter_com2);
            iter_com = iter_com2;
            if (bench_succeeded)
                json_results.timing.combine_times.push_back(iter_com2);

            double recon_dt = cur_max_loc + iter_gat + iter_com;
            accum_record(&recon_acc, recon_dt);
            if (bench_succeeded)
            {
                json_results.timing.recon_times.push_back(recon_dt);
                if (!json_results.timing.factor_times.empty())
                    json_results.timing.total_times.push_back(
                        json_results.timing.factor_times.back() + recon_dt);
            }
        }

        kfl_free(&local_factors);

    } // end bench loop

    // ---- Timing report ----
    if (rank == 0)
    {
        ksc_print_timing_report(&fac_acc, &loc_acc,
                                &gat_acc, &com_acc, &recon_acc);
        if (!skip_powm)
            ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                             &recon_acc, &fac_acc, rank, exponent);

        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_eth_root(result, challenge_mod, modulo, exponent);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    // ---- Cleanup ----
    free(prime_slice);
    if (have_delta)
        kf_delta_slice_free(&delta_slice);
    mpz_clear(blinded_challenge);
    kfl_free(&medium_factors);
    if (rank == 0)
        kf_blinding_table_free(blind_table, BLIND_TABLE_SIZE);

    if (!used_shared && cbrt_cache_dirty) // merged cache is read-only
        kf_cbrt_cache_save(&local_cbrt_cache, cbrt_cache_path);
    kf_cbrt_cache_clear(&local_cbrt_cache);

    mpz_clear(e_inv);
    mpz_clears(challenge, modulo, result, NULL);

    return json_results;
}

// solve_one — backward-compatible wrapper: run_one + write JSON
static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method, const char *primes_file,
    unsigned long exponent,
    const char *json_output_file = nullptr,
    const char *cbrt_dir = KSC_CBRT_DIR,
    bool skip_powm = false)
{
    benchmark_results_t r = run_one(rank, nranks, challenge_file, modulo_file,
                                    sp_count, sp, my_lo, my_hi,
                                    method, primes_file, exponent, cbrt_dir, skip_powm);
    if (rank == 0 && json_output_file && json_output_file[0] != '\0')
        write_benchmark_json(json_output_file, r);
    return 1;
}

// Derive the delta table path from the flat path: k32_primes.bin -> k32_primes.dlt
static void alg4_delta_path(char *dst, size_t n, const char *flat)
{
    snprintf(dst, n, "%s", flat);
    size_t L = strlen(dst);
    if (L > 4 && strcmp(dst + L - 4, ".bin") == 0)
        snprintf(dst + L - 4, 5, ".dlt");
    else if (L + 5 <= n)
        snprintf(dst + L, 5, ".dlt");
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
    // Must match ksc_ensure_k32_primes and the CUDA solvers, which both use
    // data/k32_primes.bin. Defaulting to the repo root here made
    // `--precompute` write a table that `--method table` then ignored and
    // regenerated, costing a second 813 MB copy.
    const char *primes_file = "data/k32_primes.bin";
    const char *json_file = NULL;
    const char *modulo_suffix = "";
    const char *cbrt_dir = KSC_CBRT_DIR;
    int run_all = 0;
    int run_all_multi = 0;
    bool use_delta = false;   // --table-format delta: use the delta-encoded table
    bool primes_user_set = false;
    int n_challenges = 1;
    bool skip_powm = false;
    run_method_t method = METHOD_SIEVE;
    unsigned long exponent = 3;

    int positional = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--all") == 0)
        {
            run_all = 1;
        }
        else if (strcmp(argv[argi], "--multi") == 0)
        {
            run_all_multi = 1;
        }
        else if (strcmp(argv[argi], "--data") == 0 && argi + 1 < argc)
        {
            data_dir = argv[++argi];
        }
        else if (strcmp(argv[argi], "--modulo-suffix") == 0 && argi + 1 < argc)
        {
            modulo_suffix = argv[++argi];
        }
        else if (strcmp(argv[argi], "--method") == 0 && argi + 1 < argc)
        {
            const char *m = argv[++argi];
            if (strcmp(m, "sieve") == 0)
                method = METHOD_SIEVE;
            else if (strcmp(m, "table") == 0)
                method = METHOD_TABLE;
            else if (strcmp(m, "powm") == 0)
                method = METHOD_POWM;
            else
            {
                if (rank == 0)
                    fprintf(stderr,
                            "Unknown --method '%s' (use sieve, table, or powm)\n", m);
                MPI_Finalize();
                return 1;
            }
        }
        else if (strcmp(argv[argi], "--primes") == 0 && argi + 1 < argc)
        {
            primes_file = argv[++argi];
            primes_user_set = true;
        }
        else if (strcmp(argv[argi], "--table-format") == 0 && argi + 1 < argc)
        {
            const char *tf = argv[++argi];
            if (strcmp(tf, "delta") == 0)
                use_delta = true;
            else if (strcmp(tf, "flat") == 0)
                use_delta = false;
            else
            {
                if (rank == 0)
                    fprintf(stderr, "Unknown --table-format '%s' (use flat or delta)\n", tf);
                MPI_Finalize();
                return 1;
            }
        }
        else if (strcmp(argv[argi], "--json") == 0 && argi + 1 < argc)
        {
            json_file = argv[++argi];
        }
        else if (strcmp(argv[argi], "--exponent") == 0 && argi + 1 < argc)
        {
            exponent = (unsigned long)atol(argv[++argi]);
        }
        else if (strcmp(argv[argi], "--n-challenges") == 0 && argi + 1 < argc)
        {
            n_challenges = atoi(argv[++argi]);
            if (n_challenges < 1)
                n_challenges = 1;
        }
        else if (strcmp(argv[argi], "--cbrt-dir") == 0 && argi + 1 < argc)
        {
            cbrt_dir = argv[++argi];
        }
        else if (strcmp(argv[argi], "--skip-powm-baseline") == 0)
        {
            skip_powm = true;
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

    if (!run_all && !run_all_multi &&
        (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: alg4_mpi <challenge> <modulo> "
                    "[--method sieve|table|powm] [--table-format flat|delta]\n"
                    "       alg4_mpi --all  [--data DIR] [--method sieve|table|powm] [--table-format flat|delta]\n"
                    "       alg4_mpi --multi [--data DIR] [--method sieve|table|powm]\n"
                    "       (--primes FILE defaults to k32_primes.bin; only used with --method table)\n");
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
               method == METHOD_SIEVE   ? "sieve"
               : method == METHOD_TABLE ? "table"
                                        : "powm");
        if (method == METHOD_TABLE)
            printf("Table format:   %s\n",
                   use_delta ? "delta (1 byte/prime, ~204 MB)" : "flat (uint32, ~813 MB)");
        printf("Pre-filter:     B0=2^20 threshold-discard, t=2^%d (paper §3.2)\n",
               ALG4_T_THRESHOLD_BITS);
        printf("Blinding table: %d candidates scanned before failure\n",
               BLIND_TABLE_SIZE);
        if (run_all)
            printf("Mode:           --all\n");
        if (run_all_multi)
            printf("Mode:           --all --multi\n");
        if (n_challenges > 1)
            printf("Challenges:     %d  (all timed warm runs, 0..%d averaged)\n",
                   n_challenges, n_challenges - 1);
        fflush(stdout);
    }

    // Generate small primes ≤ 65536 (used for both sieve marking and B0
    // pre-filter scoring; no separate B0 prime generation needed).
    int sp_count;
    unsigned *sp = kf_generate_small_primes(&sp_count);

    if (method == METHOD_TABLE)
    {
        const char *flat = ksc_ensure_k32_primes(sp_count, sp, rank);
        if (use_delta)
        {
            // Point at the shared delta table (k32_primes.dlt); rank 0 builds it
            // from the flat table on first use.  run_one auto-detects it by magic.
            static char dpath[512];
            if (primes_user_set)
                snprintf(dpath, sizeof(dpath), "%s", primes_file);
            else
                alg4_delta_path(dpath, sizeof(dpath), flat);
            if (rank == 0)
            {
                FILE *t = fopen(dpath, "rb");
                if (t)
                    fclose(t);
                else if (!kf_convert_prime_table_to_delta(flat, dpath, "[rank 0] "))
                    MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Barrier(MPI_COMM_WORLD); // all ranks wait for the .dlt to exist
            primes_file = dpath;
        }
        else
            primes_file = flat;
    }

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

    if (run_all || run_all_multi)
    {
        // --all:         challenge_<N>_alg4_<bits>.txt       / modulo_alg4_<bits>.txt
        // --all --multi: challenge_<N>_alg4_multi_<bits>.txt / modulo_alg4_multi_<bits>.txt
        const char *sizes[] = {KSC_CANONICAL_SIZES_LIST};
        const char *pat = run_all_multi ? "alg4_multi" : "alg4";
        for (int s = 0; s < KSC_N_CANONICAL_SIZES; s++)
        {
            char cf[512], mf[512], jf[768];
            snprintf(mf, sizeof(mf), "%s/modulo_%s_%s%s.txt",
                     data_dir, pat, sizes[s], modulo_suffix);
            const char *jf_ptr = NULL;
            if (json_file)
            {
                snprintf(jf, sizeof(jf), "%s_%s.json", json_file, sizes[s]);
                jf_ptr = jf;
            }
            if (n_challenges == 1)
            {
                snprintf(cf, sizeof(cf), "%s/challenge_0_%s_%s.txt", data_dir, pat, sizes[s]);
                solve_one(rank, nranks, cf, mf,
                          sp_count, sp, my_lo, my_hi, method, primes_file, exponent,
                          jf_ptr, cbrt_dir, skip_powm);
            }
            else
            {
                benchmark_results_t combined;
                bool has_data = false;
                for (int ci = 0; ci < n_challenges; ci++)
                {
                    char cfi[512];
                    snprintf(cfi, sizeof(cfi), "%s/challenge_%d_%s_%s.txt",
                             data_dir, ci, pat, sizes[s]);
                    int exists = 0;
                    if (rank == 0)
                    {
                        FILE *fck = fopen(cfi, "r");
                        if (fck)
                        {
                            fclose(fck);
                            exists = 1;
                        }
                    }
                    MPI_Bcast(&exists, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (!exists)
                    {
                        if (rank == 0)
                            printf("  [SKIP] challenge_%d not found for size %s\n",
                                   ci, sizes[s]);
                        continue;
                    }
                    benchmark_results_t r = run_one(rank, nranks, cfi, mf,
                                                    sp_count, sp, my_lo, my_hi,
                                                    method, primes_file, exponent,
                                                    cbrt_dir, skip_powm);
                    if (!has_data)
                    {
                        combined = std::move(r);
                        has_data = true;
                    }
                    else
                    {
                        merge_benchmark_results(combined, r);
                    }
                }
                if (has_data && rank == 0 && jf_ptr)
                    write_benchmark_json(jf_ptr, combined);
            }
        }
    }
    else
    {
        solve_one(rank, nranks, challenge_file, modulo_file,
                  sp_count, sp, my_lo, my_hi, method, primes_file, exponent,
                  json_file, cbrt_dir, skip_powm);
    }

    free(sp);
    MPI_Finalize();
    return 0;
}
