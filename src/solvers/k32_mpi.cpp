// =============================================================================
// k32_mpi.cpp — MPI+OpenMP hybrid cube root solver for 32-bit prime factors
//
// Architecture:
//   MPI layer:    splits prime range [65537, 2^32] across nodes
//   OpenMP layer: parallelises sieve / trial division within each node
//
// Key improvement over previous version — distributed reconstruction:
//   OLD: rank 0 gathers all (prime, exponent) pairs and does every powm itself.
//        For 7000-bit challenges this cost ~650 ms regardless of node count.
//   NEW: every rank computes cbrt for its own factors locally (parallel powm
//        across all ranks), then each rank sends only ONE fixed-size partial
//        product (~875 bytes) to rank 0. Rank 0 multiplies nranks numbers.
//        Reconstruction now scales with the rest of the computation.
//
// Communication pattern:
//   MPI_Bcast:   challenge + modulo bytes → all ranks (once per challenge)
//   MPI_Barrier: sync for accurate timing
//   MPI_Reduce:  max local-cbrt time across ranks → rank 0 (timing only)
//   MPI_Barrier: ensure all ranks reached gather before measuring comm time
//   MPI_Gather:  partial cbrt bytes from each rank → rank 0
//                payload: nranks × ceil(modulo_bits/8)+1 ≈ nranks × 875 bytes
//
// Methods:
//   sieve  — segmented sieve, primes generated on-the-fly   [default]
//   table  — precomputed prime file; each rank loads its slice via binary
//             search so only ~range/nranks × 4 bytes are read per rank
//
// Usage:
//   k32_mpi [challenge] [modulo] [--method sieve|table] [--primes FILE]
//           [--precompute] [--bench N]
//   k32_mpi --all [--data DIR] [--method sieve|table] [--primes FILE]
//           [--bench N]
//
// Prime table generation (rank 0 only, one-time, ~10 s):
//   srun ... k32_mpi --precompute [--primes k32_primes.bin]
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
    METHOD_TABLE,
    METHOD_POWM,
} run_method_t;

// ---------------------------------------------------------------------------
// Small prime generator (same as k32.cpp)
// ---------------------------------------------------------------------------

// kf_kf_generate_small_primes() — in k_factorization.cpp

// ---------------------------------------------------------------------------
// Prime table I/O
//
// File format: [uint64_t magic][uint64_t count][uint32_t prime[count]]
// Sorted ascending, range [65537, 2^32], ~813 MB total.
//
// generate_and_save_prime_table: called by rank 0 only.
// load_prime_slice: called by every rank; uses binary search to seek
//   directly to [range_lo, range_hi] — only that slice is read into RAM.
// ---------------------------------------------------------------------------

// kf_generate_and_save_prime_table() — in k_factorization.cpp

// Binary search: index of first prime >= target in the sorted uint32_t file.
// Primes start at byte offset 16 (after magic + count header).
// prime_file_lower_bound() — internal to kf_load_prime_slice()

// Load only the slice [range_lo, range_hi] of the prime table.
// Returns a malloc'd uint32_t array; sets *slice_count.
// Returns NULL on failure (file not found or I/O error).
// kf_load_prime_slice() — in k_factorization.cpp

// kf_kf_trial_divide_sieve() — in k_factorization.cpp

// kf_kf_trial_divide_table() — in k_factorization.cpp

// Timing report: print_timing_stat_moments() from k_common.h at width 30.
// print_stat() is a thin wrapper kept for call-site readability.
#define print_stat(lbl, sum, sumsq, n) print_timing_stat_moments(lbl, sum, sumsq, n, 30, true)

// ---------------------------------------------------------------------------
// solve_one — solve a single (challenge, modulo) pair across all MPI ranks
//
// Reconstruction flow (NEW vs OLD):
//   OLD: gather all (prime,exp) pairs to rank 0 → rank 0 does all powm
//   NEW: every rank computes its partial cbrt locally (powm parallelised)
//        → MPI_Gather of nranks × ~875 bytes → rank 0 multiplies nranks terms
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count, int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method, const char *primes_file,
    const char *json_output_file = nullptr)
{
    if (rank == 0)
    {
        printf("\n=== %s ===\n", challenge_file);
        fflush(stdout);
    }

    // ---- Rank 0 reads files; all ranks get the values via Bcast ----
    // If rank 0 cannot open the files it broadcasts read_ok=0 and all ranks
    // call MPI_Abort together — avoiding the hang caused by exit(1) on rank 0
    // while other ranks block on MPI_Bcast.

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
        fflush(stdout);
    }

    // ---- Load prime slice for table method (once, outside bench loop) ----
    // Each rank loads only its [my_lo, my_hi] slice via binary search on the
    // sorted prime file — no full 800+ MB read per rank.

    uint32_t *prime_slice = NULL;
    int64_t slice_count = 0;

    if (method == METHOD_TABLE)
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
        if (rank == 0)
            printf("Rank 0 slice: %" PRId64 " primes\n", slice_count);
        fflush(stdout);
    }

    // Fixed-size byte buffer for partial cbrt export (big-endian, zero-padded).
    // Size is the same on all ranks since modulo is broadcast.
    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1; // +1 safety

    // Precompute e = 3^-1 mod (M-1) once — same value on every rank.
    mpz_t phi, e_inv, three_mpz;
    mpz_inits(phi, e_inv, three_mpz, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three_mpz, 3);
    mpz_invert(e_inv, three_mpz, phi);
    mpz_clears(phi, three_mpz, NULL);

    // Per-rank cbrt cache — persists cbrt(p) values across bench iterations.
    // Path: data/cbrt/<modulo_stem>.cbrt.<rank>
    // On a warm run (cache file present) all powms are skipped → ~0 ms recon.
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
                 "%s/%s.cbrt.%d", KSC_CBRT_DIR, stem, rank);
    }

    cbrt_cache_t local_cbrt_cache;
    kf_cbrt_cache_init(&local_cbrt_cache);
    bool cbrt_cache_dirty = false;

    // Guarantee cbrt cache is fully populated before any timed run.
    // If the file exists it is loaded; otherwise one untimed factoring
    // pass is run to compute all needed cbrt values and save them.
    ksc_ensure_k32_cbrt(
        &local_cbrt_cache, &cbrt_cache_dirty, cbrt_cache_path,
        challenge, modulo, e_inv,
        my_lo, my_hi, sp_count, sp,
        prime_slice, slice_count,
        rank, nranks, method == METHOD_TABLE ? 1 : 0);

    // Blinding pair table — precomputed on rank 0 only (Algorithm 2 precomputation).
    // Rank 0 uses table[attempt-1].r3 to compute the blinded challenge and
    // table[attempt-1].r_inv to unblind the result.  All ranks receive the
    // blinded challenge via MPI_Bcast so they factor the same number.
    // No mpz_powm_ui (r^3) or mpz_invert (r^{-1}) in the online loop.
    const int BLIND_TABLE_SIZE = 255;
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

    // Timing accumulators (rank 0 records all; max_local via MPI_Reduce).
    // recon_{sum,sq} track the per-iteration reconstruction total directly
    // (loc + gat + com per run) so that stddev is computed correctly.
    // Summing the squares of individual phases gives the wrong variance.
    bench_accum_t fac_acc = {0, 0};
    bench_accum_t loc_acc = {0, 0};
    bench_accum_t gat_acc = {0, 0};
    bench_accum_t com_acc = {0, 0};
    bench_accum_t recon_acc = {0, 0};
    double cur_max_loc = 0.0;

    // JSON results
    int n_threads_json = 1;
#ifdef _OPENMP
    n_threads_json = omp_get_max_threads();
#endif
    benchmark_results_t json_results;
    const char *mname32 = (method == METHOD_TABLE)  ? "table"
                          : (method == METHOD_POWM) ? "powm"
                                                    : "sieve";
    ksc_init_json_results(&json_results, 32, bench_count, mname32,
                          nranks, n_threads_json, challenge, rank);

    // ---- 2 warmup runs (silent, primes CPU caches before bench loop) ----
    // Runs the full factoring + reconstruction pipeline twice without
    // recording any timing. This ensures instruction cache, data cache,
    // and branch predictor are all hot before the first timed iteration.
    if (method != METHOD_POWM)
    {
        for (int wu = 0; wu < 2; wu++)
        {
            kfactor_list_t wu_factors;
            kfl_init(&wu_factors);
            // Blind and broadcast challenge (same as bench loop, r=1)
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
            if (method == METHOD_SIEVE)
                kf_trial_divide_sieve(wu_ch, my_lo, my_hi, sp_count, sp, &wu_factors);
            else
                kf_trial_divide_table(wu_ch, prime_slice, slice_count, &wu_factors);
            mpz_t wu_cbrt;
            mpz_init(wu_cbrt);
            // dummy cbrt (populates cache on first warmup)
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

    // ---- Benchmark loop (skipped for METHOD_POWM) ----

    for (int bench = 0; bench < bench_count && method != METHOD_POWM; bench++)
    {

        kfactor_list_t local_factors;
        kfl_init(&local_factors);
        // Per-iteration components at rank 0 — used for correct recon stddev
        double iter_gat = 0.0, iter_com = 0.0;

        // ---- BLINDING + FACTORING (retry loop) ----
        //
        // Rank 0 draws r, computes blinded = challenge * r^3 mod M, bcasts it.
        // All ranks factor their slice of blinded.  Rank 0 gathers each rank's
        // local factor product to check completeness (product == blinded).
        // On incomplete factorisation rank 0 signals retry via MPI_Bcast.
        // The factoring barrier and timing envelope span all retry attempts.

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        int attempt = 0;
        const int MAX_ATTEMPTS = 256;
        // blinded_nbytes: fixed size large enough for any value < M
        size_t blinded_nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
        unsigned char *blinded_buf = (unsigned char *)malloc(blinded_nbytes);

        for (attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
        {
            // -- Rank 0 computes blinded challenge and broadcasts it --
            //
            // attempt 0: r = 1, blinded = challenge (no arithmetic).
            // attempt i: use precomputed table[i-1].r3 — one mul + mod,
            //            no mpz_powm_ui, no mpz_invert in the hot path.
            if (rank == 0)
            {
                if (attempt == 0)
                {
                    mpz_set(blinded_challenge, challenge);
                }
                else
                {
                    int idx = attempt - 1;
                    if (idx < BLIND_TABLE_SIZE)
                    {
                        // Precomputed path: y = challenge * r^3 mod M.
                        mpz_mul(blinded_challenge, challenge,
                                blind_table[idx].r3);
                        mpz_mod(blinded_challenge, blinded_challenge, modulo);
                    }
                    else
                    {
                        // Fallback: table exhausted (shouldn't happen with
                        // MAX_ATTEMPTS <= BLIND_TABLE_SIZE + 1).
                        fprintf(stderr,
                                "[rank 0] WARNING: blinding table exhausted "
                                "at attempt %d\n",
                                attempt);
                        mpz_set(blinded_challenge, challenge);
                    }
                }

                // Export blinded_challenge to fixed-size big-endian buffer.
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

            // Broadcast blinded challenge to all ranks.
            MPI_Bcast(blinded_buf, (int)blinded_nbytes, MPI_UNSIGNED_CHAR,
                      0, MPI_COMM_WORLD);
            if (rank != 0)
                mpz_import(blinded_challenge, blinded_nbytes, 1, 1, 0, 0,
                           blinded_buf);

            // -- All ranks factor their slice of blinded_challenge --
            kfl_free(&local_factors);
            kfl_init(&local_factors);

            if (method == METHOD_SIEVE)
                kf_trial_divide_sieve(blinded_challenge, my_lo, my_hi,
                                      sp_count, sp, &local_factors);
            else
            {
                int nT = 1;
#ifdef _OPENMP
                nT = omp_get_max_threads();
#endif
                bool do_thr = (bench == 1 && rank == 0);
                std::vector<int64_t> thr_tested(do_thr ? nT : 0, 0);
                std::vector<int> thr_found(do_thr ? nT : 0, 0);
                kf_trial_divide_table(blinded_challenge, prime_slice,
                                      slice_count, &local_factors,
                                      do_thr ? thr_tested.data() : nullptr,
                                      do_thr ? thr_found.data() : nullptr,
                                      do_thr ? nT : 0);
                if (do_thr)
                {
                    json_results.thread_primes_tested.insert(
                        json_results.thread_primes_tested.end(),
                        thr_tested.begin(), thr_tested.end());
                    json_results.thread_primes_found.insert(
                        json_results.thread_primes_found.end(),
                        thr_found.begin(), thr_found.end());
                }
            }

            // -- Completeness check: gather local factor products to rank 0 --
            //
            // Each rank computes local_prod = ∏ p_i^{e_i} for its found factors
            // and exports it as a fixed-size buffer.  Rank 0 multiplies all
            // products and checks whether the result equals blinded_challenge.

            mpz_t local_prod;
            mpz_init_set_ui(local_prod, 1);
            for (int i = 0; i < local_factors.count; i++)
            {
                unsigned long p = mpz_get_ui(local_factors.items[i].prime);
                for (int ex = 0; ex < local_factors.items[i].exponent; ex++)
                    mpz_mul_ui(local_prod, local_prod, p);
            }

            // Export local_prod into a fixed-size buffer (same size as cbrt_bytes
            // — both are bounded by modulo, which is larger than any factor product).
            unsigned char *prod_buf = (unsigned char *)calloc(cbrt_bytes, 1);
            {
                size_t written = 0;
                unsigned char *tmp = (unsigned char *)mpz_export(
                    NULL, &written, 1, 1, 0, 0, local_prod);
                if (tmp && written > 0 && written <= cbrt_bytes)
                {
                    memcpy(prod_buf + cbrt_bytes - written, tmp, written);
                    free(tmp);
                }
            }
            mpz_clear(local_prod);

            unsigned char *all_prods = NULL;
            if (rank == 0)
                all_prods = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);

            MPI_Gather(prod_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                       all_prods, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                       0, MPI_COMM_WORLD);
            free(prod_buf);

            // Rank 0 checks completeness; broadcasts result.
            int complete = 0;
            if (rank == 0)
            {
                mpz_t combined_prod, part;
                mpz_init_set_ui(combined_prod, 1);
                mpz_init(part);
                for (int rk = 0; rk < nranks; rk++)
                {
                    mpz_import(part, cbrt_bytes, 1, 1, 0, 0,
                               all_prods + (size_t)rk * cbrt_bytes);
                    mpz_mul(combined_prod, combined_prod, part);
                }
                mpz_clear(part);
                free(all_prods);

                complete = (mpz_cmp(combined_prod, blinded_challenge) == 0) ? 1 : 0;
                mpz_clear(combined_prod);

                if (bench == 0)
                {
                    printf("Blinding attempt %d: %s\n", attempt + 1,
                           complete ? "smooth — proceeding" : "not smooth — retrying");
                    fflush(stdout);
                }
            }
            MPI_Bcast(&complete, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (complete)
                break;
        } // end blinding retry loop

        free(blinded_buf);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(
                            t_fac1 - t_fac0)
                            .count();
            accum_record(&fac_acc, dt);
            if (rank == 0 && bench > 0)
                json_results.factor_times.push_back(dt);
            if (bench == 0)
            {
                printf("Factors found by rank 0: %d distinct primes "
                       "(after %d attempt(s))\n",
                       local_factors.count, attempt + 1);
                fflush(stdout);
            }
            if (rank == 0 && bench > 0)
                json_results.blinding_attempts.push_back(attempt + 1);
        }

        // ---- DISTRIBUTED RECONSTRUCTION ----
        //
        // Step 1 — every rank computes partial cbrt for its own factors.
        //   Runs in parallel across all ranks (like the factoring phase).
        //   Each rank does ~(total_factors / nranks) powm calls.
        auto t_loc0 = std::chrono::high_resolution_clock::now();

        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double cbrt_hit_ms = 0.0, cbrt_miss_ms = 0.0;
        int cbrt_hits = 0, cbrt_misses = 0;
        bool record_cbrt = (bench > 0); // only warm runs
        ksc_compute_local_cbrt(local_cbrt, &local_factors, e_inv, modulo,
                           &local_cbrt_cache, &cbrt_cache_dirty,
                           &cbrt_hit_ms, &cbrt_miss_ms,
                           &cbrt_hits, &cbrt_misses, record_cbrt);

        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(t_loc1 - t_loc0).count();

        // Gather prime stats from all ranks (first warm run only)
        if (bench == 1)
            ksc_gather_prime_stats(&json_results,
                                   local_factors.count,
                                   (int64_t)(my_hi - my_lo + 1),
                                   rank, nranks);

        // Record cbrt hit/miss stats for warm runs
        if (rank == 0 && bench > 0)
        {
            json_results.cbrt_hit_ms.push_back(cbrt_hit_ms);
            json_results.cbrt_miss_ms.push_back(cbrt_miss_ms);
            json_results.cbrt_hits.push_back(cbrt_hits);
            json_results.cbrt_misses.push_back(cbrt_misses);
        }

        // Reduce to get max local cbrt time (the bottleneck rank) at rank 0
        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            accum_record(&loc_acc, max_local_dt);
            cur_max_loc = max_local_dt;
            if (bench > 0)
                json_results.local_cbrt_times.push_back(max_local_dt);
        }

        // Barrier so gather time measures only MPI communication, not waiting
        // for slower ranks to finish their local cbrt.
        MPI_Barrier(MPI_COMM_WORLD);

        // Gather partial cbrt products and combine at rank 0
        double iter_gat2 = 0.0, iter_com2 = 0.0;
        ksc_gather_and_combine_cbrt(result, local_cbrt, modulo, cbrt_bytes,
                                    rank, nranks, attempt,
                                    blind_table, BLIND_TABLE_SIZE,
                                    &iter_gat2, &iter_com2);
        mpz_clear(local_cbrt);
        if (rank == 0)
        {
            accum_record(&gat_acc, iter_gat2);
            iter_gat = iter_gat2;
            if (bench > 0)
                json_results.gather_times.push_back(iter_gat2);
            accum_record(&com_acc, iter_com2);
            iter_com = iter_com2;
            if (bench > 0)
                json_results.combine_times.push_back(iter_com2);
        }

        // Per-iteration reconstruction total — rank 0 only.
        // Must be accumulated AFTER both gather and combine are recorded.
        if (rank == 0)
        {
            double recon_dt = cur_max_loc + iter_gat + iter_com;
            accum_record(&recon_acc, recon_dt);
            if (bench > 0)
            {
                json_results.recon_times.push_back(recon_dt);
                if (!json_results.factor_times.empty())
                    json_results.total_times.push_back(
                        json_results.factor_times.back() + recon_dt);
            }
        }

        kfl_free(&local_factors);
    }

    // ---- Timing report (rank 0 only) ----
    if (rank == 0)
    {
        printf("\n");
        if (bench_count == 1)
        {
            printf("Factoring:         %.3f ms\n", fac_acc.sum);
            printf("Local cbrt (max):  %.3f ms\n", loc_acc.sum);
            printf("Gather:            %.3f ms\n", gat_acc.sum);
            printf("Combine:           %.3f ms\n", com_acc.sum);
            printf("Reconstruction:    %.3f ms\n", loc_acc.sum + gat_acc.sum + com_acc.sum);
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

        // Sequential powm baseline + speedup report
        ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                         bench_count, &recon_acc, &fac_acc, rank);
        // Verify
        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_cube_root(result, challenge_mod, modulo);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    free(prime_slice);
    mpz_clear(blinded_challenge);
    if (rank == 0)
        kf_blinding_table_free(blind_table, BLIND_TABLE_SIZE);

    // Save per-rank cbrt cache if new entries were computed this run.
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
    // MPI_THREAD_FUNNELED: only the main thread calls MPI functions.
    // OpenMP threads run inside trial_divide_* and compute_local_cbrt
    // but never touch MPI directly — safe with FUNNELED.
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
    const char *data_dir = "data/challenges/k32";
    const char *primes_file = "k32_primes.bin";
    const char *json_file = NULL;
    int bench_count = 1;
    int run_all = 0;
    int do_precompute = 0;
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
            else if (strcmp(m, "table") == 0)
                method = METHOD_TABLE;
            else if (strcmp(m, "powm") == 0)
                method = METHOD_POWM;
            else
            {
                if (rank == 0)
                    fprintf(stderr, "Unknown --method '%s' (use sieve, table, or powm)\n", m);
                MPI_Finalize();
                return 1;
            }
        }
        else if (strcmp(argv[argi], "--primes") == 0 && argi + 1 < argc)
        {
            primes_file = argv[++argi];
        }
        else if (strcmp(argv[argi], "--json") == 0 && argi + 1 < argc)
        {
            json_file = argv[++argi];
        }
        else if (strcmp(argv[argi], "--precompute") == 0)
        {
            do_precompute = 1;
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

    if (!run_all && !do_precompute && (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: k32_mpi <challenge> <modulo> [--method sieve|table] [--bench N]\n"
                    "       k32_mpi --all [--data DIR] [--method sieve|table] [--bench N]\n"
                    "       k32_mpi --precompute [--primes FILE]\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("=== k32_mpi ===\n");
        printf("MPI ranks:      %d\n", nranks);
#ifdef _OPENMP
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
#endif
        printf("Method:         %s\n",
               method == METHOD_SIEVE   ? "sieve"
               : method == METHOD_TABLE ? "table"
                                        : "powm");
        if (bench_count > 1)
            printf("Benchmark runs: %d\n", bench_count);
        if (run_all)
            printf("Mode:           --all\n");
        fflush(stdout);
    }

    // Small primes — every rank generates these independently (~0.1 ms, no Bcast)
    int sp_count;
    unsigned *sp = kf_generate_small_primes(&sp_count);

    // Partition [65537, 2^32] evenly across MPI ranks
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

    // Warm up OpenMP thread pool before any timed region
#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    // --precompute: rank 0 generates the prime table file; all ranks then exit
    if (do_precompute)
    {
        if (rank == 0)
        {
            int sp2;
            unsigned *sp2_arr = kf_generate_small_primes(&sp2);
            kf_generate_and_save_prime_table(primes_file, sp2, sp2_arr, "[rank 0] ");
            free(sp2_arr);
        }
        free(sp);
        MPI_Finalize();
        return 0;
    }

    // --method table: rank 0 generates the prime file if it doesn't yet exist,
    // then all ranks wait before each loading their own slice inside solve_one.
    if (method == METHOD_TABLE)
    {
        // Ensure data/k32_primes.bin exists; generates if absent (rank 0 only)
        primes_file = ksc_ensure_k32_primes(sp_count, sp, rank);
    }

    if (run_all)
    {
        // Run all four standard challenge sizes in one MPI execution.
        // Avoids multiple srun calls in the SLURM script (causes reinit hangs).
        const char *sizes[] = {"1024", "2048", "5000", "7000"};
        for (int s = 0; s < 4; s++)
        {
            char cf[512], mf[512];
            snprintf(cf, sizeof(cf), "%s/challenge_32_%s.txt", data_dir, sizes[s]);
            snprintf(mf, sizeof(mf), "%s/modulo_32_%s.txt", data_dir, sizes[s]);
            solve_one(rank, nranks, cf, mf, bench_count,
                      sp_count, sp, my_lo, my_hi, method, primes_file, json_file);
        }
    }
    else
    {
        solve_one(rank, nranks, challenge_file, modulo_file, bench_count,
                  sp_count, sp, my_lo, my_hi, method, primes_file, json_file);
    }

    free(sp);
    MPI_Finalize();
    return 0;
}