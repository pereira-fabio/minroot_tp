// =============================================================================
// k_alg3_mpi.cpp — Algorithm 3 solver: almost-smooth cube root recovery
//
// Extends Algorithm 2 (k32_mpi) by tolerating one medium prime factor
// q0 ∈ (2^32, 2^65] in the blinded challenge, per Biryukov et al. CRYPTO 2024.
//
// Key difference from k32_mpi:
//   OLD (k32): completeness = (combined_small_prods == blinded)
//   NEW (alg3): completeness = (combined_small_prods == blinded)         [B-smooth]
//               OR (blinded/combined_small_prods is prime ≤ 65 bits)    [almost-smooth]
//
//   On success (almost-smooth): rank 0 folds cbrt(q0) into its local partial
//   product before the gather — ksc_gather_and_combine_cbrt is unchanged.
//
// Success probability: ~2^(-18) per attempt vs ~2^(-24.9) for k32 (64× more).
//
// Isolation from Algorithm 2:
//   Challenges:  data/challenges/alg3/          (separate from k32/k35)
//   cbrt caches: data/cbrt/<stem>.cbrt_alg3.<rank>  (different suffix)
//   Results:     data/results/alg3_mpi_N*/
//   Prime table: reuses data/k32_primes.bin read-only (sieve method only)
//
// Communication pattern: identical to k32_mpi.
//
// Usage:
//   alg3_mpi [challenge] [modulo] [--method sieve|powm] [--bench N]
//   alg3_mpi --all [--data DIR] [--bench N]
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

#define print_stat(lbl, sum, sumsq, n) print_timing_stat_moments(lbl, sum, sumsq, n, 30, true)

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
        fflush(stdout);
    }

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    mpz_t phi, e_inv, three_mpz;
    mpz_inits(phi, e_inv, three_mpz, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three_mpz, 3);
    mpz_invert(e_inv, three_mpz, phi);
    mpz_clears(phi, three_mpz, NULL);

    // ALG3: cache suffix .cbrt_alg3.<rank> — isolated from k32 caches.
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
                 "%s/%s.cbrt_alg3.%d", KSC_CBRT_DIR, stem, rank);
    }

    cbrt_cache_t local_cbrt_cache;
    kf_cbrt_cache_init(&local_cbrt_cache);
    bool cbrt_cache_dirty = false;

    // Precompute cbrt cache for the smooth part of the challenge.
    // ksc_ensure_k32_cbrt runs sieve on the challenge; q0 won't be found
    // (it's > 2^32) but that's correct — cbrt(q0) is computed on-the-fly.
    ksc_ensure_k32_cbrt(
        &local_cbrt_cache, &cbrt_cache_dirty, cbrt_cache_path,
        challenge, modulo, e_inv,
        my_lo, my_hi, sp_count, sp,
        NULL, 0,   // ALG3: sieve only, no prime table slice
        rank, nranks, 0);

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

    // cofactor is set by ksc_alg3_broadcast_cofactor each iteration:
    //   cofactor == 1  → B-smooth
    //   cofactor == q0 → almost-smooth (rank 0 will fold cbrt(q0) in)
    mpz_t cofactor;
    mpz_init_set_ui(cofactor, 1);

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
    ksc_init_json_results(&json_results, 32, bench_count, "sieve",
                          nranks, n_threads_json, challenge, rank);

    // ---- 2 warmup runs ----
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

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        int attempt = 0;
        const int MAX_ATTEMPTS = 256;
        size_t blinded_nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
        unsigned char *blinded_buf = (unsigned char *)malloc(blinded_nbytes);

        for (attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
        {
            // -- Rank 0 blinds and broadcasts --
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
                        mpz_mul(blinded_challenge, challenge,
                                blind_table[idx].r3);
                        mpz_mod(blinded_challenge, blinded_challenge, modulo);
                    }
                    else
                    {
                        fprintf(stderr,
                                "[rank 0] WARNING: blinding table exhausted "
                                "at attempt %d\n",
                                attempt);
                        mpz_set(blinded_challenge, challenge);
                    }
                }
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
            MPI_Bcast(blinded_buf, (int)blinded_nbytes, MPI_UNSIGNED_CHAR,
                      0, MPI_COMM_WORLD);
            if (rank != 0)
                mpz_import(blinded_challenge, blinded_nbytes, 1, 1, 0, 0,
                           blinded_buf);

            // -- All ranks factor their sieve slice --
            kfl_free(&local_factors);
            kfl_init(&local_factors);
            kf_trial_divide_sieve(blinded_challenge, my_lo, my_hi,
                                  sp_count, sp, &local_factors);

            // -- ALG3: gather + cofactor test (replaces k32's equality check) --
            //
            // Each rank computes its local product and passes it to the helper.
            // ksc_alg3_broadcast_cofactor gathers all products, computes
            // cofactor = blinded / combined, classifies, and broadcasts.
            mpz_t local_prod;
            mpz_init_set_ui(local_prod, 1);
            for (int i = 0; i < local_factors.count; i++)
            {
                unsigned long p = mpz_get_ui(local_factors.items[i].prime);
                for (int ex = 0; ex < local_factors.items[i].exponent; ex++)
                    mpz_mul_ui(local_prod, local_prod, p);
            }

            mpz_set_ui(cofactor, 1); // reset before each attempt
            int complete = ksc_alg3_broadcast_cofactor(
                cofactor, blinded_challenge, local_prod, cbrt_bytes,
                rank, nranks);
            mpz_clear(local_prod);

            if (rank == 0 && bench == 0)
            {
                int is_smooth = (complete && mpz_cmp_ui(cofactor, 1) == 0);
                printf("Blinding attempt %d: %s\n", attempt + 1,
                       complete ? (is_smooth ? "smooth — proceeding"
                                             : "almost-smooth — proceeding")
                                : "not smooth — retrying");
                fflush(stdout);
            }

            if (complete)
                break;
        } // end blinding retry loop

        free(blinded_buf);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(
                            t_fac1 - t_fac0).count();
            accum_record(&fac_acc, dt);
            if (bench > 0)
                json_results.factor_times.push_back(dt);
            if (bench == 0)
            {
                printf("Factors found by rank 0: %d distinct primes "
                       "(after %d attempt(s))\n",
                       local_factors.count, attempt + 1);
                fflush(stdout);
            }
            if (bench > 0)
                json_results.blinding_attempts.push_back(attempt + 1);
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

        // ALG3: rank 0 folds cbrt(q0) into its partial product before the
        // gather so ksc_gather_and_combine_cbrt needs no changes.
        if (rank == 0 && mpz_cmp_ui(cofactor, 1) != 0)
        {
            mpz_t cbrt_q0;
            mpz_init(cbrt_q0);
            mpz_powm(cbrt_q0, cofactor, e_inv, modulo);
            mpz_mul(local_cbrt, local_cbrt, cbrt_q0);
            mpz_mod(local_cbrt, local_cbrt, modulo);
            mpz_clear(cbrt_q0);
        }

        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(t_loc1 - t_loc0).count();

        if (bench == 1)
            ksc_gather_prime_stats(&json_results,
                                   local_factors.count,
                                   (int64_t)(my_hi - my_lo + 1),
                                   rank, nranks);

        if (rank == 0 && bench > 0)
        {
            json_results.cbrt_hit_ms.push_back(cbrt_hit_ms);
            json_results.cbrt_miss_ms.push_back(cbrt_miss_ms);
            json_results.cbrt_hits.push_back(cbrt_hits);
            json_results.cbrt_misses.push_back(cbrt_misses);
        }

        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            accum_record(&loc_acc, max_local_dt);
            cur_max_loc = max_local_dt;
            if (bench > 0)
                json_results.local_cbrt_times.push_back(max_local_dt);
        }

        MPI_Barrier(MPI_COMM_WORLD);

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

        ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                         bench_count, &recon_acc, &fac_acc, rank);
        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_cube_root(result, challenge_mod, modulo);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    mpz_clear(blinded_challenge);
    mpz_clear(cofactor);
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
    const char *data_dir = "data/challenges/alg3"; // ALG3: own challenge directory
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
                            "Unknown --method '%s' (alg3_mpi supports sieve and powm)\n", m);
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
                    "Usage: alg3_mpi <challenge> <modulo> [--method sieve|powm] [--bench N]\n"
                    "       alg3_mpi --all [--data DIR] [--bench N]\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("=== alg3_mpi (Algorithm 3 — almost-smooth) ===\n");
        printf("MPI ranks:      %d\n", nranks);
#ifdef _OPENMP
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
#endif
        printf("Method:         %s\n",
               method == METHOD_SIEVE ? "sieve" : "powm");
        if (bench_count > 1)
            printf("Benchmark runs: %d\n", bench_count);
        if (run_all)
            printf("Mode:           --all\n");
        fflush(stdout);
    }

    int sp_count;
    unsigned *sp = kf_generate_small_primes(&sp_count);

    // Partition sieve range [65537, 2^32] — same as k32
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

#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    if (run_all)
    {
        // ALG3 challenge naming: challenge_0_alg3_<bits>.txt / modulo_alg3_<bits>.txt
        const char *sizes[] = {"1024", "2048", "4096", "7000"};
        for (int s = 0; s < 4; s++)
        {
            char cf[512], mf[512];
            snprintf(cf, sizeof(cf), "%s/challenge_0_alg3_%s.txt", data_dir, sizes[s]);
            snprintf(mf, sizeof(mf), "%s/modulo_alg3_%s.txt", data_dir, sizes[s]);
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
