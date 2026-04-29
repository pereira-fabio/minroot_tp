// =============================================================================
// k16_mpi.cpp — MPI+OpenMP hybrid cube root solver for 16-bit prime factors
//
// Architecture mirrors k32_mpi.cpp but targets the range [2, 65536].
// With only 6542 primes in total, each MPI rank handles a small slice
// (~409 primes with 16 ranks). This makes k16_mpi useful for studying how
// MPI communication overhead compares to compute at small prime ranges.
//
// Key differences from k32_mpi:
//   - Prime range: [2, 65536] (6542 primes) vs [65537, 2^32] (203M primes)
//   - No prime file needed: all primes generated inline via Eratosthenes
//   - No blinding retry loop: k16 challenges are constructed smooth by design
//   - Fail-hard if challenge is not fully factorable (no Pollard fallback)
//   - cbrt table stored as dense array (kf_*_k16 functions), not sparse cache
//
// Methods:
//   sieve  — generate primes inline via Eratosthenes [default]
//   table  — identical to sieve at k16 scale (both fit in L1 cache);
//            kept for interface uniformity with k32_mpi
//   powm   — sequential mpz_powm baseline (rank 0 only)
//
// Usage:
//   k16_mpi [challenge] [modulo] [--method sieve|table|powm]
//           [--bench N] [--json FILE]
// =============================================================================

#include "k_common.h"
#include "k_factorization.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef enum
{
    METHOD_SIEVE,
    METHOD_TABLE,
    METHOD_POWM,
} run_method_t;

// ---------------------------------------------------------------------------
// Distributed local cbrt reconstruction (same logic as k32_mpi)
// Each rank computes cbrt for its own factors using the dense k16 cbrt table,
// then sends one partial product to rank 0.
// ---------------------------------------------------------------------------

static void compute_local_cbrt_k16(
    mpz_t result,
    const kfactor_list_t *factors,
    const mpz_t *cbrt_table,         // cbrt_table[i] = cbrt(primes[i]) mod M
    const unsigned long *all_primes, // full sorted prime array [0..prime_count)
    int prime_count,
    const mpz_t modulo,
    double *hit_ms_out, double *miss_ms_out,
    int *hits_out, int *misses_out,
    bool record)
{
    int n = factors->count;
    if (n == 0)
    {
        mpz_set_ui(result, 1);
        return;
    }

    mpz_set_ui(result, 1);
    for (int i = 0; i < n; i++)
    {
        unsigned long p = mpz_get_ui(factors->items[i].prime);
        // Binary-search for this prime in the sorted all_primes array
        // to get its index into cbrt_table.
        int lo = 0, hi = prime_count - 1, idx = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (all_primes[mid] == p)
            {
                idx = mid;
                break;
            }
            else if (all_primes[mid] < p)
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        if (idx < 0)
        {
            fprintf(stderr, "[ERROR] prime %lu not in k16 table\n", p);
            continue;
        }

        auto _t0 = std::chrono::high_resolution_clock::now();
        mpz_t term;
        mpz_init(term);
        if (factors->items[i].exponent == 1)
            mpz_set(term, cbrt_table[idx]);
        else
            mpz_powm_ui(term, cbrt_table[idx],
                        (unsigned long)factors->items[i].exponent, modulo);
        mpz_mul(result, result, term);
        mpz_mod(result, result, modulo);
        mpz_clear(term);
        auto _t1 = std::chrono::high_resolution_clock::now();

        // All cbrt_table entries are precomputed — always a "hit"
        if (record)
        {
            if (hit_ms_out)
                *hit_ms_out += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
            if (hits_out)
                (*hits_out)++;
        }
    }
    // misses always 0 for k16 (dense precomputed table, no fallback)
    if (record && misses_out)
        *misses_out = 0;
    if (record && miss_ms_out)
        *miss_ms_out = 0.0;
}

#define print_stat(lbl, sum, sumsq, n) print_timing_stat_moments(lbl, sum, sumsq, n, 30, true)

// ---------------------------------------------------------------------------
// solve_one
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count,
    run_method_t method,
    const char *json_output_file = nullptr)
{
    if (rank == 0)
    {
        printf("\n=== %s ===\n", challenge_file);
        fflush(stdout);
    }

    // ---- Read and broadcast challenge + modulo ----
    int read_ok = 1;
    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    size_t challenge_nbytes = 0, modulo_nbytes = 0;
    unsigned char *challenge_bytes = NULL, *modulo_bytes = NULL;

    if (rank == 0)
    {
        FILE *fc = fopen(challenge_file, "rb");
        FILE *fm = fopen(modulo_file, "rb");
        if (!fc || !fm)
        {
            fprintf(stderr, "Error: cannot open '%s' or '%s'\n",
                    challenge_file, modulo_file);
            if (fc)
                fclose(fc);
            if (fm)
                fclose(fm);
            read_ok = 0;
        }
        else
        {
            fclose(fc);
            fclose(fm);
            read_mpz_from_file(challenge, challenge_file);
            read_mpz_from_file(modulo, modulo_file);
            challenge_bytes = (unsigned char *)mpz_export(
                NULL, &challenge_nbytes, 1, 1, 0, 0, challenge);
            modulo_bytes = (unsigned char *)mpz_export(
                NULL, &modulo_nbytes, 1, 1, 0, 0, modulo);
        }
    }

    MPI_Bcast(&read_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!read_ok)
    {
        mpz_clears(challenge, modulo, result, NULL);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 0;
    }

    uint64_t cn64 = (uint64_t)challenge_nbytes, mn64 = (uint64_t)modulo_nbytes;
    MPI_Bcast(&cn64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mn64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    challenge_nbytes = (size_t)cn64;
    modulo_nbytes = (size_t)mn64;

    if (rank != 0)
    {
        challenge_bytes = (unsigned char *)malloc(challenge_nbytes);
        modulo_bytes = (unsigned char *)malloc(modulo_nbytes);
    }
    MPI_Bcast(challenge_bytes, (int)challenge_nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(modulo_bytes, (int)modulo_nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        mpz_import(challenge, challenge_nbytes, 1, 1, 0, 0, challenge_bytes);
        mpz_import(modulo, modulo_nbytes, 1, 1, 0, 0, modulo_bytes);
    }
    free(challenge_bytes);
    free(modulo_bytes);

    if (rank == 0)
    {
        printf("Challenge: %zu bits\nModulo:    %zu bits\n",
               mpz_sizeinbase(challenge, 2), mpz_sizeinbase(modulo, 2));
        printf("Ranks: %d  Threads/rank: %d  Total workers: %d\n",
               nranks,
#ifdef _OPENMP
               omp_get_max_threads(), nranks * omp_get_max_threads());
#else
               1, nranks);
#endif
        fflush(stdout);
    }

    // ---- Generate all k16 primes [2, 65536] on every rank ----
    // 6542 primes × 8 bytes = 52KB — fits in L1 cache, generated in ~0.01ms.
    int prime_count = 0;
    unsigned long *all_primes = generate_primes_up_to(65535UL, &prime_count);

    // ---- Partition prime array across ranks ----
    // Each rank gets a contiguous slice of the sorted prime array.
    int base_per_rank = prime_count / nranks;
    int remainder = prime_count % nranks;
    int my_start = rank * base_per_rank + (rank < remainder ? rank : remainder);
    int my_count = base_per_rank + (rank < remainder ? 1 : 0);

    if (rank == 0)
        printf("Primes per rank: ~%d (total %d)\n\n", base_per_rank, prime_count);

    // ---- Build cbrt table for MY slice ----
    // cbrt_table_full is built by rank 0 from the modulo, broadcast to all,
    // so every rank has the complete table for fast lookup.
    // This is feasible because the k16 table is tiny (6542 entries).
    //
    // Derive cache file path from modulo filename.
    char cbrt_path[520];
    {
        char base[512];
        strncpy(base, modulo_file ? modulo_file : "modulo", sizeof(base) - 1);
        base[sizeof(base) - 1] = '\0';
        char *dot = strrchr(base, '.');
        if (dot)
            *dot = '\0';
        snprintf(cbrt_path, sizeof(cbrt_path), "%s.cbrt", base);
    }

    // Allocate cbrt_table for ALL primes (every rank needs the full table for lookup)
    mpz_t *cbrt_table = (mpz_t *)malloc((size_t)prime_count * sizeof(mpz_t));
    for (int i = 0; i < prime_count; i++)
        mpz_init(cbrt_table[i]);

    if (!kf_load_cbrt_table_k16(cbrt_table, prime_count, cbrt_path))
    {
        if (rank == 0)
        {
            kf_precompute_cbrt_table_k16(cbrt_table, all_primes, prime_count, modulo);
            kf_save_cbrt_table_k16(cbrt_table, prime_count, cbrt_path);
            printf("cbrt table: computed and saved to %s\n", cbrt_path);
        }
        // Broadcast each entry to all ranks (or serialize/deserialize)
        // Easier: rank 0 exports as raw bytes, broadcasts, all import.
        // cbrt entries are GMP integers — export as decimal strings for simplicity.
        // For 6542 entries at ~2100 chars each, total ~14MB — acceptable once.
        for (int i = 0; i < prime_count; i++)
        {
            // Each rank except 0 needs to receive cbrt_table[i]
            // Use mpz_sizeinbase + mpz_export/import pattern
            size_t nbytes = 0;
            unsigned char *buf = NULL;
            if (rank == 0)
            {
                buf = (unsigned char *)mpz_export(NULL, &nbytes, 1, 1, 0, 0, cbrt_table[i]);
            }
            uint64_t nb64 = (uint64_t)nbytes;
            MPI_Bcast(&nb64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            nbytes = (size_t)nb64;
            if (rank != 0)
                buf = (unsigned char *)malloc(nbytes ? nbytes : 1);
            MPI_Bcast(buf, (int)nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            if (rank != 0 && nbytes > 0)
                mpz_import(cbrt_table[i], nbytes, 1, 1, 0, 0, buf);
            free(buf);
        }
    }
    else if (rank == 0)
        printf("cbrt table: loaded from %s\n", cbrt_path);

    // All ranks now have the full cbrt_table.

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    // ---- Timing accumulators ----
    bench_accum_t fac_acc = {0, 0};
    bench_accum_t loc_acc = {0, 0};
    bench_accum_t gat_acc = {0, 0};
    bench_accum_t com_acc = {0, 0};
    bench_accum_t recon_acc = {0, 0};
    double cur_max_loc = 0.0;

    // ---- JSON results ----
    int n_threads_json = 1;
#ifdef _OPENMP
    n_threads_json = omp_get_max_threads();
#endif
    benchmark_results_t json_results;
    json_results.k_size = 16;
    json_results.challenge_bits = (rank == 0) ? mpz_sizeinbase(challenge, 2) : 0;
    json_results.method_name = (method == METHOD_POWM)    ? "powm"
                               : (method == METHOD_TABLE) ? "table"
                                                          : "sieve";
    json_results.num_threads = n_threads_json;
    json_results.mpi_ranks = nranks;
    json_results.bench_count = bench_count;
    json_results.powm_baseline_ms = 0.0;

    // ---- 2 warmup runs ----
    if (method != METHOD_POWM)
    {
        for (int wu = 0; wu < 2; wu++)
        {
            // Each rank trial-divides its slice of the challenge
            kfactor_list_t wu_factors;
            kfl_init(&wu_factors);
            for (int i = my_start; i < my_start + my_count; i++)
            {
                if (mpz_tdiv_ui(challenge, all_primes[i]) == 0)
                {
                    int e = count_exponent(challenge, all_primes[i]);
                    if (e > 0)
                        kfl_append_ul(&wu_factors, all_primes[i], e);
                }
            }
            mpz_t wu_cbrt;
            mpz_init(wu_cbrt);
            auto wu_t0 = std::chrono::high_resolution_clock::now();
            compute_local_cbrt_k16(wu_cbrt, &wu_factors, cbrt_table,
                                   all_primes, prime_count, modulo,
                                   nullptr, nullptr, nullptr, nullptr, false);
            auto wu_t1 = std::chrono::high_resolution_clock::now();
            if (rank == 0)
                json_results.warmup_total_ms.push_back(
                    std::chrono::duration<double, std::milli>(wu_t1 - wu_t0).count());
            mpz_clear(wu_cbrt);
            kfl_free(&wu_factors);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0)
            printf("Warmup complete (2 runs)\n\n");
    }

    // ---- Benchmark loop ----
    for (int bench = 0; bench < bench_count; bench++)
    {
        kfactor_list_t local_factors;
        kfl_init(&local_factors);

        // ---- FACTORING ----
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        // Per-thread counters for load balance measurement (first warm run only)
        int nT = 1;
#ifdef _OPENMP
        nT = omp_get_max_threads();
#endif
        bool do_thread_stats = (bench == 1 && rank == 0);
        std::vector<int64_t> thr_tested(do_thread_stats ? nT : 0, 0);
        std::vector<int> thr_found(do_thread_stats ? nT : 0, 0);

        // Thread-private factor lists
        kfactor_list_t *tf = (kfactor_list_t *)malloc((size_t)nT * sizeof(kfactor_list_t));
        for (int t = 0; t < nT; t++)
            kfl_init(&tf[t]);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = my_start; i < my_start + my_count; i++)
        {
            int t = 0;
#ifdef _OPENMP
            t = omp_get_thread_num();
#endif
            if (do_thread_stats)
                thr_tested[t]++;
            unsigned long p = all_primes[i];
            if (mpz_tdiv_ui(challenge, p) != 0)
                continue;
            int e = count_exponent(challenge, p);
            if (e > 0)
            {
                kfl_append_ul(&tf[t], p, e);
                if (do_thread_stats)
                    thr_found[t]++;
            }
        }
        for (int t = 0; t < nT; t++)
        {
            for (int j = 0; j < tf[t].count; j++)
                kfl_append_mpz(&local_factors, tf[t].items[j].prime, tf[t].items[j].exponent);
            kfl_free(&tf[t]);
        }
        free(tf);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(t_fac1 - t_fac0).count();
            accum_record(&fac_acc, dt);
            if (bench > 0)
                json_results.factor_times.push_back(dt);
            if (bench == 0)
                printf("Factors found by rank 0: %d (after factoring [%lu, %lu])\n",
                       local_factors.count, all_primes[my_start],
                       all_primes[my_start + my_count - 1]);
        }

        // Record thread stats (first warm run)
        if (do_thread_stats)
        {
            json_results.thread_primes_tested.insert(
                json_results.thread_primes_tested.end(),
                thr_tested.begin(), thr_tested.end());
            json_results.thread_primes_found.insert(
                json_results.thread_primes_found.end(),
                thr_found.begin(), thr_found.end());
        }

        // Gather primes_found per rank (first warm run)
        if (bench == 1)
        {
            int lf = local_factors.count;
            if (rank == 0)
            {
                std::vector<int> all_found(nranks);
                MPI_Gather(&lf, 1, MPI_INT, all_found.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
                for (int r = 0; r < nranks; r++)
                    json_results.primes_found_per_rank.push_back(all_found[r]);
                // Primes tested per rank (static)
                for (int r = 0; r < nranks; r++)
                {
                    int r_start = r * base_per_rank + (r < remainder ? r : remainder);
                    int r_count = base_per_rank + (r < remainder ? 1 : 0);
                    json_results.primes_tested_per_rank.push_back((int64_t)r_count);
                    (void)r_start;
                }
            }
            else
            {
                MPI_Gather(&lf, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }

        // ---- Completeness check ----
        // Gather local factor products, verify product == challenge
        mpz_t local_prod;
        mpz_init_set_ui(local_prod, 1);
        for (int i = 0; i < local_factors.count; i++)
        {
            unsigned long p = mpz_get_ui(local_factors.items[i].prime);
            for (int ex = 0; ex < local_factors.items[i].exponent; ex++)
                mpz_mul_ui(local_prod, local_prod, p);
        }
        unsigned char *prod_buf = (unsigned char *)calloc(cbrt_bytes, 1);
        {
            size_t wr = 0;
            unsigned char *tmp = (unsigned char *)mpz_export(NULL, &wr, 1, 1, 0, 0, local_prod);
            if (tmp && wr > 0 && wr <= cbrt_bytes)
            {
                memcpy(prod_buf + cbrt_bytes - wr, tmp, wr);
                free(tmp);
            }
        }
        mpz_clear(local_prod);

        unsigned char *all_prods = NULL;
        if (rank == 0)
            all_prods = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);
        MPI_Gather(prod_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                   all_prods, (int)cbrt_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        free(prod_buf);

        if (rank == 0)
        {
            mpz_t combined;
            mpz_init_set_ui(combined, 1);
            mpz_t part;
            mpz_init(part);
            for (int rk = 0; rk < nranks; rk++)
            {
                mpz_import(part, cbrt_bytes, 1, 1, 0, 0, all_prods + (size_t)rk * cbrt_bytes);
                mpz_mul(combined, combined, part);
            }
            mpz_clear(part);
            free(all_prods);

            mpz_t ch_mod;
            mpz_init(ch_mod);
            mpz_mod(ch_mod, challenge, modulo);
            if (mpz_cmp(combined, ch_mod) != 0)
            {
                fprintf(stderr,
                        "[ERROR] Challenge not fully factorable by primes <= 65536.\n"
                        "        Ensure challenge was generated with bitbound=16.\n");
                mpz_clears(combined, ch_mod, NULL);
                kfl_free(&local_factors);
                // Abort gracefully
                int abort_flag = 1;
                MPI_Bcast(&abort_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 0;
            }
            mpz_clears(combined, ch_mod, NULL);
        }

        // ---- DISTRIBUTED RECONSTRUCTION ----
        auto t_loc0 = std::chrono::high_resolution_clock::now();

        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double cbrt_hit_ms = 0.0, cbrt_miss_ms = 0.0;
        int cbrt_hits = 0, cbrt_misses = 0;
        compute_local_cbrt_k16(local_cbrt, &local_factors, cbrt_table,
                               all_primes, prime_count, modulo,
                               &cbrt_hit_ms, &cbrt_miss_ms,
                               &cbrt_hits, &cbrt_misses, bench > 0);

        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(t_loc1 - t_loc0).count();

        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            accum_record(&loc_acc, max_local_dt);
            cur_max_loc = max_local_dt;
            if (bench > 0)
            {
                json_results.local_cbrt_times.push_back(max_local_dt);
                json_results.cbrt_hit_ms.push_back(cbrt_hit_ms);
                json_results.cbrt_miss_ms.push_back(cbrt_miss_ms);
                json_results.cbrt_hits.push_back(cbrt_hits);
                json_results.cbrt_misses.push_back(cbrt_misses);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Export and gather partial cbrt
        unsigned char *local_buf = (unsigned char *)calloc(cbrt_bytes, 1);
        {
            size_t wr = 0;
            unsigned char *tmp = (unsigned char *)mpz_export(NULL, &wr, 1, 1, 0, 0, local_cbrt);
            if (tmp && wr > 0 && wr <= cbrt_bytes)
            {
                memcpy(local_buf + cbrt_bytes - wr, tmp, wr);
                free(tmp);
            }
        }
        mpz_clear(local_cbrt);

        unsigned char *all_bufs = NULL;
        if (rank == 0)
            all_bufs = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);

        auto t_gat0 = std::chrono::high_resolution_clock::now();
        MPI_Gather(local_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                   all_bufs, (int)cbrt_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        auto t_gat1 = std::chrono::high_resolution_clock::now();
        free(local_buf);

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(t_gat1 - t_gat0).count();
            accum_record(&gat_acc, dt);
            double iter_gat = dt;
            if (bench > 0)
                json_results.gather_times.push_back(dt);

            auto t_com0 = std::chrono::high_resolution_clock::now();
            mpz_set_ui(result, 1);
            mpz_t partial;
            mpz_init(partial);
            for (int rk = 0; rk < nranks; rk++)
            {
                mpz_import(partial, cbrt_bytes, 1, 1, 0, 0,
                           all_bufs + (size_t)rk * cbrt_bytes);
                mpz_mul(result, result, partial);
                mpz_mod(result, result, modulo);
            }
            mpz_clear(partial);
            free(all_bufs);
            auto t_com1 = std::chrono::high_resolution_clock::now();
            double com_dt = std::chrono::duration<double, std::milli>(t_com1 - t_com0).count();
            accum_record(&com_acc, com_dt);
            if (bench > 0)
                json_results.combine_times.push_back(com_dt);

            double recon_dt = cur_max_loc + iter_gat + com_dt;
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
        printf("\nResults over %d runs:\n", bench_count);
        print_stat("Factoring:", fac_acc.sum, fac_acc.sumsq, bench_count);
        print_stat("Local cbrt (max):", loc_acc.sum, loc_acc.sumsq, bench_count);
        print_stat("Gather:", gat_acc.sum, gat_acc.sumsq, bench_count);
        print_stat("Combine (rank 0):", com_acc.sum, com_acc.sumsq, bench_count);
        print_stat("Reconstruction total:", recon_acc.sum, recon_acc.sumsq, bench_count);
        printf("    %-30s avg=%9.3f ms\n", "Total:",
               (fac_acc.sum + recon_acc.sum) / bench_count);

        // Sequential powm baseline (all methods)
        printf("\n--- Sequential powm baseline (rank 0) ---\n");
        mpz_t powm_result;
        mpz_init(powm_result);
        double powm_avg = run_powm_baseline(powm_result, challenge, modulo,
                                            bench_count, 30);
        json_results.powm_baseline_ms = powm_avg;
        mpz_clear(powm_result);

        double recon_avg = accum_mean(&recon_acc, bench_count);
        double total_avg = (fac_acc.sum + recon_acc.sum) / bench_count;
        printf("\n--- Speedup vs sequential powm ---\n");
        printf("  powm vs reconstruction only:  %.2fx\n",
               recon_avg > 0 ? powm_avg / recon_avg : 0.0);
        printf("  powm vs total (factor+recon): %.2fx\n",
               total_avg > 0 ? powm_avg / total_avg : 0.0);

        // Verify
        mpz_t ch_mod;
        mpz_init(ch_mod);
        mpz_mod(ch_mod, challenge, modulo);
        verify_cube_root(result, ch_mod, modulo);
        mpz_clear(ch_mod);
        fflush(stdout);
    }

    // ---- Cleanup ----
    for (int i = 0; i < prime_count; i++)
        mpz_clear(cbrt_table[i]);
    free(cbrt_table);
    free(all_primes);
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

    const char *challenge_file = "data/challenges/k16/challenge_2_16_7000.txt";
    const char *modulo_file = "data/challenges/k16/modulo_16_7000.txt";
    const char *json_file = NULL;
    int bench_count = 1;
    run_method_t method = METHOD_SIEVE;

    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--bench") == 0 && argi + 1 < argc)
            bench_count = atoi(argv[++argi]);
        else if (strcmp(argv[argi], "--json") == 0 && argi + 1 < argc)
            json_file = argv[++argi];
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
        else if (argv[argi][0] != '-')
        {
            if (!challenge_file || strcmp(challenge_file, "data/challenges/k16/challenge_2_16_7000.txt") == 0)
                challenge_file = argv[argi];
            else
                modulo_file = argv[argi];
        }
        else
        {
            if (rank == 0)
                fprintf(stderr, "Unknown option '%s'\n", argv[argi]);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0)
    {
        printf("=== k16_mpi ===\n");
        printf("MPI ranks:      %d\n", nranks);
#ifdef _OPENMP
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
#endif
        printf("Method:         %s\n",
               method == METHOD_POWM    ? "powm"
               : method == METHOD_TABLE ? "table"
                                        : "sieve");
        printf("Benchmark runs: %d\n\n", bench_count);
        fflush(stdout);
    }

    // Warm up OMP thread pool
#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    solve_one(rank, nranks, challenge_file, modulo_file,
              bench_count, method, json_file);

    MPI_Finalize();
    return 0;
}