// =============================================================================
// k16_mpi.cpp — MPI+OpenMP hybrid cube root solver for 16-bit prime factors
//
// Partitions the 6542 primes ≤ 65536 across MPI ranks. Each rank factors its
// slice of the challenge, computes local cbrt products, and sends one partial
// product to rank 0 for final combination.
//
// Methods:  sieve | table | powm
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
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef enum
{
    METHOD_SIEVE,
    METHOD_TABLE,
    METHOD_POWM
} run_method_t;

// ---------------------------------------------------------------------------
// Local cbrt reconstruction using the dense k16 cbrt table
// ---------------------------------------------------------------------------

static void compute_local_cbrt_k16(
    mpz_t result,
    const kfactor_list_t *factors,
    const mpz_t *cbrt_table,
    const unsigned long *all_primes,
    int prime_count,
    const mpz_t modulo,
    double *hit_ms, double *miss_ms,
    int *hits, int *misses,
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
        // Binary search for prime index
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
        auto t0 = std::chrono::high_resolution_clock::now();
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
        auto t1 = std::chrono::high_resolution_clock::now();
        // k16 always hits (dense precomputed table)
        if (record)
        {
            if (hit_ms)
                *hit_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (hits)
                (*hits)++;
        }
    }
    if (record && misses)
        *misses = 0;
    if (record && miss_ms)
        *miss_ms = 0.0;
}

// ---------------------------------------------------------------------------
// Factor the challenge with this rank's prime slice (OMP parallel)
// ---------------------------------------------------------------------------

static void factor_slice(
    kfactor_list_t *result,
    const mpz_t challenge,
    const unsigned long *all_primes,
    int my_start, int my_count,
    std::vector<int64_t> *thr_tested,
    std::vector<int> *thr_found)
{
    int nT = 1;
#ifdef _OPENMP
    nT = omp_get_max_threads();
#endif
    kfactor_list_t *tf = (kfactor_list_t *)malloc((size_t)nT * sizeof(kfactor_list_t));
    for (int t = 0; t < nT; t++)
        kfl_init(&tf[t]);

    bool do_stats = (thr_tested != nullptr);
    if (do_stats)
    {
        thr_tested->assign(nT, 0);
        thr_found->assign(nT, 0);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = my_start; i < my_start + my_count; i++)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        if (do_stats)
            (*thr_tested)[t]++;
        unsigned long p = all_primes[i];
        if (mpz_tdiv_ui(challenge, p) != 0)
            continue;
        int e = count_exponent(challenge, p);
        if (e > 0)
        {
            kfl_append_ul(&tf[t], p, e);
            if (do_stats)
                (*thr_found)[t]++;
        }
    }
    for (int t = 0; t < nT; t++)
    {
        for (int j = 0; j < tf[t].count; j++)
            kfl_append_mpz(result, tf[t].items[j].prime, tf[t].items[j].exponent);
        kfl_free(&tf[t]);
    }
    free(tf);
}

// ---------------------------------------------------------------------------
// solve_one
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count, run_method_t method,
    const char *json_output_file)
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
        printf("Challenge: %zu bits  Modulo: %zu bits\n",
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

    // ── Primes ──────────────────────────────────────────────────────────────
    int prime_count = 0;
    unsigned long *all_primes = ksc_ensure_k16_primes(&prime_count);

    // Partition prime array across ranks
    int base_per_rank = prime_count / nranks;
    int remainder = prime_count % nranks;
    int my_start = rank * base_per_rank + (rank < remainder ? rank : remainder);
    int my_count = base_per_rank + (rank < remainder ? 1 : 0);
    int64_t my_tested = (int64_t)my_count;

    if (rank == 0)
        printf("Primes: %d total, ~%d per rank\n\n", prime_count, base_per_rank);

    // ── cbrt table ──────────────────────────────────────────────────────────
    mpz_t *cbrt_table = (mpz_t *)malloc((size_t)prime_count * sizeof(mpz_t));
    for (int i = 0; i < prime_count; i++)
        mpz_init(cbrt_table[i]);
    ksc_ensure_k16_cbrt(cbrt_table, all_primes, prime_count,
                        modulo, modulo_file, rank);

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    // ── JSON + accumulators ─────────────────────────────────────────────────
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    benchmark_results_t json_results;
    const char *mname = method == METHOD_POWM    ? "powm"
                        : method == METHOD_TABLE ? "table"
                                                 : "sieve";
    ksc_init_json_results(&json_results, 16, bench_count, mname,
                          nranks, n_threads, challenge, rank);

    bench_accum_t fac_acc = {0, 0}, loc_acc = {0, 0}, gat_acc = {0, 0},
                  com_acc = {0, 0}, recon_acc = {0, 0};
    double cur_max_loc = 0.0;

    // ── 2 warmup runs ───────────────────────────────────────────────────────
    if (method != METHOD_POWM)
    {
        for (int wu = 0; wu < 2; wu++)
        {
            kfactor_list_t wf;
            kfl_init(&wf);
            auto wt0 = std::chrono::high_resolution_clock::now();
            factor_slice(&wf, challenge, all_primes, my_start, my_count, nullptr, nullptr);
            mpz_t wc;
            mpz_init(wc);
            compute_local_cbrt_k16(wc, &wf, cbrt_table, all_primes, prime_count,
                                   modulo, nullptr, nullptr, nullptr, nullptr, false);
            auto wt1 = std::chrono::high_resolution_clock::now();
            if (rank == 0)
                json_results.warmup_total_ms.push_back(
                    std::chrono::duration<double, std::milli>(wt1 - wt0).count());
            mpz_clear(wc);
            kfl_free(&wf);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0)
            printf("Warmup complete (2 runs)\n\n");
    }

    // ── Benchmark loop ───────────────────────────────────────────────────────
    for (int bench = 0; bench < bench_count; bench++)
    {
        kfactor_list_t local_factors;
        kfl_init(&local_factors);

        // ---- FACTORING ----
        bool do_thr_stats = (bench == 1 && rank == 0);
        std::vector<int64_t> thr_tested;
        std::vector<int> thr_found;

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        factor_slice(&local_factors, challenge, all_primes, my_start, my_count,
                     do_thr_stats ? &thr_tested : nullptr,
                     do_thr_stats ? &thr_found : nullptr);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(t_fac1 - t_fac0).count();
            accum_record(&fac_acc, dt);
            if (bench > 0)
                json_results.factor_times.push_back(dt);
            if (bench == 0)
                printf("Rank 0 factors: %d (range [%lu, %lu])\n",
                       local_factors.count,
                       all_primes[my_start],
                       all_primes[my_start + my_count - 1]);
        }
        if (do_thr_stats)
        {
            json_results.thread_primes_tested.insert(
                json_results.thread_primes_tested.end(), thr_tested.begin(), thr_tested.end());
            json_results.thread_primes_found.insert(
                json_results.thread_primes_found.end(), thr_found.begin(), thr_found.end());
        }
        if (bench == 1)
            ksc_gather_prime_stats(&json_results, local_factors.count, my_tested, rank, nranks);

        // ---- COMPLETENESS CHECK ----
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

        unsigned char *all_prods = nullptr;
        if (rank == 0)
            all_prods = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);
        MPI_Gather(prod_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                   all_prods, (int)cbrt_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        free(prod_buf);

        if (rank == 0)
        {
            mpz_t comb;
            mpz_init_set_ui(comb, 1);
            mpz_t part;
            mpz_init(part);
            for (int rk = 0; rk < nranks; rk++)
            {
                mpz_import(part, cbrt_bytes, 1, 1, 0, 0,
                           all_prods + (size_t)rk * cbrt_bytes);
                mpz_mul(comb, comb, part);
            }
            mpz_clear(part);
            free(all_prods);
            mpz_t ch_mod;
            mpz_init(ch_mod);
            mpz_mod(ch_mod, challenge, modulo);
            if (mpz_cmp(comb, ch_mod) != 0)
            {
                fprintf(stderr,
                        "[ERROR] Challenge not fully k16-smooth. "
                        "Ensure it was generated with bitbound=16.\n");
                mpz_clears(comb, ch_mod, NULL);
                kfl_free(&local_factors);
                MPI_Abort(MPI_COMM_WORLD, 1);
                goto cleanup;
            }
            mpz_clears(comb, ch_mod, NULL);
        }

        // ---- LOCAL CBRT ----
        auto t_loc0 = std::chrono::high_resolution_clock::now();
        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double hit_ms = 0.0, miss_ms = 0.0;
        int hits = 0, misses = 0;
        compute_local_cbrt_k16(local_cbrt, &local_factors, cbrt_table,
                               all_primes, prime_count, modulo,
                               &hit_ms, &miss_ms, &hits, &misses, bench > 0);
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
                json_results.cbrt_hit_ms.push_back(hit_ms);
                json_results.cbrt_miss_ms.push_back(miss_ms);
                json_results.cbrt_hits.push_back(hits);
                json_results.cbrt_misses.push_back(misses);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ---- GATHER + COMBINE ----
        double t_gat = 0.0, t_com = 0.0;
        ksc_gather_and_combine_cbrt(result, local_cbrt, modulo, cbrt_bytes,
                                    rank, nranks, 0, nullptr, 0,
                                    &t_gat, &t_com);
        mpz_clear(local_cbrt);

        if (rank == 0)
        {
            accum_record(&gat_acc, t_gat);
            accum_record(&com_acc, t_com);
            if (bench > 0)
            {
                json_results.gather_times.push_back(t_gat);
                json_results.combine_times.push_back(t_com);
            }
            double recon_dt = cur_max_loc + t_gat + t_com;
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

    if (rank == 0)
    {
        ksc_print_timing_report(bench_count, &fac_acc, &loc_acc,
                                &gat_acc, &com_acc, &recon_acc);
        ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                         bench_count, &recon_acc, &fac_acc, rank);
        mpz_t ch_mod;
        mpz_init(ch_mod);
        mpz_mod(ch_mod, challenge, modulo);
        verify_cube_root(result, ch_mod, modulo);
        mpz_clear(ch_mod);
        fflush(stdout);
    }

cleanup:
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
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
    {
        fprintf(stderr, "MPI_THREAD_FUNNELED not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char *challenge_file = "data/challenges/k16/challenge_0_16_7000.txt";
    const char *modulo_file = "data/challenges/k16/modulo_16_7000.txt";
    const char *json_file = nullptr;
    int bench_count = 1;
    run_method_t method = METHOD_SIEVE;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--bench") && i + 1 < argc)
            bench_count = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_file = argv[++i];
        else if (!strcmp(argv[i], "--method") && i + 1 < argc)
        {
            const char *m = argv[++i];
            if (!strcmp(m, "sieve"))
                method = METHOD_SIEVE;
            else if (!strcmp(m, "table"))
                method = METHOD_TABLE;
            else if (!strcmp(m, "powm"))
                method = METHOD_POWM;
            else
            {
                if (rank == 0)
                    fprintf(stderr, "Unknown method '%s'\n", m);
                MPI_Finalize();
                return 1;
            }
        }
        else if (argv[i][0] != '-')
        {
            if (!strcmp(challenge_file, "data/challenges/k16/challenge_0_16_7000.txt"))
                challenge_file = argv[i];
            else
                modulo_file = argv[i];
        }
        else
        {
            if (rank == 0)
                fprintf(stderr, "Unknown option '%s'\n", argv[i]);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0)
    {
        printf("=== k16_mpi ===\n");
        printf("Ranks: %d  Threads/rank: %d  Method: %s  Bench: %d\n\n",
               nranks,
#ifdef _OPENMP
               omp_get_max_threads(),
#else
               1,
#endif
               method == METHOD_POWM ? "powm" : method == METHOD_TABLE ? "table"
                                                                       : "sieve",
               bench_count);
        fflush(stdout);
    }

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