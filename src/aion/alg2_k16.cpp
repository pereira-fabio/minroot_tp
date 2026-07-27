// =============================================================================
// alg2_k16.cpp — MPI+OpenMP hybrid cube root solver for 16-bit prime factors
//
// Partitions the 6542 primes ≤ 65536 across MPI ranks. Each rank factors its
// slice of the challenge, computes local cbrt products, and sends one partial
// product to rank 0 for final combination.
//
// Methods:  sieve | table | powm
// =============================================================================

#include "common.h"
#include "factorization.h"
#include "solver_common.h"
#include "sizes.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// run_method_t defined in common.h

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
// run_one — run benchmark for one challenge file, return results (no JSON write)
// ---------------------------------------------------------------------------

static benchmark_results_t run_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    run_method_t method,
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
        return benchmark_results_t{};  // unreachable — ksc_broadcast aborts
    }

    if (rank == 0)
    {
        const char *mname = method == METHOD_TABLE ? "table"
                          : method == METHOD_POWM  ? "powm" : "sieve";
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

    // ── Primes ──────────────────────────────────────────────────────────────
    int prime_count = 0;
    unsigned long *all_primes = ksc_ensure_k16_primes(&prime_count);

    // Partition prime array across ranks
    int base_per_rank = prime_count / nranks;
    int remainder = prime_count % nranks;
    int my_start = rank * base_per_rank + (rank < remainder ? rank : remainder);
    int my_count = base_per_rank + (rank < remainder ? 1 : 0);
    int64_t my_tested = (int64_t)my_count;


    // Setup for early-exit trial division via the common library functions.
    // kf_trial_divide_sieve needs small primes; kf_trial_divide_table needs
    // a uint32_t slice (all k16 primes fit — they are all <= 65535).
    int sp_count = 0;
    unsigned *sp = nullptr;
    uint32_t *my_primes32 = nullptr;
    uint64_t my_lo = 0, my_hi = 0;
    if (my_count > 0)
    {
        my_lo = (uint64_t)all_primes[my_start];
        my_hi = (uint64_t)all_primes[my_start + my_count - 1];
    }
    if (method == METHOD_TABLE)
    {
        my_primes32 = (uint32_t *)malloc((size_t)my_count * sizeof(uint32_t));
        for (int i = 0; i < my_count; i++)
            my_primes32[i] = (uint32_t)all_primes[my_start + i];
    }
    else
    {
        sp = kf_generate_small_primes(&sp_count);
    }

    // ── cbrt table ──────────────────────────────────────────────────────────
    // powm method never uses the cbrt table — skip load and broadcast
    mpz_t *cbrt_table = (mpz_t *)malloc((size_t)prime_count * sizeof(mpz_t));
    for (int i = 0; i < prime_count; i++)
        mpz_init(cbrt_table[i]);
    if (method != METHOD_POWM)
        ksc_ensure_k16_cbrt(cbrt_table, all_primes, prime_count,
                            modulo, modulo_file, cbrt_dir, rank, exponent);

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
    ksc_init_json_results(&json_results, 16, mname,
                          nranks, n_threads, challenge, rank);
    json_results.config.algorithm = "k16";
    json_results.config.exponent  = exponent;

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
            if (my_count > 0)
            {
                if (method == METHOD_TABLE)
                    kf_trial_divide_table(challenge, my_primes32, my_count, &wf);
                else
                    kf_trial_divide_sieve(challenge, my_lo, my_hi, sp_count, sp, &wf);
            }
            mpz_t wc;
            mpz_init(wc);
            compute_local_cbrt_k16(wc, &wf, cbrt_table, all_primes, prime_count,
                                   modulo, nullptr, nullptr, nullptr, nullptr, false);
            auto wt1 = std::chrono::high_resolution_clock::now();
            if (rank == 0)
                json_results.timing.warmup_total_ms.push_back(
                    std::chrono::duration<double, std::milli>(wt1 - wt0).count());
            mpz_clear(wc);
            kfl_free(&wf);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // ── Single timed run for this challenge ──────────────────────────────────
    // (Multiple challenges are run by the caller and merged; the old per-challenge
    // repeat / --bench is removed.  The `bench < 1` loop runs exactly once but
    // keeps the early-`break` on a non-smooth challenge below.)
    bool k16_smooth = true;
    for (int bench = 0; bench < 1 && method != METHOD_POWM; bench++)
    {
        kfactor_list_t local_factors;
        kfl_init(&local_factors);

        // ---- FACTORING ----
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        if (my_count > 0)
        {
            if (method == METHOD_TABLE)
                kf_trial_divide_table(challenge, my_primes32, my_count, &local_factors);
            else
                kf_trial_divide_sieve(challenge, my_lo, my_hi, sp_count, sp, &local_factors);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(t_fac1 - t_fac0).count();
            accum_record(&fac_acc, dt);
            json_results.timing.factor_times.push_back(dt);
        }
        if (bench == 0)
            ksc_gather_prime_stats(&json_results, local_factors.count, my_tested, rank, nranks);

        // ---- COMPLETENESS CHECK (Algorithm 2 — fully B-smooth) ----
        // Gather each rank's partial factor product to rank 0, multiply them,
        // and verify that the combined product equals the challenge exactly.
        // k16 challenges have no cofactor — any remainder means the challenge
        // file was not generated with bitbound=16.
        mpz_t local_prod;
        mpz_init_set_ui(local_prod, 1);
        for (int i = 0; i < local_factors.count; i++)
        {
            unsigned long p = mpz_get_ui(local_factors.items[i].prime);
            for (int ex = 0; ex < local_factors.items[i].exponent; ex++)
                mpz_mul_ui(local_prod, local_prod, p);
        }
        // Pass challenge directly (not challenge mod modulo) — k16 factors the
        // raw challenge, so combined_prod must equal challenge, not its reduction.
        int k16_complete = ksc_alg2_broadcast_complete(
            challenge, local_prod, cbrt_bytes, rank, nranks);
        mpz_clear(local_prod);
        if (!k16_complete)
        {
            if (rank == 0)
            {
                fprintf(stderr, "[WARN] %s not fully k16-smooth — skipping challenge.\n",
                        challenge_file);
                fflush(stderr);
            }
            kfl_free(&local_factors);
            k16_smooth = false;
            break;
        }

        // ---- LOCAL CBRT ----
        auto t_loc0 = std::chrono::high_resolution_clock::now();
        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double hit_ms = 0.0, miss_ms = 0.0;
        int hits = 0, misses = 0;
        compute_local_cbrt_k16(local_cbrt, &local_factors, cbrt_table,
                               all_primes, prime_count, modulo,
                               &hit_ms, &miss_ms, &hits, &misses, /*record=*/true);
        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(t_loc1 - t_loc0).count();

        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            accum_record(&loc_acc, max_local_dt);
            cur_max_loc = max_local_dt;
            json_results.timing.local_cbrt_times.push_back(max_local_dt);
            json_results.timing.cbrt_hit_ms.push_back(hit_ms);
            json_results.timing.cbrt_miss_ms.push_back(miss_ms);
            json_results.timing.cbrt_hits.push_back(hits);
            json_results.timing.cbrt_misses.push_back(misses);
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
            json_results.timing.gather_times.push_back(t_gat);
            json_results.timing.combine_times.push_back(t_com);
            double recon_dt = cur_max_loc + t_gat + t_com;
            accum_record(&recon_acc, recon_dt);
            json_results.timing.recon_times.push_back(recon_dt);
            if (!json_results.timing.factor_times.empty())
                json_results.timing.total_times.push_back(
                    json_results.timing.factor_times.back() + recon_dt);
        }

        kfl_free(&local_factors);
    }

    if (rank == 0 && k16_smooth)
    {
        ksc_print_timing_report(&fac_acc, &loc_acc,
                                &gat_acc, &com_acc, &recon_acc);
        if (!skip_powm)
            ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                             &recon_acc, &fac_acc, rank, exponent);
        if (method != METHOD_POWM)
        {
            mpz_t ch_mod;
            mpz_init(ch_mod);
            mpz_mod(ch_mod, challenge, modulo);
            verify_eth_root(result, ch_mod, modulo, exponent);
            mpz_clear(ch_mod);
        }
        fflush(stdout);
    }

    free(sp);
    free(my_primes32);
    for (int i = 0; i < prime_count; i++)
        mpz_clear(cbrt_table[i]);
    free(cbrt_table);
    free(all_primes);
    mpz_clears(challenge, modulo, result, NULL);

    if (!k16_smooth)
        return benchmark_results_t{};
    return json_results;
}

// solve_one — backward-compatible wrapper: run_one + write JSON
static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    run_method_t method,
    unsigned long exponent,
    const char *json_output_file,
    const char *cbrt_dir = KSC_CBRT_DIR,
    bool skip_powm = false)
{
    benchmark_results_t r = run_one(rank, nranks, challenge_file, modulo_file,
                                    method, exponent, cbrt_dir, skip_powm);
    if (rank == 0 && json_output_file && json_output_file[0] != '\0')
        write_benchmark_json(json_output_file, r);
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

    const char *challenge_file = nullptr;
    const char *modulo_file = nullptr;
    const char *data_dir = "data/challenges/k16";
    const char *json_file = nullptr;
    const char *modulo_suffix = "";
    const char *cbrt_dir = KSC_CBRT_DIR;
    int run_all = 0;
    int n_challenges = 1;
    bool skip_powm = false;
    run_method_t method = METHOD_SIEVE;
    unsigned long exponent = 3;
    int positional = 0;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_file = argv[++i];
        else if (!strcmp(argv[i], "--exponent") && i + 1 < argc)
            exponent = (unsigned long)atol(argv[++i]);
        else if (!strcmp(argv[i], "--n-challenges") && i + 1 < argc)
        {
            n_challenges = atoi(argv[++i]);
            if (n_challenges < 1) n_challenges = 1;
        }
        else if (!strcmp(argv[i], "--cbrt-dir") && i + 1 < argc)
            cbrt_dir = argv[++i];
        else if (!strcmp(argv[i], "--skip-powm-baseline"))
            skip_powm = true;
        else if (!strcmp(argv[i], "--all"))
            run_all = 1;
        else if (!strcmp(argv[i], "--data") && i + 1 < argc)
            data_dir = argv[++i];
        else if (!strcmp(argv[i], "--modulo-suffix") && i + 1 < argc)
            modulo_suffix = argv[++i];
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
            if (positional == 0)      { challenge_file = argv[i]; positional++; }
            else if (positional == 1) { modulo_file    = argv[i]; positional++; }
            else
            {
                if (rank == 0) fprintf(stderr, "Unexpected argument '%s'\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
        }
        else
        {
            if (rank == 0)
                fprintf(stderr, "Unknown option '%s'\n", argv[i]);
            MPI_Finalize();
            return 1;
        }
    }

    if (!run_all && (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: k16_mpi <challenge> <modulo> [--method sieve|table|powm]\n"
                    "       k16_mpi --all [--data DIR] [--method sieve|table|powm]\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("=== k16_mpi ===\n");
        printf("Ranks: %d  Threads/rank: %d  Method: %s\n\n",
               nranks,
#ifdef _OPENMP
               omp_get_max_threads(),
#else
               1,
#endif
               method == METHOD_POWM ? "powm" : method == METHOD_TABLE ? "table"
                                                                       : "sieve");
        fflush(stdout);
    }

#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    if (run_all)
    {
        const char *sizes[] = { KSC_CANONICAL_SIZES_LIST };
        for (int s = 0; s < KSC_N_CANONICAL_SIZES; s++)
        {
            char cf[512], mf[512], jf[512];
            snprintf(mf, sizeof(mf), "%s/modulo_16_%s%s.txt", data_dir, sizes[s], modulo_suffix);
            const char *jf_ptr = nullptr;
            if (json_file)
            {
                snprintf(jf, sizeof(jf), "%s_%s.json", json_file, sizes[s]);
                jf_ptr = jf;
            }
            if (n_challenges == 1)
            {
                snprintf(cf, sizeof(cf), "%s/challenge_0_16_%s.txt", data_dir, sizes[s]);
                solve_one(rank, nranks, cf, mf, method, exponent, jf_ptr,
                          cbrt_dir, skip_powm);
            }
            else
            {
                // Plan A (2026-07-17): every challenge 0..n_challenges-1 is a timed
                // warm run — no separate cold run (the dense cbrt table is loaded at
                // setup, so challenge_0 is statistically identical to the rest).
                benchmark_results_t combined;
                bool has_data = false;
                for (int ci = 0; ci < n_challenges; ci++)
                {
                    char cfi[512];
                    snprintf(cfi, sizeof(cfi), "%s/challenge_%d_16_%s.txt",
                             data_dir, ci, sizes[s]);
                    int exists = 0;
                    if (rank == 0)
                    {
                        FILE *fck = fopen(cfi, "r");
                        if (fck) { fclose(fck); exists = 1; }
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
                                                    method, exponent, cbrt_dir, skip_powm);
                    if (!has_data) { combined = std::move(r); has_data = true; }
                    else           { merge_benchmark_results(combined, r); }
                }
                if (has_data && rank == 0 && jf_ptr)
                    write_benchmark_json(jf_ptr, combined);
            }
        }
    }
    else
    {
        solve_one(rank, nranks, challenge_file, modulo_file,
                  method, exponent, json_file, cbrt_dir, skip_powm);
    }

    MPI_Finalize();
    return 0;
}

// Edit: 2026-07-17

