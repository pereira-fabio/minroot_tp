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
//   table  — precomputed prime file; each rank loads its assigned slice
//   compare— runs distributed factor, then rank 0 runs sequential powm
// =============================================================================

#include "k_common.h"
#include "k_factorization.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>

// ---------------------------------------------------------------------------
// Main solver routine (called per challenge file)
// ---------------------------------------------------------------------------

void solve_one(int rank, int nranks, const char *challenge_file,
               const char *modulo_file, int bench_count,
               int method, // METHOD_SIEVE, METHOD_TABLE, METHOD_COMPARE
               const uint32_t *prime_table, int64_t prime_table_count,
               int sp_count, const unsigned *sp,
               const char *json_output_file)
{
    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    if (rank == 0)
    {
        printf("\n=== %s ===\n", challenge_file);
        read_mpz_from_file(challenge, challenge_file);
        read_mpz_from_file(modulo, modulo_file);
    }

    // Broadcast challenge/modulo to all ranks
    size_t cbytes, mbytes;
    unsigned char *cbuf = NULL, *mbuf = NULL;

    if (rank == 0)
    {
        cbuf = (unsigned char *)mpz_export(NULL, &cbytes, 1, 1, 0, 0, challenge);
        mbuf = (unsigned char *)mpz_export(NULL, &mbytes, 1, 1, 0, 0, modulo);
    }

    MPI_Bcast(&cbytes, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mbytes, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        cbuf = (unsigned char *)malloc(cbytes);
        mbuf = (unsigned char *)malloc(mbytes);
    }

    MPI_Bcast(cbuf, (int)cbytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(mbuf, (int)mbytes, MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        mpz_import(challenge, cbytes, 1, 1, 0, 0, cbuf);
        mpz_import(modulo, mbytes, 1, 1, 0, 0, mbuf);
    }

    free(cbuf);
    free(mbuf);

    size_t cbits = mpz_sizeinbase(challenge, 2);

    // Initialize benchmark results structure on Rank 0
    benchmark_results_t results;
    if (rank == 0)
    {
        results.k_size = 32;
        results.challenge_bits = cbits;
        results.method_name = method == 1 ? "table" : (method == 2 ? "compare" : "sieve");
        results.num_threads = omp_get_max_threads();
        results.mpi_ranks = nranks;
        results.bench_count = bench_count;
        results.powm_baseline_ms = 0.0;
    }

    if (rank == 0)
    {
        printf("Challenge: %zu bits\n", cbits);
        printf("Modulo:    %zu bits\n", mpz_sizeinbase(modulo, 2));
        printf("Ranks: %d  Threads/rank: %d  Total workers: %d\n",
               nranks, omp_get_max_threads(), nranks * omp_get_max_threads());
        if (prime_table && prime_table_count > 0)
            printf("Rank 0 slice: %" PRId64 " primes\n", prime_table_count);
    }

    char base[504];
    if (rank == 0)
    {
        strncpy(base, modulo_file, sizeof(base) - 1);
        base[sizeof(base) - 1] = '\0';
        char *dot = strrchr(base, '.');
        if (dot)
            *dot = '\0';
    }
    MPI_Bcast(base, sizeof(base), MPI_CHAR, 0, MPI_COMM_WORLD);

    char cbrt_file[512];
    snprintf(cbrt_file, sizeof(cbrt_file), "%s_r%d.cbrt", base, rank);

    cbrt_cache_t cbrt_cache;
    kf_cbrt_cache_init(&cbrt_cache);
    kf_cbrt_cache_load(&cbrt_cache, cbrt_file);
    bool cbrt_cache_dirty = false;

    // Buffer for MPI_Gather (1 mpz_t per rank)
    size_t out_capacity = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
    unsigned char *local_cbrt_bytes = (unsigned char *)malloc(out_capacity);
    unsigned char *gathered_bytes = NULL;
    if (rank == 0)
    {
        gathered_bytes = (unsigned char *)malloc(out_capacity * (size_t)nranks);
    }

    bench_accum_t f_acc, r_acc, lc_acc, g_acc, c_acc, att_acc;
    accum_reset(&f_acc);
    accum_reset(&r_acc);
    accum_reset(&lc_acc);
    accum_reset(&g_acc);
    accum_reset(&c_acc);
    accum_reset(&att_acc);

    for (int bench = 0; bench < bench_count; bench++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_start = std::chrono::high_resolution_clock::now();

        gmp_randstate_t rng;
        gmp_randinit_mt(rng);
        gmp_randseed_ui(rng, (unsigned long)(time(NULL) + bench));

        mpz_t r, r_inv, blinded, rem, result_cbrt, local_cbrt;
        mpz_inits(r, r_inv, blinded, rem, result_cbrt, local_cbrt, NULL);

        int attempt = 0;
        bool success = false;
        kfactor_list_t local_factors;
        kfl_init(&local_factors);

        double local_t_fac = 0.0;

        while (!success && attempt < 100)
        {
            auto t_fac_start = std::chrono::high_resolution_clock::now();

            // Rank 0 coordinates the blinding
            if (rank == 0)
            {
                if (attempt == 0)
                {
                    mpz_set_ui(r, 1);
                    mpz_set(blinded, challenge);
                }
                else
                {
                    do
                    {
                        mpz_urandomm(r, rng, modulo);
                    } while (mpz_cmp_ui(r, 2) < 0);

                    mpz_powm_ui(blinded, r, 3, modulo);
                    mpz_mul(blinded, blinded, challenge);
                    mpz_mod(blinded, blinded, modulo);
                    mpz_invert(r_inv, r, modulo);
                }
            }

            size_t bytes_count;
            unsigned char *b_buf = NULL;
            if (rank == 0)
            {
                b_buf = (unsigned char *)mpz_export(NULL, &bytes_count, 1, 1, 0, 0, blinded);
            }
            MPI_Bcast(&bytes_count, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
            if (rank != 0)
            {
                b_buf = (unsigned char *)malloc(bytes_count);
            }
            MPI_Bcast(b_buf, (int)bytes_count, MPI_BYTE, 0, MPI_COMM_WORLD);
            if (rank != 0)
            {
                mpz_import(blinded, bytes_count, 1, 1, 0, 0, b_buf);
            }
            free(b_buf);

            kfl_free(&local_factors);

            if (prime_table && prime_table_count > 0)
                kf_trial_divide_table(blinded, prime_table, prime_table_count, &local_factors);
            else
            {
                uint64_t full_range = (uint64_t)UINT32_MAX - 65537ULL + 1;
                uint64_t slice = full_range / (uint64_t)nranks;
                uint64_t my_lo = 65537ULL + (uint64_t)rank * slice;
                uint64_t my_hi = (rank == nranks - 1) ? (uint64_t)UINT32_MAX : my_lo + slice - 1;
                kf_trial_divide_sieve(blinded, my_lo, my_hi, sp_count, sp, &local_factors);
            }

            mpz_set(rem, blinded);
            for (int i = 0; i < local_factors.count; i++)
            {
                unsigned long p = mpz_get_ui(local_factors.items[i].prime);
                for (int ex = 0; ex < local_factors.items[i].exponent; ex++)
                    mpz_divexact_ui(rem, rem, p);
            }

            mpz_t rem_prod;
            mpz_init(rem_prod);

            size_t l_bytes;
            unsigned char *r_buf = (unsigned char *)mpz_export(NULL, &l_bytes, 1, 1, 0, 0, rem);

            if (rank == 0)
            {
                mpz_t *rems = (mpz_t *)malloc((size_t)nranks * sizeof(mpz_t));
                for (int i = 0; i < nranks; i++)
                    mpz_init(rems[i]);

                size_t *all_bytes = (size_t *)malloc((size_t)nranks * sizeof(size_t));
                // Only ONE Gather for the sizes
                MPI_Gather(&l_bytes, sizeof(size_t), MPI_BYTE, all_bytes, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

                int *displs = (int *)malloc((size_t)nranks * sizeof(int));
                int *rcounts = (int *)malloc((size_t)nranks * sizeof(int));
                int total_bytes = 0;
                for (int i = 0; i < nranks; i++)
                {
                    rcounts[i] = (int)all_bytes[i];
                    displs[i] = total_bytes;
                    total_bytes += rcounts[i];
                }

                unsigned char *recv_buf = (unsigned char *)malloc((size_t)total_bytes);
                MPI_Gatherv(r_buf, (int)l_bytes, MPI_BYTE, recv_buf, rcounts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);

                for (int i = 0; i < nranks; i++)
                {
                    if (rcounts[i] > 0)
                        mpz_import(rems[i], (size_t)rcounts[i], 1, 1, 0, 0, recv_buf + displs[i]);
                    else
                        mpz_set_ui(rems[i], 0);
                }

                mpz_set(rem_prod, blinded);
                for (int i = 0; i < nranks; i++)
                {
                    if (mpz_cmp_ui(rems[i], 0) > 0)
                        mpz_divexact(rem_prod, rem_prod, rems[i]);
                }
                mpz_mul(rem_prod, rem_prod, rems[0]);

                free(all_bytes);
                free(displs);
                free(rcounts);
                free(recv_buf);
                for (int i = 0; i < nranks; i++)
                    mpz_clear(rems[i]);
                free(rems);
            }
            else
            {
                // Worker ranks send their size, then their data
                MPI_Gather(&l_bytes, sizeof(size_t), MPI_BYTE, NULL, 0, MPI_BYTE, 0, MPI_COMM_WORLD);
                MPI_Gatherv(r_buf, (int)l_bytes, MPI_BYTE, NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
            }
            free(r_buf);
            int global_success = 0;
            if (rank == 0)
            {
                if (mpz_cmp_ui(rem_prod, 1) == 0)
                    global_success = 1;
            }
            MPI_Bcast(&global_success, 1, MPI_INT, 0, MPI_COMM_WORLD);

            attempt++;
            if (global_success)
            {
                success = true;
                if (rank == 0 && bench == 0)
                {
                    printf("Blinding attempt %d: smooth — proceeding\n", attempt);
                }
            }

            mpz_clear(rem_prod);
            auto t_fac_end = std::chrono::high_resolution_clock::now();
            local_t_fac += std::chrono::duration<double, std::milli>(t_fac_end - t_fac_start).count();
        }

        double max_t_fac = 0.0;
        MPI_Reduce(&local_t_fac, &max_t_fac, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double t_fac = max_t_fac;

        if (rank == 0 && bench == 0)
        {
            printf("Factors found by rank 0: %d distinct primes (after %d attempt(s))\n",
                   local_factors.count, attempt);
        }

        // =========================================================================
        // Distributed Reconstruction
        // =========================================================================

        auto t_lc_start = std::chrono::high_resolution_clock::now();
        kf_compute_cbrt_from_factors(local_cbrt, &local_factors, modulo,
                                     &cbrt_cache, &cbrt_cache_dirty);
        auto t_lc_end = std::chrono::high_resolution_clock::now();
        double local_t_cbrt = std::chrono::duration<double, std::milli>(t_lc_end - t_lc_start).count();

        double max_t_cbrt = 0.0;
        MPI_Reduce(&local_t_cbrt, &max_t_cbrt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_g_start = std::chrono::high_resolution_clock::now();

        size_t exp_size;
        memset(local_cbrt_bytes, 0, out_capacity);
        mpz_export(local_cbrt_bytes, &exp_size, 1, 1, 0, 0, local_cbrt);

        MPI_Gather(local_cbrt_bytes, (int)out_capacity, MPI_BYTE,
                   gathered_bytes, (int)out_capacity, MPI_BYTE,
                   0, MPI_COMM_WORLD);

        auto t_g_end = std::chrono::high_resolution_clock::now();
        double t_gather = std::chrono::duration<double, std::milli>(t_g_end - t_g_start).count();

        double t_combine = 0.0;
        if (rank == 0)
        {
            auto t_c_start = std::chrono::high_resolution_clock::now();
            mpz_set_ui(result_cbrt, 1);
            mpz_t tmp;
            mpz_init(tmp);
            for (int i = 0; i < nranks; i++)
            {
                mpz_import(tmp, out_capacity, 1, 1, 0, 0, gathered_bytes + i * out_capacity);
                mpz_mul(result_cbrt, result_cbrt, tmp);
                mpz_mod(result_cbrt, result_cbrt, modulo);
            }
            mpz_clear(tmp);

            if (attempt > 1)
            {
                mpz_mul(result, result_cbrt, r_inv);
                mpz_mod(result, result, modulo);
            }
            else
            {
                mpz_set(result, result_cbrt);
            }

            auto t_c_end = std::chrono::high_resolution_clock::now();
            t_combine = std::chrono::duration<double, std::milli>(t_c_end - t_c_start).count();
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double t_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double t_recon = max_t_cbrt + t_gather + t_combine;

        if (rank == 0)
        {
            // Record raw vectors for JSON
            results.factor_times.push_back(t_fac);
            results.local_cbrt_times.push_back(max_t_cbrt);
            results.gather_times.push_back(t_gather);
            results.combine_times.push_back(t_combine);
            results.recon_times.push_back(t_recon);
            results.total_times.push_back(t_total);
            results.blinding_attempts.push_back(attempt);

            // Record accumulators for terminal output
            accum_record(&f_acc, t_fac);
            accum_record(&lc_acc, max_t_cbrt);
            accum_record(&g_acc, t_gather);
            accum_record(&c_acc, t_combine);
            accum_record(&r_acc, t_recon);
            accum_record(&att_acc, (double)attempt);
        }

        kfl_free(&local_factors);
        mpz_clears(r, r_inv, blinded, rem, result_cbrt, local_cbrt, NULL);
        gmp_randclear(rng);
    }

    if (cbrt_cache_dirty)
    {
        kf_cbrt_cache_save(&cbrt_cache, cbrt_file);
    }

    if (rank == 0)
    {
        printf("\nResults over %d runs:\n", bench_count);
        print_timing_stat_moments("Factoring:", f_acc.sum, f_acc.sumsq, bench_count, 30, true);
        print_timing_stat_moments("Local cbrt (max):", lc_acc.sum, lc_acc.sumsq, bench_count, 30, true);
        print_timing_stat_moments("Gather:", g_acc.sum, g_acc.sumsq, bench_count, 30, true);
        print_timing_stat_moments("Combine (rank 0):", c_acc.sum, c_acc.sumsq, bench_count, 30, true);
        print_timing_stat_moments("Reconstruction total:", r_acc.sum, r_acc.sumsq, bench_count, 30, true);
        print_timing_total("Total:", (f_acc.sum + r_acc.sum) / bench_count, 30, NULL);

        double att_sd = bench_count > 1 ? compute_stddev_from_moments(att_acc.sum, att_acc.sumsq, bench_count, true) : 0.0;
        printf("    %-30s avg= %9.2f      stddev= %6.2f\n", "Blinding attempts:", att_acc.sum / bench_count, att_sd);

        if (method == 2) // METHOD_COMPARE
        {
            printf("\n--- Sequential powm baseline (rank 0, %d threads unused) ---\n", omp_get_max_threads());
            double powm_avg = run_powm_baseline(result, challenge, modulo, bench_count, 30);
            results.powm_baseline_ms = powm_avg;

            double tr_avg = r_acc.sum / bench_count;
            double f2_total_avg = (f_acc.sum + r_acc.sum) / bench_count;

            printf("\n--- Speedup vs sequential powm ---\n");
            printf("  powm vs reconstruction only: %5.2fx  (parallel cbrt, no factoring cost)\n",
                   tr_avg > 0 ? powm_avg / tr_avg : 0.0);
            printf("  powm vs total (factor+recon):%5.2fx  (end-to-end latency)\n",
                   f2_total_avg > 0 ? powm_avg / f2_total_avg : 0.0);
        }

        // Write JSON Benchmark output
        if (json_output_file)
        {
            char final_json_path[512];
            strncpy(final_json_path, json_output_file, sizeof(final_json_path) - 1);
            final_json_path[sizeof(final_json_path) - 1] = '\0';

            // Inject bit size into filename if running multiple benchmarks
            char *dot = strrchr(final_json_path, '.');
            if (dot && bench_count > 0)
            {
                char ext[16];
                strncpy(ext, dot, sizeof(ext));
                ext[sizeof(ext) - 1] = '\0';
                *dot = '\0'; // truncate at the dot
                snprintf(final_json_path + strlen(final_json_path),
                         sizeof(final_json_path) - strlen(final_json_path),
                         "_%zu%s", cbits, ext);
            }

            write_benchmark_json(final_json_path, results);
        }

        mpz_t verify_mod;
        mpz_init(verify_mod);
        mpz_mod(verify_mod, challenge, modulo);
        verify_cube_root(result, verify_mod, modulo);
        mpz_clear(verify_mod);
    }

    free(local_cbrt_bytes);
    if (rank == 0)
        free(gathered_bytes);

    mpz_clears(challenge, modulo, result, NULL);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    solver_config_t cfg = {
        "data/challenges/k32/challenge_32_2048.txt",
        "data/challenges/k32/modulo_32_2048.txt",
        1, nullptr, false};

    int method = 0; // 0=sieve, 1=table, 2=compare
    bool run_all = false;
    const char *data_dir = "data/challenges/k32";
    const char *primes_file = "k32_primes.bin";

    int argi = 1;
    while (argi < argc)
    {
        if (strcmp(argv[argi], "--method") == 0)
        {
            if (argi + 1 >= argc)
            {
                if (rank == 0)
                    fprintf(stderr, "Error: Missing value for --method\n");
                MPI_Finalize();
                return 1;
            }
            if (strcmp(argv[argi + 1], "table") == 0)
                method = 1;
            else if (strcmp(argv[argi + 1], "compare") == 0)
                method = 2;
            else
                method = 0;
            argi += 2;
        }
        else if (strcmp(argv[argi], "--all") == 0)
        {
            run_all = true;
            argi++;
        }
        else
        {
            if (parse_common_args(argc, argv, &cfg, &argi) != 0)
            {
                MPI_Finalize();
                return 1;
            }
        }
    }

    if (rank == 0)
    {
        printf("=== Job info ===\n");
        printf("MPI ranks:      %d\n", nranks);
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
        printf("Method:         %s\n", method == 1 ? "table" : (method == 2 ? "compare" : "sieve"));
        if (cfg.bench_count > 1)
            printf("Benchmark runs: %d\n", cfg.bench_count);
        if (run_all)
            printf("Mode:           --all\n");
        printf("\n");
        fflush(stdout);
    }

    int sp_count = 0;
    unsigned *sp = kf_generate_small_primes(&sp_count);

    uint32_t *prime_table = NULL;
    int64_t prime_table_count = 0;

    if (method == 1 || method == 2)
    {
        int table_ok = 0;
        if (rank == 0)
        {
            FILE *test = fopen(primes_file, "rb");
            if (!test)
            {
                printf("[rank 0] Prime table not found — generating...\n");
                fflush(stdout);
                table_ok = kf_generate_and_save_prime_table(primes_file, sp_count, sp, "[rank 0] ") ? 1 : 0;
            }
            else
            {
                fclose(test);
                table_ok = 1;
            }
        }
        MPI_Bcast(&table_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!table_ok)
        {
            if (rank == 0)
                fprintf(stderr, "Failed to create prime table\n");
            free(sp);
            MPI_Finalize();
            return 1;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        FILE *f = fopen(primes_file, "rb");
        if (f)
        {
            uint64_t magic, total_primes;
            // Verify magic header and read the total prime count
            if (fread(&magic, sizeof(uint64_t), 1, f) == 1 && magic == KF_PRIME_TABLE_MAGIC)
            {
                if (fread(&total_primes, sizeof(uint64_t), 1, f) == 1)
                {

                    // Calculate this rank's specific slice of the prime table
                    uint64_t slice = total_primes / (uint64_t)nranks;
                    prime_table_count = (rank == nranks - 1) ? (total_primes - (uint64_t)rank * slice) : slice;
                    prime_table = (uint32_t *)malloc((size_t)prime_table_count * sizeof(uint32_t));

                    // Seek to the exact offset for this rank and load
                    fseek(f, sizeof(uint64_t) * 2 + rank * slice * sizeof(uint32_t), SEEK_SET);
                    if (fread(prime_table, sizeof(uint32_t), (size_t)prime_table_count, f) != (size_t)prime_table_count)
                    {
                        free(prime_table);
                        prime_table = NULL;
                    }
                }
            }
            fclose(f);
        }

        if (!prime_table)
        {
            fprintf(stderr, "Rank %d failed to load prime table\n", rank);
            free(sp);
            MPI_Finalize();
            return 1;
        }
        if (rank == 0)
        {
            printf("Prime range per rank: ~%.1fM primes\n\n", (double)prime_table_count / 1e6);
        }
    }
    else
    {
        if (rank == 0)
        {
            uint64_t full_range = (uint64_t)UINT32_MAX - 65537ULL + 1;
            uint64_t slice = full_range / (uint64_t)nranks;
            printf("Prime range per rank: ~%.1fM primes\n\n", (double)slice / 1e6);
        }
    }

    if (run_all)
    {
        const char *sizes[] = {"1024", "2048", "5000", "7000"};
        for (int s = 0; s < 4; s++)
        {
            char cf[512], mf[512];
            snprintf(cf, sizeof(cf), "%s/challenge_32_%s.txt", data_dir, sizes[s]);
            snprintf(mf, sizeof(mf), "%s/modulo_32_%s.txt", data_dir, sizes[s]);
            solve_one(rank, nranks, cf, mf, cfg.bench_count, method,
                      prime_table, prime_table_count, sp_count, sp, cfg.json_output_file);
        }
    }
    else
    {
        solve_one(rank, nranks, cfg.challenge_file, cfg.modulo_file, cfg.bench_count, method,
                  prime_table, prime_table_count, sp_count, sp, cfg.json_output_file);
    }

    if (prime_table)
        free(prime_table);
    free(sp);

    MPI_Finalize();
    return 0;
}