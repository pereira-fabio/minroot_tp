#include "k_common.h"
#include "k_factorization.h"

#include <climits>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Prime list disk I/O
// ---------------------------------------------------------------------------

static bool save_prime_list(const unsigned long *primes, int count, const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return false;
    fwrite(&count, sizeof(int), 1, f);
    fwrite(primes, sizeof(unsigned long), (size_t)count, f);
    fclose(f);
    return true;
}

static unsigned long *load_prime_list(int *count, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        return nullptr;
    int stored_count;
    if (fread(&stored_count, sizeof(int), 1, f) != 1)
    {
        fclose(f);
        return nullptr;
    }
    unsigned long *primes = (unsigned long *)malloc((size_t)stored_count * sizeof(unsigned long));
    if (fread(primes, sizeof(unsigned long), (size_t)stored_count, f) != (size_t)stored_count)
    {
        free(primes);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    *count = stored_count;
    return primes;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    enum solver_method_t
    {
        METHOD_FACTOR,
        METHOD_POWM,
        METHOD_COMPARE
    };

    solver_config_t cfg = {
        "data/challenges/k16/challenge_2_16_2048.txt",
        "data/challenges/k16/modulo_16_2048.txt",
        1, nullptr, false};
    solver_method_t method = METHOD_FACTOR;

    int argi = 1;
    while (argi < argc)
    {
        if (strcmp(argv[argi], "--method") == 0 && argi + 1 < argc)
        {
            const char *m = argv[++argi];
            if (strcmp(m, "factor") == 0)
                method = METHOD_FACTOR;
            else if (strcmp(m, "powm") == 0)
                method = METHOD_POWM;
            else if (strcmp(m, "compare") == 0)
                method = METHOD_COMPARE;
            else
            {
                fprintf(stderr, "[ERROR] Invalid method\n");
                return 1;
            }
            argi++;
        }
        else
        {
            if (parse_common_args(argc, argv, &cfg, &argi) != 0)
                return 1;
        }
    }

    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);
    read_mpz_from_file(challenge, cfg.challenge_file);
    read_mpz_from_file(modulo, cfg.modulo_file);
    size_t cbits = mpz_sizeinbase(challenge, 2);

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    printf("\n============================================================\n");
    printf("=== k16 Modular Cube Root Solver                         ===\n");
    printf("============================================================\n");
    printf(" Size:      %zu bits\n", cbits);
    printf(" Force Fac: %s\n", cfg.force_factor ? "YES" : "NO");
    printf(" Threads:   %d\n", n_threads);
    printf("============================================================\n\n");

    mpz_t *cbrt_table = nullptr;
    int prime_count = 0;
    unsigned long *primes = nullptr;

    // Initialize benchmark results structure
    benchmark_results_t results;
    results.k_size             = 16;
    results.challenge_bits     = cbits;
    results.method_name        = "factor";
    results.num_threads        = n_threads;
    results.mpi_ranks          = 1;
    results.bench_count        = cfg.bench_count;
    results.powm_baseline_ms   = 0.0;

    // =========================================================================
    // METHOD_POWM
    // =========================================================================
    if (method == METHOD_POWM)
    {
        results.method_name = "powm";

        // We must loop to fill the JSON raw_data array for 500 runs
        for (int bench = 0; bench < cfg.bench_count; bench++)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            compute_cube_root_mod_prime(result, challenge, modulo);
            auto t1 = std::chrono::high_resolution_clock::now();

            double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            results.total_times.push_back(total_ms);

            // Still record 0.0 for factor/recon so the JSON structure is consistent
            results.factor_times.push_back(0.0);
            results.recon_times.push_back(0.0);
        }

        // Use the first run or the average as a baseline if needed
        results.powm_baseline_ms = results.total_times[0];

        printf("[RUN] Sequential powm (500 runs) finished. First run: %.3f ms\n",
               results.total_times[0]);
    }

    // =========================================================================
    // METHOD_FACTOR & COMPARE
    // =========================================================================
    if (method == METHOD_FACTOR || method == METHOD_COMPARE)
    {

        bench_accum_t fac_acc = {0, 0};
        bench_accum_t rec_acc = {0, 0};

        bool use_fast_path = (cbits <= 1500 && method != METHOD_COMPARE && !cfg.force_factor);

        if (use_fast_path)
        {
            printf("[INFO] Fast-path engaged for small modulo.\n");
            for (int bench = 0; bench < cfg.bench_count; bench++)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                compute_cube_root_mod_prime(result, challenge, modulo);
                auto t1 = std::chrono::high_resolution_clock::now();
                double dur = std::chrono::duration<double, std::milli>(t1 - t0).count();
                accum_record(&rec_acc, dur);
                results.recon_times.push_back(dur);
                results.factor_times.push_back(0.0);
                results.total_times.push_back(dur);
            }
        }
        else
        {
            char base[504];
            strncpy(base, cfg.modulo_file, sizeof(base) - 1);
            base[sizeof(base) - 1] = '\0';
            char *dot = strrchr(base, '.');
            if (dot)
                *dot = '\0';

            const char *prime_file = "k16_primes.bin";
            char table_file[512];
            snprintf(table_file, sizeof(table_file), "%s.cbrt", base);

            primes = load_prime_list(&prime_count, prime_file);
            if (!primes)
            {
                primes = generate_primes_up_to(UINT16_MAX, &prime_count);
                save_prime_list(primes, prime_count, prime_file);
            }

            cbrt_table = (mpz_t *)malloc((size_t)prime_count * sizeof(mpz_t));
            if (!kf_load_cbrt_table_k16(cbrt_table, prime_count, table_file))
            {
                kf_precompute_cbrt_table_k16(cbrt_table, primes, prime_count, modulo);
                kf_save_cbrt_table_k16(cbrt_table, prime_count, table_file);
            }

            size_t nbytes;
            unsigned char *challenge_bytes = (unsigned char *)mpz_export(NULL, &nbytes, 1, 1, 0, 0, challenge);

            for (int bench = 0; bench < cfg.bench_count; bench++)
            {
                int *exponents = (int *)calloc(prime_count, sizeof(int));
                auto factor_start = std::chrono::high_resolution_clock::now();

                kf_trial_divide_k16(challenge_bytes, nbytes, challenge, primes, prime_count, exponents);

                auto factor_end = std::chrono::high_resolution_clock::now();
                double f_dur = std::chrono::duration<double, std::milli>(factor_end - factor_start).count();
                accum_record(&fac_acc, f_dur);
                results.factor_times.push_back(f_dur);

                mpz_t remaining;
                mpz_init_set(remaining, challenge);
                for (int i = 0; i < prime_count; i++)
                {
                    if (exponents[i] == 0)
                        continue;
                    mpz_t pe;
                    mpz_init(pe);
                    mpz_ui_pow_ui(pe, primes[i], (unsigned long)exponents[i]);
                    mpz_divexact(remaining, remaining, pe);
                    mpz_clear(pe);
                    if (mpz_cmp_ui(remaining, 1) == 0)
                        break;
                }

                double r_dur = 0.0;
                if (mpz_cmp_ui(remaining, 1) == 0)
                {
                    auto root_start = std::chrono::high_resolution_clock::now();
                    kf_compute_cube_root_from_factors_k16(result, exponents, cbrt_table, prime_count, modulo);
                    auto root_end = std::chrono::high_resolution_clock::now();
                    r_dur = std::chrono::duration<double, std::milli>(root_end - root_start).count();
                    accum_record(&rec_acc, r_dur);
                }
                results.recon_times.push_back(r_dur);
                results.total_times.push_back(f_dur + r_dur);

                mpz_clear(remaining);
                free(exponents);
            }
            free(challenge_bytes);
        }

        if (method == METHOD_COMPARE)
        {
            results.method_name = "compare";
            results.powm_baseline_ms = run_powm_baseline(result, challenge, modulo, cfg.bench_count, 30);
            printf("  [1] powm baseline:      %.3f ms\n", results.powm_baseline_ms);
            printf("  [2] factor total avg:   %.3f ms\n", (fac_acc.sum + rec_acc.sum) / cfg.bench_count);
        }
        else
        {
            printf("\n=== Results (%d runs) ===\n", cfg.bench_count);
            if (!use_fast_path)
            {
                print_timing_stat_moments("Factoring:", fac_acc.sum, fac_acc.sumsq, cfg.bench_count, 22, true);
            }
            print_timing_stat_moments("Reconstruction:", rec_acc.sum, rec_acc.sumsq, cfg.bench_count, 22, true);
            printf("----------------------------------------------------------\n");
            print_timing_total("Total Time:", (fac_acc.sum + rec_acc.sum) / cfg.bench_count, 22, NULL);
        }
    }

    if (cfg.json_output_file)
    {
        write_benchmark_json(cfg.json_output_file, results);
    }

    // ---- Cleanup ----
    mpz_clears(challenge, modulo, result, NULL);
    if (cbrt_table)
    {
        for (int i = 0; i < prime_count; i++)
            mpz_clear(cbrt_table[i]);
        free(cbrt_table);
    }
    if (primes)
        free(primes);

    return 0;
}