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

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define SEG_SIZE (1 << 19)                      // 512 KiB sieve segment
#define PRIME_TABLE_MAGIC 0x4B33325052494D53ULL // "K32PRIMS"
#define WRITE_BATCH 65536

typedef enum
{
    METHOD_SIEVE,
    METHOD_TABLE,
    METHOD_COMPARE 
} run_method_t;

#define factor_list_t kfactor_list_t
#define fl_init kfl_init
#define fl_free kfl_free
#define fl_append(f, p, e) kfl_append_ul(f, (unsigned long)(p), (int)(e))

// ---------------------------------------------------------------------------
// Local cube root reconstruction — called by EVERY rank
// ---------------------------------------------------------------------------

static void compute_local_cbrt(mpz_t result,
                               const factor_list_t *factors,
                               const mpz_t e,
                               const mpz_t modulo,
                               cbrt_cache_t *cache,
                               bool *cache_dirty)
{
    int n = factors->count;

    if (n == 0)
    {
        mpz_set_ui(result, 1);
        return;
    }

    mpz_t *parts = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    for (int i = 0; i < n; i++)
        mpz_init_set_ui(parts[i], 1);

    mpz_t *cbrt_vals = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    for (int i = 0; i < n; i++)
    {
        mpz_init(cbrt_vals[i]);
        unsigned long p_ul = mpz_get_ui(factors->items[i].prime);
        int cache_idx = (cache && p_ul > 0)
                      ? kf_cbrt_cache_find(cache, p_ul) : -1;

        if (cache_idx >= 0)
        {
            mpz_set(cbrt_vals[i], cache->items[cache_idx].cbrt);
        }
        else
        {
            mpz_t p_mpz;
            mpz_init_set_ui(p_mpz, p_ul);
            mpz_powm(cbrt_vals[i], p_mpz, e, modulo);
            mpz_clear(p_mpz);

            if (cache && p_ul > 0)
            {
                kf_cbrt_cache_append(cache, p_ul, cbrt_vals[i]);
                if (cache_dirty) *cache_dirty = true;
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++)
    {
        if (factors->items[i].exponent == 1)
            mpz_set(parts[i], cbrt_vals[i]);
        else
            mpz_powm_ui(parts[i], cbrt_vals[i],
                        (unsigned long)factors->items[i].exponent, modulo);
    }

    for (int i = 0; i < n; i++)
        mpz_clear(cbrt_vals[i]);
    free(cbrt_vals);

    mpz_set_ui(result, 1);
    for (int i = 0; i < n; i++)
    {
        mpz_mul(result, result, parts[i]);
        mpz_mod(result, result, modulo);
        mpz_clear(parts[i]);
    }
    free(parts);
}

#define print_stat(lbl, sum, sumsq, n) print_timing_stat_moments(lbl, sum, sumsq, n, 30, true)

// ---------------------------------------------------------------------------
// solve_one 
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count, int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method, const char *primes_file, 
    const char *json_output_file) // NEW JSON PARAMETER
{
    if (rank == 0)
    {
        printf("\n=== %s ===\n", challenge_file);
        fflush(stdout);
    }

    int read_ok = 1;
    size_t challenge_nbytes = 0, modulo_nbytes = 0;
    unsigned char *challenge_bytes = NULL, *modulo_bytes = NULL;

    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    if (rank == 0)
    {
        FILE *fc = fopen(challenge_file, "rb");
        FILE *fm = fopen(modulo_file, "rb");
        if (!fc || !fm)
        {
            fprintf(stderr, "Error: cannot open '%s' or '%s'\n",
                    challenge_file, modulo_file);
            if (fc) fclose(fc);
            if (fm) fclose(fm);
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

    uint64_t cn64 = (uint64_t)challenge_nbytes;
    uint64_t mn64 = (uint64_t)modulo_nbytes;
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

    // Initialize benchmark results structure on Rank 0
    benchmark_results_t results;
    if (rank == 0) {
        results.k_size = 32;
        results.challenge_bits = mpz_sizeinbase(challenge, 2);
        results.method_name = method == METHOD_TABLE ? "table" : (method == METHOD_COMPARE ? "compare" : "sieve");
        results.num_threads = omp_get_max_threads();
        results.mpi_ranks = nranks;
        results.bench_count = bench_count;
        results.powm_baseline_ms = 0.0;
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

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1; 

    mpz_t phi, e_inv, three_mpz;
    mpz_inits(phi, e_inv, three_mpz, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three_mpz, 3);
    mpz_invert(e_inv, three_mpz, phi);
    mpz_clears(phi, three_mpz, NULL);

    char cbrt_cache_path[600];
    {
        const char *base = modulo_file ? modulo_file : "modulo";
        char stripped[512];
        strncpy(stripped, base, sizeof(stripped) - 1);
        stripped[sizeof(stripped) - 1] = '\0';
        size_t blen = strlen(stripped);
        if (blen >= 4 && strcmp(stripped + blen - 4, ".txt") == 0)
            stripped[blen - 4] = '\0';
        snprintf(cbrt_cache_path, sizeof(cbrt_cache_path),
                 "%s.cbrt.%d", stripped, rank);
    }

    cbrt_cache_t local_cbrt_cache;
    kf_cbrt_cache_init(&local_cbrt_cache);
    bool cbrt_cache_dirty = false;

    if (kf_cbrt_cache_load(&local_cbrt_cache, cbrt_cache_path))
    {
        if (rank == 0)
            printf("cbrt cache: loaded %d entries from %s\n",
                   local_cbrt_cache.count, cbrt_cache_path);
    }
    else if (rank == 0)
        printf("cbrt cache: not found at %s — will build on first run\n",
               cbrt_cache_path);

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

    bench_accum_t fac_acc = {0, 0};   
    bench_accum_t loc_acc = {0, 0};   
    bench_accum_t gat_acc = {0, 0};   
    bench_accum_t com_acc = {0, 0};   
    bench_accum_t recon_acc = {0, 0}; 
    double cur_max_loc = 0.0;         

    for (int bench = 0; bench < bench_count; bench++)
    {
        factor_list_t local_factors;
        fl_init(&local_factors);
        
        // Iteration metrics for JSON
        double iter_fac = 0.0, iter_loc = 0.0, iter_gat = 0.0, iter_com = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        int attempt = 0;
        const int MAX_ATTEMPTS = 256;
        size_t blinded_nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
        unsigned char *blinded_buf = (unsigned char *)malloc(blinded_nbytes);

        for (attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
        {
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

            fl_free(&local_factors);
            fl_init(&local_factors);

            if (method == METHOD_SIEVE || method == METHOD_COMPARE)
                kf_trial_divide_sieve(blinded_challenge, my_lo, my_hi,
                                      sp_count, sp, &local_factors);
            else
                kf_trial_divide_table(blinded_challenge, prime_slice,
                                      slice_count, &local_factors);

            // STABLE LOGIC: Gathering local_prod, NOT remainders!
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
        } 

        free(blinded_buf);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(t_fac1 - t_fac0).count();
            accum_record(&fac_acc, dt);
            iter_fac = dt;
            if (bench == 0)
            {
                printf("Factors found by rank 0: %d distinct primes "
                       "(after %d attempt(s))\n",
                       local_factors.count, attempt + 1);
                fflush(stdout);
            }
        }

        auto t_loc0 = std::chrono::high_resolution_clock::now();

        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        compute_local_cbrt(local_cbrt, &local_factors, e_inv, modulo,
                           &local_cbrt_cache, &cbrt_cache_dirty);

        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(t_loc1 - t_loc0).count();

        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            accum_record(&loc_acc, max_local_dt);
            cur_max_loc = max_local_dt;
            iter_loc = max_local_dt;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        unsigned char *local_buf = (unsigned char *)calloc(cbrt_bytes, 1);
        {
            size_t written = 0;
            unsigned char *tmp = (unsigned char *)mpz_export(
                NULL, &written, 1, 1, 0, 0, local_cbrt);
            if (tmp && written > 0 && written <= cbrt_bytes)
            {
                memcpy(local_buf + cbrt_bytes - written, tmp, written);
                free(tmp);
            }
        }
        mpz_clear(local_cbrt);

        unsigned char *all_bufs = NULL;
        if (rank == 0)
            all_bufs = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);

        auto t_gat0 = std::chrono::high_resolution_clock::now();
        MPI_Gather(local_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                   all_bufs, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                   0, MPI_COMM_WORLD);
        auto t_gat1 = std::chrono::high_resolution_clock::now();
        free(local_buf);

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(t_gat1 - t_gat0).count();
            accum_record(&gat_acc, dt);
            iter_gat = dt;
        }

        if (rank == 0)
        {
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

            if (attempt > 0)
            {
                int idx = attempt - 1;
                if (idx < BLIND_TABLE_SIZE)
                {
                    mpz_mul(result, result, blind_table[idx].r_inv);
                    mpz_mod(result, result, modulo);
                }
            }

            auto t_com1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t_com1 - t_com0).count();
            accum_record(&com_acc, dt);
            iter_com = dt;
        }

        if (rank == 0)
        {
            double recon_dt = cur_max_loc + iter_gat + iter_com;
            accum_record(&recon_acc, recon_dt);

            // Record raw metrics for JSON Arrays
            results.factor_times.push_back(iter_fac);
            results.local_cbrt_times.push_back(iter_loc);
            results.gather_times.push_back(iter_gat);
            results.combine_times.push_back(iter_com);
            results.recon_times.push_back(recon_dt);
            results.total_times.push_back(iter_fac + recon_dt);
            results.blinding_attempts.push_back(attempt + 1);
        }

        fl_free(&local_factors);
    }

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

        if (method == METHOD_COMPARE)
        {
            printf("\n--- Sequential powm baseline (rank 0, %d threads unused) ---\n",
#ifdef _OPENMP
                   omp_get_max_threads());
#else
                   0);
#endif
            mpz_t powm_result;
            mpz_init(powm_result);
            double powm_avg = run_powm_baseline(powm_result, challenge, modulo,
                                                bench_count, 30);
            mpz_clear(powm_result);

            results.powm_baseline_ms = powm_avg; // Record for JSON

            double recon_avg = accum_mean(&recon_acc, bench_count);
            double total_avg = (fac_acc.sum + recon_acc.sum) / bench_count;

            printf("\n--- Speedup vs sequential powm ---\n");
            printf("  powm vs reconstruction only:  %.2fx  "
                   "(parallel cbrt, no factoring cost)\n",
                   recon_avg > 0 ? powm_avg / recon_avg : 0.0);
            printf("  powm vs total (factor+recon): %.2fx  "
                   "(end-to-end latency)\n",
                   total_avg > 0 ? powm_avg / total_avg : 0.0);
            printf("\nNote: 'recon only' is the most direct comparison to the paper's\n"
                   "      Algorithm 2 speedup claim. Total includes factoring overhead\n"
                   "      which the paper treats as separate precomputation cost.\n");
        }

        // NEW: JSON Output Logic
        if (json_output_file)
        {
            char final_json_path[512];
            strncpy(final_json_path, json_output_file, sizeof(final_json_path) - 1);
            final_json_path[sizeof(final_json_path) - 1] = '\0';

            char *dot = strrchr(final_json_path, '.');
            if (dot && bench_count > 0)
            {
                char ext[16];
                strncpy(ext, dot, sizeof(ext));
                ext[sizeof(ext) - 1] = '\0';
                *dot = '\0';
                snprintf(final_json_path + strlen(final_json_path),
                         sizeof(final_json_path) - strlen(final_json_path),
                         "_%zu%s", results.challenge_bits, ext);
            }

            write_benchmark_json(final_json_path, results);
        }

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

    if (cbrt_cache_dirty)
        kf_cbrt_cache_save(&local_cbrt_cache, cbrt_cache_path);
    kf_cbrt_cache_clear(&local_cbrt_cache);

    mpz_clear(e_inv);
    mpz_clears(challenge, modulo, result, NULL);
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

    solver_config_t cfg = {
        "data/challenges/k32/challenge_32_2048.txt",
        "data/challenges/k32/modulo_32_2048.txt",
        1, nullptr, false};

    const char *data_dir = "data/challenges/k32";
    const char *primes_file = "k32_primes.bin";
    int run_all = 0;
    int do_precompute = 0;
    run_method_t method = METHOD_SIEVE;

    int argi = 1;
    while (argi < argc)
    {
        if (strcmp(argv[argi], "--all") == 0)
        {
            run_all = 1;
            argi++;
        }
        else if (strcmp(argv[argi], "--data") == 0 && argi + 1 < argc)
        {
            data_dir = argv[++argi];
            argi++;
        }
        else if (strcmp(argv[argi], "--method") == 0 && argi + 1 < argc)
        {
            const char *m = argv[++argi];
            if (strcmp(m, "sieve") == 0) method = METHOD_SIEVE;
            else if (strcmp(m, "table") == 0) method = METHOD_TABLE;
            else if (strcmp(m, "compare") == 0) method = METHOD_COMPARE;
            else
            {
                if (rank == 0) fprintf(stderr, "Unknown --method '%s'\n", m);
                MPI_Finalize();
                return 1;
            }
            argi++;
        }
        else if (strcmp(argv[argi], "--primes") == 0 && argi + 1 < argc)
        {
            primes_file = argv[++argi];
            argi++;
        }
        else if (strcmp(argv[argi], "--precompute") == 0)
        {
            do_precompute = 1;
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

    if (!run_all && !do_precompute && (!cfg.challenge_file || !cfg.modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: k32_mpi <challenge> <modulo> [--method sieve|table] [--bench N]\n");
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
               method == METHOD_SIEVE ? "sieve" : method == METHOD_TABLE ? "table" : "compare");
        if (cfg.bench_count > 1)
            printf("Benchmark runs: %d\n", cfg.bench_count);
        if (run_all)
            printf("Mode:           --all\n");
        fflush(stdout);
    }

    int sp_count;
    unsigned *sp = kf_generate_small_primes(&sp_count);

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

    if (method == METHOD_TABLE)
    {
        int table_ok = 1;
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
    }

    if (run_all)
    {
        const char *sizes[] = {"1024", "2048", "5000", "7000"};
        for (int s = 0; s < 4; s++)
        {
            char cf[512], mf[512];
            snprintf(cf, sizeof(cf), "%s/challenge_32_%s.txt", data_dir, sizes[s]);
            snprintf(mf, sizeof(mf), "%s/modulo_32_%s.txt", data_dir, sizes[s]);
            solve_one(rank, nranks, cf, mf, cfg.bench_count,
                      sp_count, sp, my_lo, my_hi, method, primes_file, cfg.json_output_file);
        }
    }
    else
    {
        solve_one(rank, nranks, cfg.challenge_file, cfg.modulo_file, cfg.bench_count,
                  sp_count, sp, my_lo, my_hi, method, primes_file, cfg.json_output_file);
    }

    free(sp);
    MPI_Finalize();
    return 0;
}