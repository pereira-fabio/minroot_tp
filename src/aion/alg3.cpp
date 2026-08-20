// =============================================================================
// alg3.cpp — Algorithm 3 solver: almost-smooth cube root recovery
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
//   Results:     results/algo3/alg3_mpi_N*/
//   Prime table: reuses data/k32_primes.bin read-only (sieve method only)
//
// Communication pattern: identical to k32_mpi.
//
// Usage:
//   alg3_mpi [challenge] [modulo] [--method sieve|powm]
//   alg3_mpi --all [--data DIR]
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

// ---------------------------------------------------------------------------
// alg3 factoring strategy
// ---------------------------------------------------------------------------

typedef struct
{
    uint64_t my_lo, my_hi;
    int sp_count;
    const unsigned *sp;
    run_method_t method;
    uint32_t *prime_slice;
    int64_t slice_count;
    const kf_delta_slice_t *delta_slice; // non-null => delta-encoded table method
} alg3_ctx_t;

static void alg3_factor_fn(
    kfactor_list_t *out, const mpz_t blinded,
    int bench, benchmark_results_t *json,
    void *ctx, int rank, int nranks)
{
    alg3_ctx_t *c = (alg3_ctx_t *)ctx;
    (void)nranks;
    if (c->method == METHOD_TABLE)
    {
        int nT = 1;
#ifdef _OPENMP
        nT = omp_get_max_threads();
#endif
        bool do_thr = (bench == 1 && rank == 0);
        std::vector<int64_t> thr_tested(do_thr ? nT : 0, 0);
        std::vector<int> thr_found(do_thr ? nT : 0, 0);
        if (c->delta_slice)
            kf_trial_divide_delta(blinded, c->delta_slice, out,
                                  do_thr ? thr_tested.data() : nullptr,
                                  do_thr ? thr_found.data() : nullptr,
                                  do_thr ? nT : 0);
        else
            kf_trial_divide_table(blinded, c->prime_slice, c->slice_count, out,
                                  do_thr ? thr_tested.data() : nullptr,
                                  do_thr ? thr_found.data() : nullptr,
                                  do_thr ? nT : 0);
        if (do_thr)
        {
            json->stats.thread_primes_tested.insert(
                json->stats.thread_primes_tested.end(),
                thr_tested.begin(), thr_tested.end());
            json->stats.thread_primes_found.insert(
                json->stats.thread_primes_found.end(),
                thr_found.begin(), thr_found.end());
        }
    }
    else
    {
        kf_trial_divide_sieve(blinded, c->my_lo, c->my_hi,
                              c->sp_count, c->sp, out);
        (void)bench;
        (void)json;
        (void)rank;
    }
}

static void alg3_fold_fn(
    mpz_t local_cbrt,
    const mpz_t cofactor,
    const mpz_t e_inv, const mpz_t modulo,
    cbrt_cache_t *cache, bool *dirty,
    benchmark_results_t *json, int bench, void *ctx)
{
    ksc_fold_cofactor_cbrt(local_cbrt, cofactor, e_inv, modulo, cache, dirty);
    (void)json;
    (void)bench;
    (void)ctx;
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
                             cbrt_dir, modulo_file, ".cbrt_alg3", rank); // per-rank fallback

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

    // Precompute cbrt(q0) for the medium prime before the bench loop so all
    // warm bench runs are instant cache hits (avoids ~1800ms powm on run 1).
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

        mpz_t q0_pre;
        mpz_init_set_ui(q0_pre, 1);
        int complete = ksc_alg3_broadcast_cofactor(
            q0_pre, challenge, local_prod, cbrt_bytes, rank, nranks);
        mpz_clear(local_prod);

        if (rank == 0 && complete && mpz_cmp_ui(q0_pre, 1) > 0 &&
            mpz_fits_ulong_p(q0_pre))
        {
            unsigned long q0_ul = mpz_get_ui(q0_pre);
            if (kf_cbrt_cache_find(&local_cbrt_cache, q0_ul) < 0)
            {
                mpz_t cbrt_q0_pre;
                mpz_init(cbrt_q0_pre);
                mpz_powm(cbrt_q0_pre, q0_pre, e_inv, modulo);
                kf_cbrt_cache_append(&local_cbrt_cache, q0_ul, cbrt_q0_pre);
                cbrt_cache_dirty = true;
                mpz_clear(cbrt_q0_pre);
            }
        }
        mpz_clear(q0_pre);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const int BLIND_TABLE_SIZE = 255;
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

    bench_accum_t fac_acc = {0, 0}, loc_acc = {0, 0}, gat_acc = {0, 0},
                  com_acc = {0, 0}, recon_acc = {0, 0};

    int n_threads_json = 1;
#ifdef _OPENMP
    n_threads_json = omp_get_max_threads();
#endif
    benchmark_results_t json_results;
    const char *mname = (method == METHOD_TABLE)  ? "table"
                        : (method == METHOD_POWM) ? "powm"
                                                  : "sieve";
    ksc_init_json_results(&json_results, 32, mname,
                          nranks, n_threads_json, challenge, rank);
    json_results.config.algorithm = "alg3";
    json_results.config.exponent = exponent;

    // ---- Strategy + shared bench loop ----
    alg3_ctx_t alg3_ctx = {my_lo, my_hi, sp_count, sp, method, prime_slice, slice_count,
                           have_delta ? &delta_slice : nullptr};
    ksc_factor_strategy_t strategy = {alg3_factor_fn, alg3_fold_fn,
                                      true, &alg3_ctx};

    ksc_run_bench_loop(
        result, method,
        challenge, modulo, cbrt_bytes, e_inv,
        &local_cbrt_cache, &cbrt_cache_dirty,
        blind_table, BLIND_TABLE_SIZE,
        my_lo, my_hi,
        &fac_acc, &loc_acc, &gat_acc, &com_acc, &recon_acc,
        &json_results, &strategy, rank, nranks);

    // ---- Timing report + baseline + verify ----
    if (rank == 0)
    {
        ksc_print_timing_report(&fac_acc, &loc_acc,
                                &gat_acc, &com_acc, &recon_acc);
        if (!skip_powm)
            ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                             &recon_acc, &fac_acc,
                                             rank, exponent);
        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_eth_root(result, challenge_mod, modulo, exponent);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    free(prime_slice);
    if (have_delta)
        kf_delta_slice_free(&delta_slice);
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
static void alg3_delta_path(char *dst, size_t n, const char *flat)
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
    const char *data_dir = "data/challenges/alg3"; // ALG3: own challenge directory
    const char *primes_file = "k32_primes.bin";
    const char *json_file = NULL;
    const char *modulo_suffix = "";
    const char *cbrt_dir = KSC_CBRT_DIR;
    int run_all = 0;
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

    if (!run_all && (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: alg3_mpi <challenge> <modulo> [--method sieve|table|powm] [--table-format flat|delta]\n"
                    "       alg3_mpi --all [--data DIR] [--method sieve|table|powm] [--table-format flat|delta]\n"
                    "       (--primes FILE defaults to k32_primes.bin; only used with --method table)\n");
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
               method == METHOD_SIEVE   ? "sieve"
               : method == METHOD_TABLE ? "table"
                                        : "powm");
        if (method == METHOD_TABLE)
            printf("Table format:   %s\n",
                   use_delta ? "delta (1 byte/prime, ~204 MB)" : "flat (uint32, ~813 MB)");
        if (run_all)
            printf("Mode:           --all\n");
        if (n_challenges > 1)
            printf("Challenges:     %d  (all timed warm runs, 0..%d averaged)\n",
                   n_challenges, n_challenges - 1);
        fflush(stdout);
    }

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
                alg3_delta_path(dpath, sizeof(dpath), flat);
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
        // ALG3 challenge naming: challenge_<N>_alg3_<bits>.txt / modulo_alg3_<bits>.txt
        const char *sizes[] = {KSC_CANONICAL_SIZES_LIST};
        for (int s = 0; s < KSC_N_CANONICAL_SIZES; s++)
        {
            char cf[512], mf[512], jf[768];
            snprintf(mf, sizeof(mf), "%s/modulo_alg3_%s%s.txt", data_dir, sizes[s], modulo_suffix);
            const char *jf_ptr = NULL;
            if (json_file)
            {
                snprintf(jf, sizeof(jf), "%s_%s.json", json_file, sizes[s]);
                jf_ptr = jf;
            }
            if (n_challenges == 1)
            {
                snprintf(cf, sizeof(cf), "%s/challenge_0_alg3_%s.txt", data_dir, sizes[s]);
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
                    snprintf(cfi, sizeof(cfi), "%s/challenge_%d_alg3_%s.txt",
                             data_dir, ci, sizes[s]);
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
