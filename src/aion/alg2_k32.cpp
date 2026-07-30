// =============================================================================
// alg2_k32.cpp — MPI+OpenMP hybrid cube root solver for 32-bit prime factors
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
//           [--precompute]
//   k32_mpi --all [--data DIR] [--method sieve|table] [--primes FILE]
//
//
// Prime table generation (rank 0 only, one-time, ~10 s):
//   srun ... k32_mpi --precompute [--primes k32_primes.bin]
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
// Small prime generator (same as k32.cpp)
// ---------------------------------------------------------------------------

// kf_kf_generate_small_primes() — in factorization.cpp

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

// kf_generate_and_save_prime_table() — in factorization.cpp

// Binary search: index of first prime >= target in the sorted uint32_t file.
// Primes start at byte offset 16 (after magic + count header).
// prime_file_lower_bound() — internal to kf_load_prime_slice()

// Load only the slice [range_lo, range_hi] of the prime table.
// Returns a malloc'd uint32_t array; sets *slice_count.
// Returns NULL on failure (file not found or I/O error).
// kf_load_prime_slice() — in factorization.cpp

// kf_kf_trial_divide_sieve() — in factorization.cpp

// kf_kf_trial_divide_table() — in factorization.cpp

// Timing report: print_timing_stat_moments() from common.h at width 30.
// kprint_stat() defined in common.h

// ---------------------------------------------------------------------------
// k32 factoring strategy
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
} k32_ctx_t;

static void k32_factor_fn(
    kfactor_list_t *out, const mpz_t blinded,
    int bench, benchmark_results_t *json,
    void *ctx, int rank, int nranks)
{
    k32_ctx_t *c = (k32_ctx_t *)ctx;
    if (c->method == METHOD_SIEVE)
    {
        kf_trial_divide_sieve(blinded, c->my_lo, c->my_hi,
                              c->sp_count, c->sp, out);
    }
    else
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

    // Load prime slice for table method (once, outside bench loop).
    // The table file may be flat (k32_primes.bin) or delta-encoded
    // (k32_primes.dlt) — auto-detect by magic so no extra flag is threaded here.
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

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    mpz_t e_inv;
    mpz_init(e_inv);
    ksc_compute_e_inv(e_inv, modulo, exponent);

    // cbrt cache (M1, 2026-07-16): prefer the merged per-variant cache
    // <cbrt_dir>/cbrt_<size>.cbrt — every rank loads the full set once, so all
    // reconstruction lookups hit (no per-rank build, no cold-run warming).  If
    // it is absent (a brand-new modulus), fall back to the per-rank build, which
    // analysis/merge_cbrt_cache.py later consolidates.
    char cbrt_cache_path[KSC_CBRT_PATH_MAX];
    ksc_make_cbrt_cache_path(cbrt_cache_path, sizeof(cbrt_cache_path),
                             cbrt_dir, modulo_file, ".cbrt", rank); // per-rank fallback

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
                prime_slice, slice_count, false, rank);
    }

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
    const char *mname32 = (method == METHOD_TABLE)  ? "table"
                          : (method == METHOD_POWM) ? "powm"
                                                    : "sieve";
    ksc_init_json_results(&json_results, 32, mname32,
                          nranks, n_threads_json, challenge, rank);
    json_results.config.algorithm = "k32";
    json_results.config.exponent = exponent;

    // ---- Strategy + shared bench loop ----
    k32_ctx_t k32_ctx = {my_lo, my_hi, sp_count, sp, method,
                         prime_slice, slice_count,
                         have_delta ? &delta_slice : NULL};
    ksc_factor_strategy_t strategy = {k32_factor_fn, NULL, false, &k32_ctx};

    // Reset AFTER cache setup (ksc_ensure_cbrt_cache does its own lookups) so the
    // counters describe the benchmark loop only — advisor request 2026-07-29.
    kf_cbrt_stats_reset();
    kf_recon_prof_reset();

    ksc_run_bench_loop(
        result, method,
        challenge, modulo, cbrt_bytes, e_inv,
        &local_cbrt_cache, &cbrt_cache_dirty,
        blind_table, BLIND_TABLE_SIZE,
        my_lo, my_hi,
        &fac_acc, &loc_acc, &gat_acc, &com_acc, &recon_acc,
        &json_results, &strategy, rank, nranks);

    // ---- cbrt cache reads per core, reduced across ranks ----
    // Each rank owns a private cache slice, so lookups are rank-local; SUM gives
    // the cluster-wide read count and MAX exposes imbalance between ranks. The
    // per-thread split within a rank is printed by rank 0 only, as a sample.
    {
        long my_h = 0, my_m = 0, my_p = 0;
        kf_cbrt_stats_total(&my_h, &my_m, &my_p);
        int my_threads = kf_cbrt_stats_threads_used();

        long mine[4] = {my_h, my_m, my_p, (long)my_threads};
        long sum[4], mx[4];
        MPI_Reduce(mine, sum, 4, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(mine, mx, 4, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            long lookups = sum[0] + sum[1];
            long cores = sum[3];
            printf("\n--- cbrt cache reads (%d ranks, %ld cores active, "
                   "|M|=%zu bits) ---\n",
                   nranks, cores, mpz_sizeinbase(modulo, 2));
            printf("  %-22s %14s %14s\n", "", "sum(all ranks)", "max(one rank)");
            printf("  %-22s %14ld %14ld\n", "lookups", lookups, mx[0] + mx[1]);
            printf("  %-22s %14ld %14ld\n", "hits", sum[0], mx[0]);
            printf("  %-22s %14ld %14ld\n", "misses", sum[1], mx[1]);
            printf("  %-22s %14ld %14ld\n", "probes", sum[2], mx[2]);
            if (lookups > 0)
                printf("  hit rate %.2f%%   probes/lookup %.1f   lookups/core %.1f\n",
                       100.0 * (double)sum[0] / (double)lookups,
                       (double)sum[2] / (double)lookups,
                       cores > 0 ? (double)lookups / (double)cores : 0.0);
            printf("CBRTSTAT|%zu|%d|ALL|%ld|%ld|%ld|%ld\n",
                   mpz_sizeinbase(modulo, 2), nranks, sum[0], sum[1], sum[2], cores);

            const kf_recon_prof_t *rp = kf_recon_prof_get();
            if (rp->calls > 0)
                printf("RECONPROF|%zu|%d|%.4f|%.4f|%ld|%ld\n",
                       mpz_sizeinbase(modulo, 2), nranks,
                       rp->parallel_ms, rp->serial_ms, rp->factors, rp->calls);
            fflush(stdout);
        }
    }

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
    // Only the per-rank fallback build is persisted; the merged shared cache is
    // read-only (never written back).
    if (!used_shared && cbrt_cache_dirty)
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
static void k32_delta_path(char *dst, size_t n, const char *flat)
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
    const char *modulo_suffix = "";
    const char *cbrt_dir = KSC_CBRT_DIR;
    int run_all = 0;
    int do_precompute = 0;
    int do_convert_delta = 0; // --convert-delta: build k32_primes.dlt and exit
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
                    fprintf(stderr, "Unknown --method '%s' (use sieve, table, or powm)\n", m);
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
        else if (strcmp(argv[argi], "--precompute") == 0)
        {
            do_precompute = 1;
        }
        else if (strcmp(argv[argi], "--convert-delta") == 0)
        {
            do_convert_delta = 1;
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

    if (!run_all && !do_precompute && !do_convert_delta &&
        (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: k32_mpi <challenge> <modulo> [--method sieve|table] [--table-format flat|delta]\n"
                    "       k32_mpi --all [--data DIR] [--method sieve|table] [--table-format flat|delta]\n"
                    "       k32_mpi --precompute [--primes FILE]\n"
                    "       k32_mpi --convert-delta   (build k32_primes.dlt from k32_primes.bin, then exit)\n");
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

    // --convert-delta: rank 0 builds the delta table (k32_primes.dlt) from the
    // flat table (~813 MB -> ~204 MB, 1 byte/prime), then all ranks exit.
    if (do_convert_delta)
    {
        if (rank == 0)
        {
            const char *flat = ksc_ensure_k32_primes(sp_count, sp, rank);
            char dpath[512];
            if (primes_user_set)
                snprintf(dpath, sizeof(dpath), "%s", primes_file);
            else
                k32_delta_path(dpath, sizeof(dpath), flat);
            if (!kf_convert_prime_table_to_delta(flat, dpath, "[rank 0] "))
            {
                free(sp);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
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
        const char *flat = ksc_ensure_k32_primes(sp_count, sp, rank);
        if (use_delta)
        {
            // Point at the delta table (k32_primes.dlt); rank 0 builds it from
            // the flat table on first use.  run_one auto-detects it by magic.
            static char dpath[512];
            if (primes_user_set)
                snprintf(dpath, sizeof(dpath), "%s", primes_file);
            else
                k32_delta_path(dpath, sizeof(dpath), flat);
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

    if (run_all)
    {
        // Run all 11 canonical challenge sizes in one MPI execution.
        // Avoids multiple srun calls in the SLURM script (causes reinit hangs).
        const char *sizes[] = {KSC_CANONICAL_SIZES_LIST};
        for (int s = 0; s < KSC_N_CANONICAL_SIZES; s++)
        {
            char cf[512], mf[512], jf[768];
            snprintf(mf, sizeof(mf), "%s/modulo_32_%s%s.txt", data_dir, sizes[s], modulo_suffix);
            const char *jf_ptr = NULL;
            if (json_file)
            {
                snprintf(jf, sizeof(jf), "%s_%s.json", json_file, sizes[s]);
                jf_ptr = jf;
            }
            if (n_challenges == 1)
            {
                snprintf(cf, sizeof(cf), "%s/challenge_0_32_%s.txt", data_dir, sizes[s]);
                solve_one(rank, nranks, cf, mf,
                          sp_count, sp, my_lo, my_hi, method, primes_file, exponent,
                          jf_ptr, cbrt_dir, skip_powm);
            }
            else
            {
                // Plan A (2026-07-17): every challenge 0..n_challenges-1 is a timed
                // warm run — no separate cold run.  The merged cbrt cache (M1) makes
                // challenge_0 statistically identical to the rest (cbrt_misses=0 for
                // all runs), so there is nothing to warm and no cold_run section.
                benchmark_results_t combined;
                bool has_data = false;
                for (int ci = 0; ci < n_challenges; ci++)
                {
                    char cfi[512];
                    snprintf(cfi, sizeof(cfi), "%s/challenge_%d_32_%s.txt",
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
// Edit: 2026-07-30
