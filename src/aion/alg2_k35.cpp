// =============================================================================
// alg2_k35.cpp — MPI+OpenMP hybrid cube root solver for 35-bit prime factors
// Extends k32 sieve range from [65537, 2^32] to [65537, 2^35].
// Matches the paper's concrete Algorithm 2 parameters: B = 2^35.
// No --method table: primes in (2^32, 2^35] exceed uint32_t storage.
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
//   k35_mpi [challenge] [modulo] [--method sieve|table]
//   k35_mpi --all [--data DIR] [--method sieve|table]
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

// Sieve bounds — file-scope so both solve_one() and main() see them.
// RANGE_LO = 65537: primes below this are handled by rank 0 separately.
// RANGE_HI = 2^35+1024: slightly above 2^35 to catch nextprime overflows.
static const uint64_t RANGE_LO = 65537ULL;
static const uint64_t RANGE_HI = (1ULL << 35) + 1024ULL;


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
// k35 factoring strategy — sieve with small-prime prefix + optional two-slice
// table (uint32 lower range + uint64 upper range above 2^32).
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t my_lo, my_hi;
    int sp_count;
    const unsigned *sp;
    run_method_t method;
    uint32_t *k32_slice;
    uint64_t k32_slice_count;
    uint64_t *k35u_slice;
    int64_t  k35u_slice_count;
    const kf_delta_slice_t *k32_delta;  // non-null => delta table, lower range
    const kf_delta_slice_t *k35u_delta; // non-null => delta table, upper range
} k35_ctx_t;

static void k35_factor_fn(
    kfactor_list_t *out, const mpz_t blinded,
    int bench, benchmark_results_t *json,
    void *ctx, int rank, int nranks)
{
    k35_ctx_t *c = (k35_ctx_t *)ctx;

    // Small primes below sieve floor (RANGE_LO=65537) — rank 0 only.
    // k35 challenges may contain factors < 65537 from the nextprime step.
    if (rank == 0)
    {
        for (int si = 0; si < c->sp_count && (uint64_t)c->sp[si] < RANGE_LO; si++)
        {
            unsigned long p = (unsigned long)c->sp[si];
            if (mpz_divisible_ui_p(blinded, p))
            {
                mpz_t tmp_c;
                mpz_init_set(tmp_c, blinded);
                int e = 0;
                while (mpz_divisible_ui_p(tmp_c, p))
                {
                    mpz_divexact_ui(tmp_c, tmp_c, p);
                    e++;
                }
                mpz_clear(tmp_c);
                kfl_append_ul(out, p, e);
            }
        }
    }

    if (c->method == METHOD_TABLE)
    {
        // lower range [65537, 2^32]
        if (c->k32_delta)
            kf_trial_divide_delta(blinded, c->k32_delta, out);
        else if (c->k32_slice && c->k32_slice_count > 0)
            kf_trial_divide_table(blinded, c->k32_slice, c->k32_slice_count, out);
        // upper range (2^32, 2^35]
        if (c->k35u_delta)
            kf_trial_divide_delta(blinded, c->k35u_delta, out);
        else if (c->k35u_slice && c->k35u_slice_count > 0)
            kf_trial_divide_table_u64(blinded, c->k35u_slice,
                                       c->k35u_slice_count, out);
    }
    else // METHOD_SIEVE
    {
        int nT = 1;
#ifdef _OPENMP
        nT = omp_get_max_threads();
#endif
        bool do_thr = (bench == 1 && rank == 0);
        std::vector<int64_t> thr_tested(do_thr ? nT : 0, 0);
        std::vector<int>     thr_found (do_thr ? nT : 0, 0);
        kf_trial_divide_sieve(blinded, c->my_lo, c->my_hi,
                              c->sp_count, c->sp, out,
                              do_thr ? thr_tested.data() : nullptr,
                              do_thr ? thr_found.data()  : nullptr,
                              nT);
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
// solve_one
// ---------------------------------------------------------------------------

// Derive a delta table path from a flat path: <name>.bin -> <name>.dlt
static void k35_delta_path(char *dst, size_t n, const char *flat)
{
    snprintf(dst, n, "%s", flat);
    size_t L = strlen(dst);
    if (L > 4 && strcmp(dst + L - 4, ".bin") == 0)
        snprintf(dst + L - 4, 5, ".dlt");
    else if (L + 5 <= n)
        snprintf(dst + L, 5, ".dlt");
}

static benchmark_results_t run_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method,
    const char *primes_upper_file,
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

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    mpz_t e_inv;
    mpz_init(e_inv);
    ksc_compute_e_inv(e_inv, modulo, exponent);

    // cbrt cache (M1, 2026-07-16): prefer the merged per-variant cache
    // <cbrt_dir>/cbrt_<size>.cbrt (loaded read-only by every rank); fall back to
    // the per-rank build only if it is absent (a brand-new modulus).
    char cbrt_cache_path[KSC_CBRT_PATH_MAX];
    ksc_make_cbrt_cache_path(cbrt_cache_path, sizeof(cbrt_cache_path),
                             cbrt_dir, modulo_file, ".cbrt35", rank);  // per-rank fallback

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
                NULL, 0,      // k35 always uses sieve for precompute pass
                true, rank);  // check sp[] below 65537 on rank 0
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
    const char *mname35 = (method == METHOD_TABLE)  ? "table"
                          : (method == METHOD_POWM) ? "powm"
                                                    : "sieve";
    ksc_init_json_results(&json_results, 35, mname35,
                          nranks, n_threads_json, challenge, rank);
    json_results.config.algorithm = "k35";
    json_results.config.exponent  = exponent;

    // Load k35 prime slices for --method table.  The upper file's magic selects
    // the format for BOTH ranges: main passes either the flat .bin or the delta
    // .dlt upper path, and the lower range follows suit (its .dlt sibling is
    // derived here).  This keeps flat and delta from ever mixing.
    uint32_t *k32_slice = nullptr;
    uint64_t  k32_slice_count = 0;
    uint64_t *k35u_slice = nullptr;
    int64_t   k35u_slice_count = 0;
    kf_delta_slice_t k32_delta, k35u_delta;
    memset(&k32_delta, 0, sizeof(k32_delta));
    memset(&k35u_delta, 0, sizeof(k35u_delta));
    bool have_k32_delta = false, have_k35u_delta = false;
    if (method == METHOD_TABLE && primes_upper_file)
    {
        bool is_delta = false;
        FILE *uf = fopen(primes_upper_file, "rb");
        if (uf)
        {
            uint64_t m = 0;
            if (fread(&m, 8, 1, uf) == 1 &&
                (m == KF_DELTA_MAGIC_U32 || m == KF_DELTA_MAGIC_U64))
                is_delta = true;
            fclose(uf);
        }
        const char *k32_path = ksc_ensure_k32_primes(sp_count, sp, rank);
        if (is_delta)
        {
            char k32_dlt[512];
            k35_delta_path(k32_dlt, sizeof(k32_dlt), k32_path);
            if (my_lo <= (uint64_t)UINT32_MAX)
            {
                uint64_t k32_hi = std::min(my_hi, (uint64_t)UINT32_MAX);
                if (!kf_load_delta_slice(&k32_delta, k32_dlt, my_lo, k32_hi))
                {
                    fprintf(stderr, "[rank %d] Failed to load k32 delta slice %s\n",
                            rank, k32_dlt);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                have_k32_delta = true;
            }
            if (my_hi > (uint64_t)UINT32_MAX)
            {
                uint64_t k35u_lo = std::max(my_lo, (uint64_t)UINT32_MAX + 1);
                if (!kf_load_delta_slice(&k35u_delta, primes_upper_file, k35u_lo, my_hi))
                {
                    fprintf(stderr, "[rank %d] Failed to load k35u delta slice %s\n",
                            rank, primes_upper_file);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                have_k35u_delta = true;
            }
        }
        else
        {
            if (my_lo <= (uint64_t)UINT32_MAX)
            {
                uint64_t k32_hi = std::min(my_hi, (uint64_t)UINT32_MAX);
                k32_slice = kf_load_prime_slice(&k32_slice_count, k32_path,
                                                (uint32_t)my_lo, (uint32_t)k32_hi);
            }
            if (my_hi > (uint64_t)UINT32_MAX)
            {
                uint64_t k35u_lo = std::max(my_lo, (uint64_t)UINT32_MAX + 1);
                k35u_slice = kf_load_prime_slice_u64(&k35u_slice_count,
                                                      primes_upper_file,
                                                      k35u_lo, my_hi);
            }
        }
    }

    // ---- Strategy + shared bench loop ----
    k35_ctx_t k35_ctx = { my_lo, my_hi, sp_count, sp, method,
                           k32_slice, k32_slice_count,
                           k35u_slice, k35u_slice_count,
                           have_k32_delta ? &k32_delta : nullptr,
                           have_k35u_delta ? &k35u_delta : nullptr };
    ksc_factor_strategy_t strategy = { k35_factor_fn, NULL, false, &k35_ctx };

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
        // For METHOD_POWM the bench loop is skipped; compute the result once.
        if (method == METHOD_POWM)
            run_powm_baseline(result, challenge, modulo, 1, 0, exponent);

        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        // Diagnostic: verify result^e mod M
        {
            mpz_t cube;
            mpz_init(cube);
            mpz_powm_ui(cube, result, exponent, modulo);
            if (mpz_cmp(cube, challenge_mod) != 0)
            {
                printf("DEBUG: result^%lu mod M = ", exponent);
                mpz_out_str(stdout, 16, cube);
                printf("\nDEBUG: challenge mod M = ");
                mpz_out_str(stdout, 16, challenge_mod);
                printf("\nDEBUG: result = ");
                mpz_out_str(stdout, 16, result);
                printf("\n");
                fflush(stdout);
            }
            mpz_clear(cube);
        }
        verify_eth_root(result, challenge_mod, modulo, exponent);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    if (rank == 0)
        kf_blinding_table_free(blind_table, BLIND_TABLE_SIZE);
    if (!used_shared && cbrt_cache_dirty)   // merged cache is read-only
        kf_cbrt_cache_save(&local_cbrt_cache, cbrt_cache_path);
    free(k32_slice);
    free(k35u_slice);
    if (have_k32_delta)
        kf_delta_slice_free(&k32_delta);
    if (have_k35u_delta)
        kf_delta_slice_free(&k35u_delta);
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
    run_method_t method,
    const char *primes_upper_file,
    unsigned long exponent,
    const char *json_output_file = nullptr,
    const char *cbrt_dir = KSC_CBRT_DIR,
    bool skip_powm = false)
{
    benchmark_results_t r = run_one(rank, nranks, challenge_file, modulo_file,
                                    sp_count, sp, my_lo, my_hi,
                                    method, primes_upper_file, exponent, cbrt_dir, skip_powm);
    if (rank == 0 && json_output_file && json_output_file[0] != '\0')
        write_benchmark_json(json_output_file, r);
    return 1;
}

// --table-format delta: ensure both delta tables exist (rank 0 builds them from
// the flat tables: k32 813 MB -> ~204 MB, k35 upper 9.6 GB -> ~1.2 GB), then
// return the UPPER delta path.  run_one keys the lower range off the upper
// file's magic and derives the lower .dlt itself, so the two never mix.
static const char *k35_ensure_delta(int sp_count, const unsigned *sp, int rank)
{
    const char *k32_flat = ksc_ensure_k32_primes(sp_count, sp, rank);
    const char *k35u_flat = ksc_ensure_k35_primes(sp_count, sp, rank);
    static char k32_dlt[512], k35u_dlt[512];
    k35_delta_path(k32_dlt, sizeof(k32_dlt), k32_flat);
    k35_delta_path(k35u_dlt, sizeof(k35u_dlt), k35u_flat);
    if (rank == 0)
    {
        FILE *t;
        if ((t = fopen(k32_dlt, "rb"))) fclose(t);
        else if (!kf_convert_prime_table_to_delta(k32_flat, k32_dlt, "[rank 0] "))
            MPI_Abort(MPI_COMM_WORLD, 1);
        if ((t = fopen(k35u_dlt, "rb"))) fclose(t);
        else if (!kf_convert_prime_table_to_delta_u64(k35u_flat, k35u_dlt, "[rank 0] "))
            MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD); // all ranks wait for both .dlt files
    return k35u_dlt;
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
    const char *data_dir = "data/challenges/k35";
    const char *json_file = NULL;
    const char *modulo_suffix = "";
    const char *cbrt_dir = KSC_CBRT_DIR;
    int run_all = 0;
    int do_convert_delta = 0;   // --convert-delta: build both .dlt tables and exit
    bool use_delta = false;     // --table-format delta: use delta-encoded tables
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
                    fprintf(stderr, "Unknown --method '%s' (use sieve|table|powm)\n", m);
                MPI_Finalize();
                return 1;
            }
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
        else if (strcmp(argv[argi], "--convert-delta") == 0)
        {
            do_convert_delta = 1;
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
            if (n_challenges < 1) n_challenges = 1;
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

    if (!run_all && !do_convert_delta && (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: k35_mpi <challenge> <modulo> [--method sieve|table] [--table-format flat|delta]\n"
                    "       k35_mpi --all [--data DIR] [--method sieve|table] [--table-format flat|delta]\n"
                    "       k35_mpi --convert-delta   (build k32_primes.dlt + k35_upper_primes.dlt, then exit)\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("=== k35_mpi (B=2^35, paper Algorithm 2) ===\n");
        printf("MPI ranks:      %d\n", nranks);
#ifdef _OPENMP
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
#endif
        printf("Method:         %s\n",
               method == METHOD_TABLE   ? "table"
               : method == METHOD_SIEVE ? "sieve"
                                        : "powm");
        if (method == METHOD_TABLE)
            printf("Table format:   %s\n",
                   use_delta ? "delta (1 byte/prime: k32 ~204 MB, k35u ~1.2 GB)"
                             : "flat (k32 ~813 MB, k35u ~9.6 GB)");
        if (run_all)
            printf("Mode:           --all\n");
        if (n_challenges > 1)
            printf("Challenges:     %d  (all timed warm runs, 0..%d averaged)\n",
                   n_challenges, n_challenges - 1);
        fflush(stdout);
    }

    // Small primes — every rank generates these independently (~0.1 ms, no Bcast)
    // k35: need small primes up to sqrt(2^35) ~ 2^17.5 -> generate up to 2^18
    int sp_count = 0;
    unsigned long *sp_ul = generate_primes_up_to(1UL << 18, &sp_count);
    unsigned *sp = (unsigned *)malloc((size_t)sp_count * sizeof(unsigned));
    for (int i = 0; i < sp_count; i++)
        sp[i] = (unsigned)sp_ul[i];
    free(sp_ul);

    // Partition [65537, 2^32] evenly across MPI ranks
    // RANGE_LO is a file-scope constant (defined above solve_one).
    // RANGE_HI extends slightly above 2^35 to catch primes that mpz_nextprime
    // places just above 2^35 (prime gap ≈ ln(2^35) ≈ 24; 1024 is a safe margin).
    // The new challenges.c fixes this at source, but existing challenge files
    // on disk may contain such primes, so the sieve covers them here too.
    const uint64_t RANGE_HI = (1ULL << 35) + 1024; // B ≈ 2^35 (paper Algorithm 2)
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

    // --convert-delta: rank 0 builds both delta tables from the flat tables, exit.
    if (do_convert_delta)
    {
        k35_ensure_delta(sp_count, sp, rank);
        free(sp);
        MPI_Finalize();
        return 0;
    }

    if (run_all)
    {
        const char *sizes[] = { KSC_CANONICAL_SIZES_LIST };
        const char *k35u_p2 = (method != METHOD_TABLE) ? nullptr
                              : use_delta ? k35_ensure_delta(sp_count, sp, rank)
                                          : ksc_ensure_k35_primes(sp_count, sp, rank);
        for (int s = 0; s < KSC_N_CANONICAL_SIZES; s++)
        {
            char cf[512], mf[512], jf[768];
            snprintf(mf, sizeof(mf), "%s/modulo_35_%s%s.txt", data_dir, sizes[s], modulo_suffix);
            const char *jf_ptr = NULL;
            if (json_file)
            {
                snprintf(jf, sizeof(jf), "%s_%s.json", json_file, sizes[s]);
                jf_ptr = jf;
            }
            if (n_challenges == 1)
            {
                snprintf(cf, sizeof(cf), "%s/challenge_0_35_%s.txt", data_dir, sizes[s]);
                solve_one(rank, nranks, cf, mf,
                          sp_count, sp, my_lo, my_hi, method, k35u_p2, exponent,
                          jf_ptr, cbrt_dir, skip_powm);
            }
            else
            {
                // Plan A (2026-07-17): every challenge 0..n_challenges-1 is a timed
                // warm run — no separate cold run (merged cbrt cache makes
                // challenge_0 statistically identical; cbrt_misses=0 for all runs).
                benchmark_results_t combined;
                bool has_data = false;
                for (int ci = 0; ci < n_challenges; ci++)
                {
                    char cfi[512];
                    snprintf(cfi, sizeof(cfi), "%s/challenge_%d_35_%s.txt",
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
                                                    sp_count, sp, my_lo, my_hi,
                                                    method, k35u_p2, exponent,
                                                    cbrt_dir, skip_powm);
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
        const char *k35u_path = (method != METHOD_TABLE) ? nullptr
                                : use_delta ? k35_ensure_delta(sp_count, sp, rank)
                                            : ksc_ensure_k35_primes(sp_count, sp, rank);
        solve_one(rank, nranks, challenge_file, modulo_file,
                  sp_count, sp, my_lo, my_hi, method, k35u_path, exponent,
                  json_file, cbrt_dir, skip_powm);
    }

    free(sp);
    MPI_Finalize();
    return 0;
}


// Edit: 2026-07-22

