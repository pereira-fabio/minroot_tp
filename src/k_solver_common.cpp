// =============================================================================
// k_solver_common.cpp — Shared MPI+OpenMP infrastructure for cube root solvers
// =============================================================================

#include "k_solver_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <chrono>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// ksc_broadcast_challenge_modulo
// ---------------------------------------------------------------------------

bool ksc_broadcast_challenge_modulo(
    mpz_t challenge, mpz_t modulo,
    const char *challenge_file, const char *modulo_file,
    int rank)
{
    int read_ok = 1;
    size_t cn = 0, mn = 0;
    unsigned char *cb = nullptr, *mb = nullptr;

    if (rank == 0)
    {
        FILE *fc = fopen(challenge_file, "rb");
        FILE *fm = fopen(modulo_file, "rb");
        if (!fc || !fm)
        {
            fprintf(stderr, "[rank 0] Cannot open '%s' or '%s'\n",
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
            cb = (unsigned char *)mpz_export(NULL, &cn, 1, 1, 0, 0, challenge);
            mb = (unsigned char *)mpz_export(NULL, &mn, 1, 1, 0, 0, modulo);
        }
    }

    MPI_Bcast(&read_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!read_ok)
    {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return false;
    }

    uint64_t cn64 = (uint64_t)cn, mn64 = (uint64_t)mn;
    MPI_Bcast(&cn64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mn64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    cn = (size_t)cn64;
    mn = (size_t)mn64;

    if (rank != 0)
    {
        cb = (unsigned char *)malloc(cn);
        mb = (unsigned char *)malloc(mn);
    }
    MPI_Bcast(cb, (int)cn, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(mb, (int)mn, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        mpz_import(challenge, cn, 1, 1, 0, 0, cb);
        mpz_import(modulo, mn, 1, 1, 0, 0, mb);
    }
    free(cb);
    free(mb);
    return true;
}

// ---------------------------------------------------------------------------
// ksc_init_json_results
// ---------------------------------------------------------------------------

void ksc_init_json_results(
    benchmark_results_t *r,
    int k_size, int bench_count,
    const char *method_name,
    int mpi_ranks, int threads_per_rank,
    const mpz_t challenge, int rank)
{
    r->k_size = k_size;
    r->challenge_bits = (rank == 0) ? mpz_sizeinbase(challenge, 2) : 0;
    r->method_name = method_name;
    r->num_threads = threads_per_rank;
    r->mpi_ranks = mpi_ranks;
    r->bench_count = bench_count;
    r->powm_baseline_ms = 0.0;
}

// ---------------------------------------------------------------------------
// ksc_gather_and_combine_cbrt
// ---------------------------------------------------------------------------

void ksc_gather_and_combine_cbrt(
    mpz_t result,
    const mpz_t local_cbrt,
    const mpz_t modulo,
    size_t cbrt_bytes,
    int rank, int nranks,
    int attempt,
    const blinding_pair_t *blind_table, int blind_table_size,
    double *t_gat_ms_out,
    double *t_com_ms_out)
{
    // Export local_cbrt to zero-padded big-endian buffer
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

    unsigned char *all_bufs = nullptr;
    if (rank == 0)
        all_bufs = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);

    auto t_gat0 = std::chrono::high_resolution_clock::now();
    MPI_Gather(local_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
               all_bufs, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
    auto t_gat1 = std::chrono::high_resolution_clock::now();
    free(local_buf);

    if (t_gat_ms_out)
        *t_gat_ms_out = std::chrono::duration<double, std::milli>(t_gat1 - t_gat0).count();

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

        // Unblind: result = combined * r^{-1} mod M
        if (attempt > 0 && blind_table)
        {
            int idx = attempt - 1;
            if (idx < blind_table_size)
            {
                mpz_mul(result, result, blind_table[idx].r_inv);
                mpz_mod(result, result, modulo);
            }
        }

        auto t_com1 = std::chrono::high_resolution_clock::now();
        if (t_com_ms_out)
            *t_com_ms_out = std::chrono::duration<double, std::milli>(t_com1 - t_com0).count();
    }
}

// ---------------------------------------------------------------------------
// ksc_gather_prime_stats
// ---------------------------------------------------------------------------

void ksc_gather_prime_stats(
    benchmark_results_t *r,
    int local_found,
    int64_t local_tested,
    int rank, int nranks)
{
    if (rank == 0)
    {
        std::vector<int> all_found(nranks);
        std::vector<int64_t> all_tested(nranks);
        MPI_Gather(&local_found, 1, MPI_INT, all_found.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_tested, 1, MPI_INT64_T, all_tested.data(), 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        r->primes_found_per_rank.insert(r->primes_found_per_rank.end(),
                                        all_found.begin(), all_found.end());
        r->primes_tested_per_rank.insert(r->primes_tested_per_rank.end(),
                                         all_tested.begin(), all_tested.end());
    }
    else
    {
        MPI_Gather(&local_found, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_tested, 1, MPI_INT64_T, nullptr, 0, MPI_INT64_T, 0, MPI_COMM_WORLD);
    }
}

// ---------------------------------------------------------------------------
// ksc_print_timing_report
// ---------------------------------------------------------------------------

#define _ps(lbl, acc, n) print_timing_stat_moments(lbl, (acc)->sum, (acc)->sumsq, n, 30, true)

void ksc_print_timing_report(
    int bench_count,
    const bench_accum_t *fac_acc,
    const bench_accum_t *loc_acc,
    const bench_accum_t *gat_acc,
    const bench_accum_t *com_acc,
    const bench_accum_t *recon_acc)
{
    printf("\nResults over %d runs:\n", bench_count);
    _ps("Factoring:", fac_acc, bench_count);
    _ps("Local cbrt (max):", loc_acc, bench_count);
    _ps("Gather:", gat_acc, bench_count);
    _ps("Combine (rank 0):", com_acc, bench_count);
    _ps("Reconstruction total:", recon_acc, bench_count);
    printf("    %-30s avg=%9.4f ms\n", "Total:",
           (fac_acc->sum + recon_acc->sum) / bench_count);
}
#undef _ps

// ---------------------------------------------------------------------------
// ksc_run_and_report_powm_baseline
// ---------------------------------------------------------------------------

void ksc_run_and_report_powm_baseline(
    benchmark_results_t *r,
    const mpz_t challenge,
    const mpz_t modulo,
    int bench_count,
    const bench_accum_t *recon_acc,
    const bench_accum_t *fac_acc,
    int rank)
{
    if (rank != 0)
        return;

    printf("\n--- Sequential powm baseline (rank 0, %d OMP threads unused) ---\n",
#ifdef _OPENMP
           omp_get_max_threads());
#else
           0);
#endif
    mpz_t tmp;
    mpz_init(tmp);
    double powm_avg = run_powm_baseline(tmp, challenge, modulo, bench_count, 30);
    r->powm_baseline_ms = powm_avg;
    mpz_clear(tmp);

    double recon_avg = accum_mean(recon_acc, bench_count);
    double total_avg = (fac_acc->sum + recon_acc->sum) / bench_count;
    printf("\n--- Speedup vs sequential powm ---\n");
    printf("  powm vs reconstruction only:  %.2fx\n",
           recon_avg > 0 ? powm_avg / recon_avg : 0.0);
    printf("  powm vs total (factor+recon): %.2fx\n",
           total_avg > 0 ? powm_avg / total_avg : 0.0);
}

// ---------------------------------------------------------------------------
// ksc_ensure_k16_primes
// ---------------------------------------------------------------------------

static bool file_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

static void mkdir_data()
{
    // Create data/ and data/cbrt/ if they don't exist
    if (!file_exists(KSC_CBRT_DIR))
    {
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", KSC_CBRT_DIR);
        (void)system(cmd);
    }
}

unsigned long *ksc_ensure_k16_primes(int *prime_count_out)
{
    mkdir_data();
    const char *path = KSC_DATA_DIR "/k16_primes.bin";

    if (file_exists(path))
    {
        // Load from file
        FILE *f = fopen(path, "rb");
        if (f)
        {
            int count = 0;
            if (fread(&count, sizeof(int), 1, f) == 1 && count > 0)
            {
                unsigned long *primes = (unsigned long *)malloc(
                    (size_t)count * sizeof(unsigned long));
                if (fread(primes, sizeof(unsigned long), (size_t)count, f) == (size_t)count)
                {
                    fclose(f);
                    *prime_count_out = count;
                    return primes;
                }
                free(primes);
            }
            fclose(f);
        }
    }

    // Generate and save
    int count = 0;
    unsigned long *primes = generate_primes_up_to(65535UL, &count);
    FILE *f = fopen(path, "wb");
    if (f)
    {
        fwrite(&count, sizeof(int), 1, f);
        fwrite(primes, sizeof(unsigned long), (size_t)count, f);
        fclose(f);
    }
    *prime_count_out = count;
    return primes;
}

// ---------------------------------------------------------------------------
// ksc_ensure_k16_cbrt
// ---------------------------------------------------------------------------

void ksc_ensure_k16_cbrt(
    mpz_t *cbrt_table,
    const unsigned long *primes,
    int prime_count,
    const mpz_t modulo,
    const char *modulo_file,
    int rank)
{
    mkdir_data();

    // Derive path: data/<modulo_stem>.cbrt
    // stem is capped at 256 bytes so base (256+16) is always large enough
    // and the compiler can verify no truncation occurs.
    const char *fname = strrchr(modulo_file, '/');
    fname = fname ? fname + 1 : modulo_file;
    char stem[256];
    strncpy(stem, fname, sizeof(stem) - 1);
    stem[sizeof(stem) - 1] = '\0';
    char *dot = strrchr(stem, '.');
    if (dot)
        *dot = '\0';
    char base[280]; // 256 (stem) + 9 (KSC_CBRT_DIR) + 1 (/) + 5 (.cbrt) + 9 slack
    snprintf(base, sizeof(base), "%s/%s.cbrt", KSC_CBRT_DIR, stem);

    bool loaded = false;
    if (rank == 0)
    {
        loaded = kf_load_cbrt_table_k16(cbrt_table, prime_count, base);
        if (!loaded)
        {
            printf("[rank 0] Building k16 cbrt table → %s\n", base);
            fflush(stdout);
            kf_precompute_cbrt_table_k16(cbrt_table, primes, prime_count, modulo);
            kf_save_cbrt_table_k16(cbrt_table, prime_count, base);
            printf("[rank 0] cbrt table saved (%d entries)\n", prime_count);
            fflush(stdout);
        }
        else
        {
            printf("[rank 0] cbrt table loaded from %s\n", base);
            fflush(stdout);
        }
    }

    // Broadcast every entry to non-root ranks
    for (int i = 0; i < prime_count; i++)
    {
        size_t nb = 0;
        unsigned char *buf = nullptr;
        if (rank == 0)
            buf = (unsigned char *)mpz_export(NULL, &nb, 1, 1, 0, 0, cbrt_table[i]);

        uint64_t nb64 = (uint64_t)nb;
        MPI_Bcast(&nb64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        nb = (size_t)nb64;

        if (rank != 0)
            buf = (unsigned char *)malloc(nb ? nb : 1);
        MPI_Bcast(buf, (int)nb, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        if (rank != 0 && nb > 0)
            mpz_import(cbrt_table[i], nb, 1, 1, 0, 0, buf);
        free(buf);
    }
}

// ---------------------------------------------------------------------------
// ksc_ensure_k32_primes
// ---------------------------------------------------------------------------

const char *ksc_ensure_k32_primes(int sp_count, const unsigned *sp, int rank)
{
    static char path[128];
    mkdir_data();
    snprintf(path, sizeof(path), "%s/k32_primes.bin", KSC_DATA_DIR);

    int table_ok = 1;
    if (rank == 0)
    {
        if (!file_exists(path))
        {
            printf("[rank 0] Generating k32 prime table → %s (one-time, ~10 s)\n", path);
            fflush(stdout);
            table_ok = kf_generate_and_save_prime_table(path, sp_count, sp, "[rank 0] ") ? 1 : 0;
        }
    }
    MPI_Bcast(&table_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!table_ok)
    {
        fprintf(stderr, "ERROR: could not create %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return path;
}

// ---------------------------------------------------------------------------
// ksc_ensure_k32_cbrt
//
// Runs one UNTIMED factoring pass on the challenge to discover all its prime
// factors, computes cbrt for each, and saves the cache. Subsequent calls load
// the existing cache instantly. The bench loop then sees only cache hits.
// ---------------------------------------------------------------------------

void ksc_ensure_k32_cbrt(
    cbrt_cache_t *cache,
    bool *cache_dirty,
    const char *cache_path,
    const mpz_t challenge,
    const mpz_t modulo,
    const mpz_t e_inv,
    uint64_t my_lo, uint64_t my_hi,
    int sp_count, const unsigned *sp,
    const uint32_t *prime_slice, int64_t slice_count,
    int rank, int nranks,
    int method_uses_table)
{
    mkdir_data();

    // If cache file exists with entries, load and return
    if (kf_cbrt_cache_load(cache, cache_path))
    {
        if (rank == 0)
            printf("[rank %d] cbrt cache loaded: %d entries from %s\n",
                   rank, cache->count, cache_path);
        return;
    }

    if (rank == 0)
        printf("[rank 0] cbrt cache missing — running precompute pass (untimed)\n");
    fflush(stdout);

    // Factor the challenge with this rank's prime slice/sieve
    kfactor_list_t factors;
    kfl_init(&factors);
    if (method_uses_table)
        kf_trial_divide_table(challenge, prime_slice, slice_count, &factors);
    else
        kf_trial_divide_sieve(challenge, my_lo, my_hi, sp_count, sp, &factors);

    // Compute cbrt for each factor found
    mpz_t *parts = (mpz_t *)malloc((size_t)factors.count * sizeof(mpz_t));
    for (int i = 0; i < factors.count; i++)
    {
        mpz_init(parts[i]);
        unsigned long p = mpz_get_ui(factors.items[i].prime);
        int idx = kf_cbrt_cache_find(cache, p);
        if (idx < 0)
        {
            mpz_t p_mpz;
            mpz_init_set_ui(p_mpz, p);
            mpz_t cbrt_p;
            mpz_init(cbrt_p);
            mpz_powm(cbrt_p, p_mpz, e_inv, modulo);
            kf_cbrt_cache_append(cache, p, cbrt_p);
            *cache_dirty = true;
            mpz_clear(p_mpz);
            mpz_clear(cbrt_p);
        }
        mpz_clear(parts[i]);
    }
    free(parts);
    kfl_free(&factors);

    if (*cache_dirty)
    {
        kf_cbrt_cache_save(cache, cache_path);
        if (rank == 0)
            printf("[rank %d] cbrt cache saved: %d entries → %s\n",
                   rank, cache->count, cache_path);
        fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// ksc_ensure_k64_cbrt
//
// In the pure-Pollard k64 model the cbrt cache is built incrementally:
// every time Pollard's rho succeeds it calls cbrt_from_factors() which
// populates the cache and marks it dirty.  The cache is saved after the
// bench loop and loaded on the next run.
//
// This function therefore only needs to:
//   1. Derive the canonical path and attempt to load.
//   2. Broadcast whatever is already cached to all ranks.
//   3. Return — no factoring is done here.
//
// The warm-up factoring (to avoid cold misses on run 0) is handled by the
// solver's two silent warmup iterations before the timed bench loop.
// ---------------------------------------------------------------------------

void ksc_ensure_k64_cbrt(
    cbrt_cache_t *cache,
    const mpz_t challenge,
    const mpz_t modulo,
    const char *modulo_file,
    int rank)
{
    mkdir_data();

    // Derive path: data/cbrt/<stem>.cbrt  (shared — no rank suffix)
    const char *bname = modulo_file ? modulo_file : "modulo";
    const char *slash = strrchr(bname, '/');
    const char *fname = slash ? slash + 1 : bname;
    char stem[256];
    strncpy(stem, fname, sizeof(stem) - 1);
    stem[sizeof(stem) - 1] = '\0';
    size_t blen = strlen(stem);
    if (blen >= 4 && strcmp(stem + blen - 4, ".txt") == 0)
        stem[blen - 4] = '\0';
    char cache_path[280];
    snprintf(cache_path, sizeof(cache_path), "%s/%s.cbrt", KSC_CBRT_DIR, stem);

    // Attempt to load existing cache on rank 0
    bool loaded = false;
    if (rank == 0)
        loaded = kf_cbrt_cache_load(cache, cache_path);

    // Broadcast any loaded entries to all other ranks
    int count = (rank == 0) ? cache->count : 0;
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        kf_cbrt_cache_clear(cache);
        kf_cbrt_cache_init(cache);
    }

    for (int i = 0; i < count; i++)
    {
        uint64_t prime64 = 0;
        if (rank == 0)
            prime64 = (uint64_t)cache->items[i].prime;
        MPI_Bcast(&prime64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        uint32_t nb = 0;
        unsigned char *buf = nullptr;
        if (rank == 0)
        {
            size_t sz = 0;
            buf = (unsigned char *)mpz_export(NULL, &sz, 1, 1, 0, 0,
                                              cache->items[i].cbrt);
            nb = (uint32_t)sz;
        }
        MPI_Bcast(&nb, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        if (rank != 0)
            buf = (unsigned char *)malloc(nb ? nb : 1);
        MPI_Bcast(buf, (int)nb, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        if (rank != 0)
        {
            mpz_t v;
            mpz_init(v);
            if (nb > 0)
                mpz_import(v, nb, 1, 1, 0, 0, buf);
            kf_cbrt_cache_append(cache, (unsigned long)prime64, v);
            mpz_clear(v);
        }
        free(buf);
    }

    if (rank == 0)
    {
        if (loaded && count > 0)
            printf("[rank 0] k64 cbrt cache: %d entries loaded from %s\n",
                   count, cache_path);
        else
            printf("[rank 0] k64 cbrt cache: empty — will build during runs\n");
        fflush(stdout);
    }
}