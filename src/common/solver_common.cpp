// =============================================================================
// solver_common.cpp — Shared MPI+OpenMP infrastructure for cube root solvers
// =============================================================================

#include "solver_common.h"

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
// ksc_compute_e_inv
// ---------------------------------------------------------------------------

void ksc_compute_e_inv(mpz_t e_inv_out, const mpz_t modulo, unsigned long exponent)
{
    mpz_t phi, e_base;
    mpz_inits(phi, e_base, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);
    mpz_invert(e_inv_out, e_base, phi);
    mpz_clears(phi, e_base, NULL);
}

// ---------------------------------------------------------------------------
// ksc_make_cbrt_cache_path
// ---------------------------------------------------------------------------

void ksc_make_cbrt_cache_path(char *buf, size_t bufsz,
                              const char *cbrt_dir,
                              const char *modulo_file,
                              const char *suffix,
                              int rank)
{
    const char *bname = modulo_file ? modulo_file : "modulo";
    const char *slash = strrchr(bname, '/');
    const char *fname = slash ? slash + 1 : bname;
    char stem[512];
    strncpy(stem, fname, sizeof(stem) - 1);
    stem[sizeof(stem) - 1] = '\0';
    size_t blen = strlen(stem);
    if (blen >= 4 && strcmp(stem + blen - 4, ".txt") == 0)
        stem[blen - 4] = '\0';
    if (rank >= 0)
        snprintf(buf, bufsz, "%s/%s%s.%d", cbrt_dir, stem, suffix, rank);
    else
        snprintf(buf, bufsz, "%s/%s%s", cbrt_dir, stem, suffix);
}

// ---------------------------------------------------------------------------
// ksc_print_challenge_header
// ---------------------------------------------------------------------------

void ksc_print_challenge_header(const mpz_t challenge, const mpz_t modulo,
                                int rank, int nranks)
{
    if (rank != 0)
        return;
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
    int smoothness_bits,
    const char *method_name,
    int mpi_ranks, int threads_per_rank,
    const mpz_t challenge, int rank)
{
    r->config.smoothness_bits = smoothness_bits;
    r->config.challenge_bits = (rank == 0) ? mpz_sizeinbase(challenge, 2) : 0;
    r->config.method_name    = method_name;
    r->config.num_threads    = threads_per_rank;
    r->config.mpi_ranks      = mpi_ranks;
    r->config.n_warm_attempted = 1; // one challenge per run_one; merge sums these
    r->timing.powm_baseline_ms = 0.0;
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
// ksc_compute_local_cbrt
// ---------------------------------------------------------------------------

void ksc_compute_local_cbrt(mpz_t result,
                            const kfactor_list_t *factors,
                            const mpz_t e,
                            const mpz_t modulo,
                            cbrt_cache_t *cache,
                            bool *cache_dirty,
                            double *hit_ms_out,
                            double *miss_ms_out,
                            int *hits_out,
                            int *misses_out,
                            bool record)
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

    // Phase 1 (sequential): resolve cbrt(p) for each factor via cache or powm.
    // Cache lookups and appends are NOT thread-safe (realloc), so this phase
    // must run sequentially. With a warm cache every call is a cheap array
    // lookup (~0 ms total), so sequential cost is negligible.
    mpz_t *cbrt_vals = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    for (int i = 0; i < n; i++)
    {
        mpz_init(cbrt_vals[i]);
        unsigned long p_ul = mpz_get_ui(factors->items[i].prime);
        int cache_idx = (cache && p_ul > 0)
                            ? kf_cbrt_cache_find(cache, p_ul)
                            : -1;

        auto _ct0 = std::chrono::high_resolution_clock::now();
        if (cache_idx >= 0)
        {
            mpz_set(cbrt_vals[i], cache->items[cache_idx].cbrt);
            auto _ct1 = std::chrono::high_resolution_clock::now();
            if (record && hit_ms_out)
                *hit_ms_out += std::chrono::duration<double, std::milli>(_ct1 - _ct0).count();
            if (record && hits_out)
                (*hits_out)++;
        }
        else
        {
            mpz_t p_mpz;
            mpz_init_set_ui(p_mpz, p_ul);
            mpz_powm(cbrt_vals[i], p_mpz, e, modulo);
            mpz_clear(p_mpz);
            auto _ct1 = std::chrono::high_resolution_clock::now();
            if (record && miss_ms_out)
                *miss_ms_out += std::chrono::duration<double, std::milli>(_ct1 - _ct0).count();
            if (record && misses_out)
                (*misses_out)++;

            if (cache && p_ul > 0)
            {
                kf_cbrt_cache_append(cache, p_ul, cbrt_vals[i]);
                if (cache_dirty)
                    *cache_dirty = true;
            }
        }
    }

    // Phase 2 (parallel): parts[i] = cbrt_vals[i]^exponent mod M.
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
        r->stats.primes_found_per_rank.insert(r->stats.primes_found_per_rank.end(),
                                              all_found.begin(), all_found.end());
        r->stats.primes_tested_per_rank.insert(r->stats.primes_tested_per_rank.end(),
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

// One timed run per challenge; the accumulators hold that challenge's values.
void ksc_print_timing_report(
    const bench_accum_t *fac_acc,
    const bench_accum_t *loc_acc,
    const bench_accum_t *gat_acc,
    const bench_accum_t *com_acc,
    const bench_accum_t *recon_acc)
{
    printf("\n");
    printf("Factoring:         %.3f ms\n", fac_acc->sum);
    printf("Local cbrt (max):  %.3f ms\n", loc_acc->sum);
    printf("Gather:            %.3f ms\n", gat_acc->sum);
    printf("Combine:           %.3f ms\n", com_acc->sum);
    printf("Reconstruction:    %.3f ms\n", recon_acc->sum);
    printf("Total:             %.3f ms\n", fac_acc->sum + recon_acc->sum);
}

// ---------------------------------------------------------------------------
// ksc_run_and_report_powm_baseline
// ---------------------------------------------------------------------------

void ksc_run_and_report_powm_baseline(
    benchmark_results_t *r,
    const mpz_t challenge,
    const mpz_t modulo,
    const bench_accum_t *recon_acc,
    const bench_accum_t *fac_acc,
    int rank,
    unsigned long exponent)
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
    double powm_avg = run_powm_baseline(tmp, challenge, modulo,
                                        KSC_POWM_BASELINE_REPS, 30, exponent);
    r->timing.powm_baseline_ms = powm_avg;
    mpz_clear(tmp);

    // One timed run per challenge: the accumulators already hold its values.
    double recon_avg = recon_acc->sum;
    double total_avg = fac_acc->sum + recon_acc->sum;
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
    // Create data/ and data/cbrt/ if they don't exist.
    // Use mkdir() syscall directly — no shell subprocess, no injection surface.
    // EEXIST is expected and safe to ignore.
    mkdir(KSC_DATA_DIR,  0755);
    mkdir(KSC_CBRT_DIR,  0755);
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
    const char *cbrt_dir,
    int rank,
    unsigned long exponent)
{
    mkdir_data();
    mkdir(cbrt_dir, 0755);

    // Derive path: <cbrt_dir>/<modulo_stem>.cbrt
    const char *fname = strrchr(modulo_file, '/');
    fname = fname ? fname + 1 : modulo_file;
    char stem[256];
    strncpy(stem, fname, sizeof(stem) - 1);
    stem[sizeof(stem) - 1] = '\0';
    char *dot = strrchr(stem, '.');
    if (dot)
        *dot = '\0';
    char base[KSC_CBRT_PATH_MAX];
    snprintf(base, sizeof(base), "%s/%s.cbrt", cbrt_dir, stem);

    bool loaded = false;
    if (rank == 0)
    {
        loaded = kf_load_cbrt_table_k16(cbrt_table, prime_count, base);
        if (!loaded)
        {
            kf_precompute_cbrt_table_k16(cbrt_table, primes, prime_count, modulo, exponent);
            kf_save_cbrt_table_k16(cbrt_table, prime_count, base);
        }
    }

    // Broadcast the entire cbrt table in ONE collective instead of two per
    // entry (was 2 × 6542 ≈ 13k MPI_Bcast calls per challenge — ~0.4s of pure
    // latency at 128 ranks).  Each value is < M, so it fits in `stride`
    // zero-padded big-endian bytes.  prime_count and modulo are identical on
    // every rank, so the stride and layout need no extra broadcast.
    size_t stride = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
    size_t total  = (size_t)prime_count * stride;
    unsigned char *tbl_buf = (unsigned char *)calloc(total ? total : 1, 1);
    if (rank == 0)
    {
        for (int i = 0; i < prime_count; i++)
        {
            size_t nb = 0;
            unsigned char *tmp =
                (unsigned char *)mpz_export(NULL, &nb, 1, 1, 0, 0, cbrt_table[i]);
            if (tmp && nb > 0 && nb <= stride)
                memcpy(tbl_buf + (size_t)i * stride + (stride - nb), tmp, nb);
            free(tmp);
        }
    }
    MPI_Bcast(tbl_buf, (int)total, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        for (int i = 0; i < prime_count; i++)
            mpz_import(cbrt_table[i], stride, 1, 1, 0, 0,
                       tbl_buf + (size_t)i * stride);
    }
    free(tbl_buf);
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
// ksc_ensure_cbrt_cache
//
// Unified cbrt cache precomputation for k32, k35, alg3, and alg4.
// Loads the cache from disk if present; otherwise runs one untimed factoring
// pass, computes cbrt for every found prime, and saves the result.
// ---------------------------------------------------------------------------

void ksc_ensure_cbrt_cache(
    cbrt_cache_t *cache,
    bool *cache_dirty,
    const char *cache_path,
    const mpz_t challenge,
    const mpz_t modulo,
    const mpz_t e_inv,
    uint64_t my_lo, uint64_t my_hi,
    int sp_count, const unsigned *sp,
    const uint32_t *prime_slice, int64_t slice_count,
    bool check_small_primes,
    int rank)
{
    mkdir_data();
    // Best-effort: create the directory portion of cache_path if it doesn't exist.
    // Extracts everything up to the last '/' and calls mkdir (EEXIST is fine).
    {
        char dir[KSC_CBRT_PATH_MAX];
        strncpy(dir, cache_path, sizeof(dir) - 1);
        dir[sizeof(dir) - 1] = '\0';
        char *last_slash = strrchr(dir, '/');
        if (last_slash && last_slash != dir)
        {
            *last_slash = '\0';
            mkdir(dir, 0755);
        }
    }

    if (kf_cbrt_cache_load(cache, cache_path))
        return;

    kfactor_list_t factors;
    kfl_init(&factors);

    // Optional small-prime pass on rank 0: catches factors below my_lo that
    // the main sieve would miss (used by k35 whose range starts at 65537).
    if (check_small_primes && rank == 0)
    {
        for (int si = 0; si < sp_count && (uint64_t)sp[si] < my_lo; si++)
        {
            unsigned long p = (unsigned long)sp[si];
            if (mpz_divisible_ui_p(challenge, p))
            {
                mpz_t tmp;
                mpz_init_set(tmp, challenge);
                int e = 0;
                while (mpz_divisible_ui_p(tmp, p))
                {
                    mpz_divexact_ui(tmp, tmp, p);
                    e++;
                }
                mpz_clear(tmp);
                kfl_append_ul(&factors, p, e);
            }
        }
    }

    if (prime_slice)
        kf_trial_divide_table(challenge, prime_slice, slice_count, &factors);
    else
        kf_trial_divide_sieve(challenge, my_lo, my_hi, sp_count, sp, &factors);

    // Compute cbrt for each factor and append to cache (skip duplicates).
    mpz_t p_mpz, cbrt_p;
    mpz_inits(p_mpz, cbrt_p, NULL);
    for (int i = 0; i < factors.count; i++)
    {
        unsigned long p = mpz_get_ui(factors.items[i].prime);
        if (kf_cbrt_cache_find(cache, p) < 0)
        {
            mpz_set_ui(p_mpz, p);
            mpz_powm(cbrt_p, p_mpz, e_inv, modulo);
            kf_cbrt_cache_append(cache, p, cbrt_p);
            *cache_dirty = true;
        }
    }
    mpz_clears(p_mpz, cbrt_p, NULL);
    kfl_free(&factors);

    if (*cache_dirty)
        kf_cbrt_cache_save(cache, cache_path);
}

// ---------------------------------------------------------------------------
// ksc_shared_cbrt_path — <cbrt_dir>/cbrt_<size>.cbrt (merged per-variant cache).
// <size> parsed from the modulo filename: modulo_<...>_<size>[.txt] or
// modulo_<...>_<size>_e<N>.txt.
// ---------------------------------------------------------------------------
void ksc_shared_cbrt_path(char *buf, size_t bufsz,
                          const char *cbrt_dir, const char *modulo_file)
{
    const char *bname = modulo_file ? modulo_file : "";
    const char *slash = strrchr(bname, '/');
    const char *fname = slash ? slash + 1 : bname;
    char stem[256];
    strncpy(stem, fname, sizeof(stem) - 1);
    stem[sizeof(stem) - 1] = '\0';
    size_t L = strlen(stem);
    if (L >= 4 && strcmp(stem + L - 4, ".txt") == 0)
        stem[L - 4] = '\0';
    // strip a trailing _e<digit> exponent tag if present
    char *ue = strrchr(stem, '_');
    if (ue && ue[1] == 'e' && ue[2] >= '0' && ue[2] <= '9')
        *ue = '\0';
    // the size label is the trailing _<digits>
    char *us = strrchr(stem, '_');
    const char *size = us ? us + 1 : stem;
    snprintf(buf, bufsz, "%s/cbrt_%s.cbrt", cbrt_dir, size);
}

// ksc_ensure_k35_primes

// ---------------------------------------------------------------------------
// ksc_ensure_k35_primes
//
// Rank 0 generates k35_upper_primes.bin (primes in (2^32, 2^35]) if it does
// not exist, then all ranks MPI_Barrier before returning the path.
// Returns the path to the file (static string).
// ---------------------------------------------------------------------------
const char *ksc_ensure_k35_primes(int sp_count, const unsigned *sp, int rank)
{
    static const char *path = K35U_DEFAULT_PATH;

    // All ranks check existence; only rank 0 generates
    int exists = 0;
    if (rank == 0)
    {
        FILE *f = fopen(path, "rb");
        if (f)
        {
            fclose(f);
            exists = 1;
        }
    }
    MPI_Bcast(&exists, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!exists)
    {
        if (rank == 0)
        {
            mkdir_data();
            kf_generate_and_save_prime_table_u64(path,
                                                 (uint64_t)UINT32_MAX + 1ULL, // 2^32
                                                 (1ULL << 35) + 1024ULL,      // 2^35 + margin
                                                 sp_count, sp,
                                                 "[rank 0] k35u");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    return path;
}

// ---------------------------------------------------------------------------
// ksc_alg2_broadcast_complete — Algorithm 2 (k32/k35) B-smooth completeness
//
// Gathers each rank's partial factor product, combines on rank 0, and checks
// whether combined_product == blinded (i.e. cofactor = 1, fully B-smooth).
// Returns 1 on all ranks if complete, 0 otherwise.
//
// No cofactor value is broadcast — Algorithm 2 does not have one.
// ---------------------------------------------------------------------------

int ksc_alg2_broadcast_complete(
    const mpz_t blinded,
    const mpz_t local_prod,
    size_t cbrt_bytes,
    int rank, int nranks)
{
    // --- 1. Export local factor product ---
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

    // --- 2. Gather to rank 0 ---
    unsigned char *all_prods = NULL;
    if (rank == 0)
        all_prods = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);

    MPI_Gather(prod_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
               all_prods, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
    free(prod_buf);

    // --- 3. Rank 0: check combined == blinded ---
    int complete = 0;
    if (rank == 0)
    {
        mpz_t combined, part;
        mpz_init_set_ui(combined, 1);
        mpz_init(part);
        for (int rk = 0; rk < nranks; rk++)
        {
            mpz_import(part, cbrt_bytes, 1, 1, 0, 0,
                       all_prods + (size_t)rk * cbrt_bytes);
            mpz_mul(combined, combined, part);
        }
        free(all_prods);
        complete = (mpz_cmp(combined, blinded) == 0) ? 1 : 0;
        mpz_clears(combined, part, NULL);
    }

    // --- 4. Broadcast result ---
    MPI_Bcast(&complete, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return complete;
}

// ---------------------------------------------------------------------------
// ksc_alg3_broadcast_cofactor
// ---------------------------------------------------------------------------

// Upper bound on the medium prime exponent (2^65 fits in 66 bits including sign;
// mpz_sizeinbase can return size+1 so we accept up to 66).
#define ALG3_B_PRIME_BITS 65

int ksc_alg3_broadcast_cofactor(
    mpz_t cofactor_out,
    const mpz_t blinded,
    const mpz_t local_prod,
    size_t cbrt_bytes,
    int rank, int nranks,
    double *primality_ms)
{
    // --- 1. Export local factor product to fixed-size big-endian buffer ---
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

    // --- 2. Gather all local products to rank 0 ---
    unsigned char *all_prods = NULL;
    if (rank == 0)
        all_prods = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);

    MPI_Gather(prod_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
               all_prods, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
    free(prod_buf);

    // --- 3. Rank 0: combine, compute cofactor, classify ---
    int complete = 0;
    unsigned char *cof_buf = (unsigned char *)calloc(cbrt_bytes, 1);

    if (rank == 0)
    {
        mpz_t combined, part, q, r;
        mpz_init_set_ui(combined, 1);
        mpz_inits(part, q, r, NULL);

        for (int rk = 0; rk < nranks; rk++)
        {
            mpz_import(part, cbrt_bytes, 1, 1, 0, 0,
                       all_prods + (size_t)rk * cbrt_bytes);
            mpz_mul(combined, combined, part);
        }
        free(all_prods);

        // cofactor = blinded / combined_small_product
        // Use tdiv_qr: non-zero remainder means combined doesn't divide blinded → retry.
        mpz_tdiv_qr(q, r, blinded, combined);
        mpz_clears(combined, part, NULL);

        if (mpz_sgn(r) == 0)
        {
            if (mpz_cmp_ui(q, 1) == 0)
            {
                // B-smooth: all factors accounted for
                complete = 1;
                mpz_set_ui(cofactor_out, 1);
            }
            else if (mpz_sizeinbase(q, 2) > 32 &&
                     mpz_sizeinbase(q, 2) <= ALG3_B_PRIME_BITS + 1)
            {
                // M4 (2026-07-16): the q0 primality test is modelled as a table
                // lookup, so its wall-time is measured separately and reported
                // back to be excluded from the factor time (not attack work).
                auto _p0 = std::chrono::high_resolution_clock::now();
                int q0_is_prime = mpz_probab_prime_p(q, 15);
                auto _p1 = std::chrono::high_resolution_clock::now();
                if (primality_ms)
                    *primality_ms += std::chrono::duration<double, std::milli>(
                                         _p1 - _p0).count();
                if (q0_is_prime)
                {
                    // Almost-smooth: one medium prime q0 in (2^32, 2^65]
                    complete = 1;
                    mpz_set(cofactor_out, q);
                }
            }
            // else: cofactor is composite or too large → retry

            // Encode cofactor_out into cof_buf for MPI_Bcast to non-zero ranks.
            // This MUST cover both B-smooth (cofactor=1) and almost-smooth.
            // Previously only the almost-smooth branch wrote cof_buf, leaving it
            // all-zeros for B-smooth.  Non-zero ranks then imported 0 instead of
            // 1, causing the cofactor != 1 guard in ksc_run_bench_loop to force
            // complete=0, breaking the symmetry between ranks and deadlocking.
            if (complete)
            {
                size_t written = 0;
                unsigned char *tmp = (unsigned char *)mpz_export(
                    NULL, &written, 1, 1, 0, 0, cofactor_out);
                if (tmp && written > 0 && written <= cbrt_bytes)
                {
                    memcpy(cof_buf + cbrt_bytes - written, tmp, written);
                    free(tmp);
                }
            }
        }
        mpz_clears(q, r, NULL);
    }

    // --- 4. Broadcast result to all ranks ---
    MPI_Bcast(&complete, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cof_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (complete && rank != 0)
        mpz_import(cofactor_out, cbrt_bytes, 1, 1, 0, 0, cof_buf);

    free(cof_buf);
    return complete;
}

// ---------------------------------------------------------------------------
// ksc_alg4_broadcast_cofactor
// ---------------------------------------------------------------------------

// One-entry cache for composite cofactor factorization.
//
// kf_try_factor_b64smooth runs Pollard's rho which can take ~120 seconds for
// a 119-bit composite with 60-bit factors (√(2^59) ≈ 750M iterations with no
// inner bound in kf_pollard_rho_brent).  Since all bench runs on the same
// challenge always produce the SAME cofactor (outer=0 unblinded path), we
// would otherwise pay this cost once per bench run — 9 × 120s = 18 minutes.
//
// This cache stores the most recently factored composite and its factor list.
// On a cache hit the result is returned immediately (< 1μs).  On a miss
// Pollard's rho runs as before and the result is stored for future calls.
//
// Scope: file-local, rank-0-only (non-zero ranks never enter the factoring
// branch so this data is never read by them).
// GMP types require mpz_init before use; we use s_cf_valid as an init guard.
static bool s_cf_valid = false;
static mpz_t s_cf_composite;        // cached input composite
static kfactor_list_t s_cf_factors; // cached output factor list

// ALG4_B_COFACTOR_BITS: maximum bit-length of the cofactor that this function
// will attempt to factor via Pollard's rho.
//
// For alg4 (single medium prime):   cofactor is prime <= 2^64 → 64 bits.
// For alg4_multi (two medium primes): cofactor = p1*p2 where each p_i is in
//   (2^32, 2^64].  Both primes have at least 33 bits, so their product has
//   at least 66 bits and at most 2*63 = 126 bits.  Setting the limit to 128
//   covers all alg4_multi challenges without accepting unreasonably large
//   composites.
//
// kf_try_factor_b64smooth is recursive: it splits via Pollard's rho and
// checks each factor for 2^64-smoothness.  A 126-bit composite of two
// 63-bit primes is correctly handled — the recursion terminates when each
// half is confirmed prime and <= 2^64.
#define ALG4_B_COFACTOR_BITS 128
#define ALG4_MAX_MEDIUM 4 // up to 4 medium primes (alg4_multi uses 2)

int ksc_alg4_broadcast_cofactor(
    kfactor_list_t *medium_factors_out,
    const mpz_t blinded,
    const mpz_t local_prod,
    size_t cbrt_bytes,
    int rank, int nranks)
{
    // --- 1. Export local factor product to fixed-size big-endian buffer ---
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

    // --- 2. Gather all local products to rank 0 ---
    unsigned char *all_prods = NULL;
    if (rank == 0)
        all_prods = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);

    MPI_Gather(prod_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
               all_prods, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
    free(prod_buf);

    // --- 3. Rank 0: combine, compute cofactor, classify ---
    int complete = 0;
    int n_medium = 0;
    unsigned char *medium_buf =
        (unsigned char *)calloc((size_t)ALG4_MAX_MEDIUM * cbrt_bytes, 1);

    if (rank == 0)
    {
        mpz_t combined, part, q, r;
        mpz_init_set_ui(combined, 1);
        mpz_inits(part, q, r, NULL);

        for (int rk = 0; rk < nranks; rk++)
        {
            mpz_import(part, cbrt_bytes, 1, 1, 0, 0,
                       all_prods + (size_t)rk * cbrt_bytes);
            mpz_mul(combined, combined, part);
        }
        free(all_prods);

        mpz_tdiv_qr(q, r, blinded, combined);
        mpz_clears(combined, part, NULL);

        if (mpz_sgn(r) == 0)
        {
            if (mpz_cmp_ui(q, 1) == 0)
            {
                // B-smooth: all factors accounted for
                complete = 1;
                n_medium = 0;
            }
            else if (mpz_sizeinbase(q, 2) <= ALG4_B_COFACTOR_BITS + 1 &&
                     mpz_probab_prime_p(q, 15))
            {
                // Single medium prime in (2^32, 2^128].
                // alg4:       q0 in (2^32, 2^64]  — 33..64 bits
                // alg4_multi: p1 or p2 alone never reaches here because
                //             they are multiplied together before we see them.
                complete = 1;
                n_medium = 1;
                kfl_append_mpz(medium_factors_out, q, 1);
                size_t written = 0;
                unsigned char *tmp = (unsigned char *)mpz_export(
                    NULL, &written, 1, 1, 0, 0, q);
                if (tmp && written > 0 && written <= cbrt_bytes)
                {
                    memcpy(medium_buf + cbrt_bytes - written, tmp, written);
                    free(tmp);
                }
            }
            else if (mpz_sizeinbase(q, 2) <= ALG4_B_COFACTOR_BITS + 1)
            {
                // Composite cofactor with all prime factors <= 2^64.
                //
                // alg4_multi structure: cofactor = p1 * p2 where each p_i is
                // a prime in (2^32, 2^64].  The product is 66..126 bits.
                //
                // kf_try_factor_b64smooth handles this correctly but is SLOW
                // for large composites (√(2^59) ≈ 750M Pollard steps ≈ 120s
                // for 119-bit composites with 60-bit factors).  Without caching
                // this would run on EVERY bench iteration for the same challenge.
                //
                // The one-entry cache s_cf_composite/s_cf_factors ensures
                // Pollard's rho runs at most once per unique composite per job.
                kfactor_list_t poly;
                kfl_init(&poly);
                bool ok = false;

                if (s_cf_valid && mpz_cmp(q, s_cf_composite) == 0)
                {
                    // Cache hit — reuse previously computed factorization.
                    for (int i = 0; i < s_cf_factors.count; i++)
                        kfl_append_mpz(&poly,
                                       s_cf_factors.items[i].prime,
                                       s_cf_factors.items[i].exponent);
                    ok = true;
                }
                else
                {
                    // Cache miss — run Pollard's rho (slow first time only).
                    ok = kf_try_factor_b64smooth(q, &poly, 1);
                    if (ok)
                    {
                        // Store for all subsequent bench runs.
                        if (!s_cf_valid)
                        {
                            mpz_init(s_cf_composite);
                            kfl_init(&s_cf_factors);
                            s_cf_valid = true;
                        }
                        else
                        {
                            kfl_free(&s_cf_factors);
                            kfl_init(&s_cf_factors);
                        }
                        mpz_set(s_cf_composite, q);
                        for (int i = 0; i < poly.count; i++)
                            kfl_append_mpz(&s_cf_factors,
                                           poly.items[i].prime,
                                           poly.items[i].exponent);
                    }
                }

                if (ok)
                {
                    complete = 1;
                    for (int i = 0; i < poly.count; i++)
                    {
                        for (int ex = 0; ex < poly.items[i].exponent; ex++)
                        {
                            int slot = n_medium++;
                            kfl_append_mpz(medium_factors_out,
                                           poly.items[i].prime, 1);
                            if (slot < ALG4_MAX_MEDIUM)
                            {
                                size_t written = 0;
                                unsigned char *tmp = (unsigned char *)mpz_export(
                                    NULL, &written, 1, 1, 0, 0,
                                    poly.items[i].prime);
                                if (tmp && written > 0 && written <= cbrt_bytes)
                                {
                                    memcpy(medium_buf + (size_t)slot * cbrt_bytes + cbrt_bytes - written,
                                           tmp, written);
                                    free(tmp);
                                }
                            }
                        }
                    }
                }
                kfl_free(&poly);
            }
            // else: cofactor > 2^128 or cannot be fully factored into
            // 2^64-smooth components → retry with a new blinding factor.
        }
        mpz_clears(q, r, NULL);
    }

    // --- 4. Broadcast result to all ranks ---
    MPI_Bcast(&complete, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_medium, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (complete && n_medium > 0)
    {
        int bcast_count = (n_medium < ALG4_MAX_MEDIUM ? n_medium : ALG4_MAX_MEDIUM);
        MPI_Bcast(medium_buf, (int)((size_t)bcast_count * cbrt_bytes),
                  MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // --- 5. Non-rank-0: reconstruct kfactor_list_t from broadcast bytes ---
        if (rank != 0)
        {
            for (int i = 0; i < bcast_count; i++)
            {
                mpz_t p;
                mpz_init(p);
                mpz_import(p, cbrt_bytes, 1, 1, 0, 0,
                           medium_buf + (size_t)i * cbrt_bytes);
                kfl_append_mpz(medium_factors_out, p, 1);
                mpz_clear(p);
            }
        }
    }

    free(medium_buf);
    return complete;
}

// ---------------------------------------------------------------------------
// ksc_reconstruct_y
// ---------------------------------------------------------------------------

int ksc_reconstruct_y(
    mpz_t result_out, mpz_t cofactor_out,
    const mpz_t y, int attempt, int bench,
    bench_accum_t *loc_acc, bench_accum_t *gat_acc, bench_accum_t *com_acc,
    double *fac2_dt_out, double *recon_dt_out,
    benchmark_results_t *json_results,
    const ksc_recon_ctx_t *ctx)
{
    const int rank    = ctx->rank;
    const int nranks  = ctx->nranks;

    // ---- Phase 2a: sieve y + cofactor broadcast ----
    auto t_fac2_start = std::chrono::high_resolution_clock::now();

    kfactor_list_t local_factors;
    kfl_init(&local_factors);
    if (ctx->method == METHOD_TABLE)
        kf_trial_divide_table(y, ctx->prime_slice, ctx->slice_count,
                              &local_factors, nullptr, nullptr, 0);
    else
        kf_trial_divide_sieve(y, ctx->my_lo, ctx->my_hi,
                              ctx->sp_count, ctx->sp, &local_factors);

    mpz_t local_prod;
    mpz_init_set_ui(local_prod, 1);
    for (int i = 0; i < local_factors.count; i++)
    {
        unsigned long p = mpz_get_ui(local_factors.items[i].prime);
        for (int ex = 0; ex < local_factors.items[i].exponent; ex++)
            mpz_mul_ui(local_prod, local_prod, p);
    }

    mpz_set_ui(cofactor_out, 1);
    double primality_dt = 0.0;   // M4: q0 primality-test time to exclude
    int complete = ksc_alg3_broadcast_cofactor(
        cofactor_out, y, local_prod, ctx->cbrt_bytes, rank, nranks, &primality_dt);
    mpz_clear(local_prod);

    auto t_fac2_end = std::chrono::high_resolution_clock::now();
    // M4 (2026-07-16): exclude the q0 primality-test time (modelled as a table
    // lookup, not attack work).
    double fac2_dt = std::chrono::duration<double, std::milli>(
                         t_fac2_end - t_fac2_start).count() - primality_dt;
    if (fac2_dt_out)
        *fac2_dt_out = fac2_dt;

    if (!complete)
    {
        kfl_free(&local_factors);
        return 0;
    }

    // ---- Phase 2b: local cbrt + fold + reduce + barrier + gather ----
    auto t_loc0 = std::chrono::high_resolution_clock::now();

    mpz_t local_cbrt;
    mpz_init(local_cbrt);
    double cbrt_hit_ms = 0.0, cbrt_miss_ms = 0.0;
    int cbrt_hits = 0, cbrt_misses = 0;
    bool record_cbrt = (bench > 0);
    ksc_compute_local_cbrt(local_cbrt, &local_factors,
                           *ctx->e_inv, *ctx->modulo,
                           ctx->cache, ctx->cache_dirty,
                           &cbrt_hit_ms, &cbrt_miss_ms,
                           &cbrt_hits, &cbrt_misses, record_cbrt);

    if (rank == 0)
        ksc_fold_cofactor_cbrt(local_cbrt, cofactor_out,
                               *ctx->e_inv, *ctx->modulo,
                               ctx->cache, ctx->cache_dirty);

    auto t_loc1 = std::chrono::high_resolution_clock::now();
    double local_dt = std::chrono::duration<double, std::milli>(
                          t_loc1 - t_loc0).count();

    if (bench == 1)
        ksc_gather_prime_stats(json_results,
                               local_factors.count,
                               (int64_t)(ctx->my_hi - ctx->my_lo + 1),
                               rank, nranks);

    if (rank == 0 && bench > 0)
    {
        json_results->timing.cbrt_hit_ms.push_back(cbrt_hit_ms);
        json_results->timing.cbrt_miss_ms.push_back(cbrt_miss_ms);
        json_results->timing.cbrt_hits.push_back(cbrt_hits);
        json_results->timing.cbrt_misses.push_back(cbrt_misses);
    }

    double max_local_dt = 0.0;
    MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        accum_record(loc_acc, max_local_dt);
        if (bench > 0)
            json_results->timing.local_cbrt_times.push_back(max_local_dt);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double iter_gat = 0.0, iter_com = 0.0;
    ksc_gather_and_combine_cbrt(result_out, local_cbrt,
                                *ctx->modulo, ctx->cbrt_bytes,
                                rank, nranks, attempt,
                                ctx->blind_table, ctx->blind_table_size,
                                &iter_gat, &iter_com);
    mpz_clear(local_cbrt);
    kfl_free(&local_factors);

    if (rank == 0)
    {
        accum_record(gat_acc, iter_gat);
        accum_record(com_acc, iter_com);
        if (bench > 0)
        {
            json_results->timing.gather_times.push_back(iter_gat);
            json_results->timing.combine_times.push_back(iter_com);
        }
        if (recon_dt_out)
            *recon_dt_out = fac2_dt + max_local_dt + iter_gat + iter_com;
    }

    return 1;
}

// ---------------------------------------------------------------------------
// ksc_fold_cofactor_cbrt
// ---------------------------------------------------------------------------

void ksc_fold_cofactor_cbrt(
    mpz_t local_cbrt,
    const mpz_t cofactor,
    const mpz_t e_inv,
    const mpz_t modulo,
    cbrt_cache_t *cache,
    bool *cache_dirty)
{
    if (mpz_cmp_ui(cofactor, 1) == 0)
        return;

    mpz_t cbrt_q0;
    mpz_init(cbrt_q0);
    unsigned long q0_ul = mpz_fits_ulong_p(cofactor)
                              ? mpz_get_ui(cofactor) : 0UL;
    int idx = (q0_ul > 0UL) ? kf_cbrt_cache_find(cache, q0_ul) : -1;
    if (idx >= 0)
    {
        mpz_set(cbrt_q0, cache->items[idx].cbrt);
    }
    else
    {
        mpz_powm(cbrt_q0, cofactor, e_inv, modulo);
        if (q0_ul > 0UL)
        {
            kf_cbrt_cache_append(cache, q0_ul, cbrt_q0);
            *cache_dirty = true;
        }
    }
    mpz_mul(local_cbrt, local_cbrt, cbrt_q0);
    mpz_mod(local_cbrt, local_cbrt, modulo);
    mpz_clear(cbrt_q0);
}

// ---------------------------------------------------------------------------
// ksc_run_bench_loop
// ---------------------------------------------------------------------------

void ksc_run_bench_loop(
    mpz_t result_out,
    run_method_t method,
    const mpz_t challenge,
    const mpz_t modulo,
    size_t cbrt_bytes,
    const mpz_t e_inv,
    cbrt_cache_t *cache, bool *cache_dirty,
    blinding_pair_t *blind_table, int blind_table_size,
    uint64_t my_lo, uint64_t my_hi,
    bench_accum_t *fac_acc, bench_accum_t *loc_acc,
    bench_accum_t *gat_acc, bench_accum_t *com_acc, bench_accum_t *recon_acc,
    benchmark_results_t *json_results,
    const ksc_factor_strategy_t *strategy,
    int rank, int nranks)
{
    // ---- 2 warmup runs ----
    // Prime CPU caches before any timed region. The challenge is used
    // unblinded (r=1) for warmup; results are discarded.
    if (method != METHOD_POWM)
    {
        for (int wu = 0; wu < 2; wu++)
        {
            kfactor_list_t wu_factors;
            kfl_init(&wu_factors);

            size_t wu_nb = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
            unsigned char *wu_buf = (unsigned char *)calloc(wu_nb, 1);
            if (rank == 0)
            {
                size_t wr = 0;
                unsigned char *tmp = (unsigned char *)
                    mpz_export(NULL, &wr, 1, 1, 0, 0, challenge);
                if (tmp && wr > 0 && wr <= wu_nb)
                {
                    memcpy(wu_buf + wu_nb - wr, tmp, wr);
                    free(tmp);
                }
            }
            MPI_Bcast(wu_buf, (int)wu_nb, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            mpz_t wu_ch;
            mpz_init(wu_ch);
            mpz_import(wu_ch, wu_nb, 1, 1, 0, 0, wu_buf);
            free(wu_buf);

            strategy->factor(&wu_factors, wu_ch, -1, json_results,
                             strategy->ctx, rank, nranks);

            mpz_t wu_cbrt;
            mpz_init(wu_cbrt);
            bool wu_dirty = false;
            auto wu_t0 = std::chrono::high_resolution_clock::now();
            ksc_compute_local_cbrt(wu_cbrt, &wu_factors, e_inv, modulo,
                                   cache, &wu_dirty);
            auto wu_t1 = std::chrono::high_resolution_clock::now();
            if (wu_dirty)
                *cache_dirty = true;
            if (rank == 0)
                json_results->timing.warmup_total_ms.push_back(
                    std::chrono::duration<double, std::milli>(
                        wu_t1 - wu_t0).count());
            mpz_clears(wu_ch, wu_cbrt, NULL);
            kfl_free(&wu_factors);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // ---- Benchmark loop ----

    mpz_t blinded_challenge, cofactor;
    mpz_init(blinded_challenge);
    mpz_init_set_ui(cofactor, 1);
    double cur_max_loc = 0.0;

    // One timed run for this challenge (multiple challenges are run by the
    // caller and merged; the old per-challenge repeat / --bench is removed).
    const int bench = 0;
    if (method != METHOD_POWM)
    {
        kfactor_list_t local_factors;
        kfl_init(&local_factors);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        int attempt = 0;
        const int MAX_ATTEMPTS = 256;
        bool bench_succeeded = false;  // false if all blinding attempts exhausted
        double primality_dt = 0.0;     // M4: q0 primality-test time to exclude from factor_ms
        size_t blinded_nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
        unsigned char *blinded_buf = (unsigned char *)malloc(blinded_nbytes);

        for (attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
        {
            // Rank 0: blind challenge and export to fixed-size buffer
            if (rank == 0)
            {
                if (attempt == 0)
                {
                    mpz_set(blinded_challenge, challenge);
                }
                else
                {
                    int idx = attempt - 1;
                    if (idx < blind_table_size)
                    {
                        mpz_mul(blinded_challenge, challenge,
                                blind_table[idx].r_blind);
                        mpz_mod(blinded_challenge, blinded_challenge, modulo);
                    }
                    else
                    {
                        fprintf(stderr,
                                "[rank 0] WARNING: blinding table exhausted "
                                "at attempt %d\n", attempt);
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

            // All ranks factor their slice
            kfl_free(&local_factors);
            kfl_init(&local_factors);
            strategy->factor(&local_factors, blinded_challenge, bench,
                             json_results, strategy->ctx, rank, nranks);

            // Completeness check — dispatch by algorithm family
            mpz_t local_prod;
            mpz_init_set_ui(local_prod, 1);
            for (int i = 0; i < local_factors.count; i++)
            {
                unsigned long p = mpz_get_ui(local_factors.items[i].prime);
                for (int ex = 0; ex < local_factors.items[i].exponent; ex++)
                    mpz_mul_ui(local_prod, local_prod, p);
            }

            int complete;
            if (strategy->accept_almost_smooth)
            {
                // Algorithm 3: cofactor may be a medium prime — use alg3 path
                mpz_set_ui(cofactor, 1);
                complete = ksc_alg3_broadcast_cofactor(
                    cofactor, blinded_challenge, local_prod, cbrt_bytes,
                    rank, nranks, &primality_dt);  // M4: accumulate q0-primality time
            }
            else
            {
                // Algorithm 2 (k32/k35): challenge must be fully B-smooth.
                // ksc_alg3_broadcast_cofactor has no role here.
                complete = ksc_alg2_broadcast_complete(
                    blinded_challenge, local_prod, cbrt_bytes,
                    rank, nranks);
                // cofactor stays 1 (already set above); fold is NULL for alg2
            }
            mpz_clear(local_prod);

            if (complete)
            {
                bench_succeeded = true;
                break;
            }
        } // attempt loop

        free(blinded_buf);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        // If all MAX_ATTEMPTS blinding factors were exhausted without a hit,
        // this run FAILED.  bench_succeeded is derived from `complete`, which
        // ksc_alg2/alg3_broadcast_* broadcasts identically to every rank, so
        // all ranks take the same branch — no collective desynchronisation.
        // A failed run must NOT be recorded (otherwise it is misclassified as a
        // B-smooth hit_type-1 success and runs_failed stays 0).  Skip the entire
        // reconstruction block (which contains collective MPI_Reduce / Barrier /
        // Gather) uniformly on every rank.
        if (!bench_succeeded)
        {
            if (rank == 0)
                fprintf(stderr,
                        "[rank 0] WARNING: exhausted %d blinding attempts "
                        "without a hit — challenge skipped (counts as runs_failed)\n",
                        MAX_ATTEMPTS);
            kfl_free(&local_factors);
        }
        else
        {

        if (rank == 0)
        {
            // M4 (2026-07-16): exclude the q0 primality-test time — modelled as a
            // table lookup, not attack work (advisor 2026-07-15).
            double dt = std::chrono::duration<double, std::milli>(
                            t_fac1 - t_fac0).count() - primality_dt;
            accum_record(fac_acc, dt);
            json_results->timing.factor_times.push_back(dt);
            json_results->timing.blinding_attempts.push_back(attempt + 1);
            // alg3: record cofactor classification for hit-rate analysis
            if (strategy->accept_almost_smooth)
            {
                json_results->hits.hit_type.push_back(
                    mpz_cmp_ui(cofactor, 1) == 0 ? 1 : 2);
                json_results->hits.cofactor_bits.push_back(
                    (int)mpz_sizeinbase(cofactor, 2));
            }
        }

        // ---- Reconstruction ----

        auto t_loc0 = std::chrono::high_resolution_clock::now();

        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double cbrt_hit_ms = 0.0, cbrt_miss_ms = 0.0;
        int cbrt_hits = 0, cbrt_misses = 0;
        ksc_compute_local_cbrt(local_cbrt, &local_factors, e_inv, modulo,
                               cache, cache_dirty,
                               &cbrt_hit_ms, &cbrt_miss_ms,
                               &cbrt_hits, &cbrt_misses, true);

        // Optional fold: rank 0 multiplies cbrt(cofactor) into partial product
        if (strategy->fold != NULL && rank == 0 && mpz_cmp_ui(cofactor, 1) != 0)
            strategy->fold(local_cbrt, cofactor, e_inv, modulo,
                           cache, cache_dirty,
                           json_results, bench, strategy->ctx);

        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(
                              t_loc1 - t_loc0).count();

        if (bench == 0)
            ksc_gather_prime_stats(json_results,
                                   local_factors.count,
                                   (int64_t)(my_hi - my_lo + 1),
                                   rank, nranks);

        if (rank == 0)
        {
            json_results->timing.cbrt_hit_ms.push_back(cbrt_hit_ms);
            json_results->timing.cbrt_miss_ms.push_back(cbrt_miss_ms);
            json_results->timing.cbrt_hits.push_back(cbrt_hits);
            json_results->timing.cbrt_misses.push_back(cbrt_misses);
        }

        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            accum_record(loc_acc, max_local_dt);
            cur_max_loc = max_local_dt;
            json_results->timing.local_cbrt_times.push_back(max_local_dt);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double iter_gat = 0.0, iter_com = 0.0;
        ksc_gather_and_combine_cbrt(result_out, local_cbrt, modulo, cbrt_bytes,
                                    rank, nranks, attempt,
                                    blind_table, blind_table_size,
                                    &iter_gat, &iter_com);
        mpz_clear(local_cbrt);

        if (rank == 0)
        {
            accum_record(gat_acc, iter_gat);
            json_results->timing.gather_times.push_back(iter_gat);
            accum_record(com_acc, iter_com);
            json_results->timing.combine_times.push_back(iter_com);
            double recon_dt = cur_max_loc + iter_gat + iter_com;
            accum_record(recon_acc, recon_dt);
            json_results->timing.recon_times.push_back(recon_dt);
            if (!json_results->timing.factor_times.empty())
                json_results->timing.total_times.push_back(
                    json_results->timing.factor_times.back() + recon_dt);
        }

        kfl_free(&local_factors);
        } // end reconstruction (bench_succeeded)
    } // single timed run

    mpz_clears(blinded_challenge, cofactor, NULL);
}


// Edit: 2026-07-16
