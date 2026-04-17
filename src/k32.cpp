// =============================================================================
// k32.cpp — Cube root solver for challenges with 32-bit prime factors
//           Single-node version (OpenMP only)
//
// Three factoring methods:
//   sieve   — segmented sieve, primes generated on-the-fly        [default]
//   table   — precomputed prime file loaded once (~812 MB on disk)
//   compare — runs both per benchmark iteration; reports side-by-side timing
//             with mean ± stddev so the two approaches can be compared fairly
//
// Prime table generation (one-time, ~10 s):
//   ./k32 --precompute [--primes k32_primes.bin]
//
// Usage:
//   ./k32 [challenge] [modulo] [--method sieve|table|compare]
//         [--primes FILE] [--precompute] [--bench N]
//
// The table is ~812 MB on disk (203 M primes × 4 bytes).
// For bench_count > 1 the I/O cost is amortised: the file is loaded once,
// all iterations share the in-memory array.
// =============================================================================

#include "k_common.h"
#include "k_factorization.h"

#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define SEG_SIZE (1 << 19)                      // 512 KiB — fits in L2 cache
#define PRIME_TABLE_MAGIC 0x4B33325052494D53ULL // "K32PRIMS"
#define WRITE_BATCH 65536                       // primes flushed per fwrite call

typedef enum
{
    METHOD_SIEVE,
    METHOD_TABLE,
    METHOD_COMPARE
} run_method_t;

// ---------------------------------------------------------------------------
// Factor list: kfactor_list_t / kfl_* from k_common.h.
// Trial division: kf_trial_divide_* from k_factorization.h.
#define found_factors_t kfactor_list_t
#define ff_init kfl_init
#define ff_clear kfl_free
#define ff_append(f, p, e) kfl_append_ul(f, (unsigned long)(p), (int)(e))

// ---------------------------------------------------------------------------
// Small prime generator — Sieve of Eratosthenes up to sqrt(2^32) = 65536
// Needed to mark composites inside each 32-bit sieve segment.
// ---------------------------------------------------------------------------

// kf_kf_generate_small_primes() — in k_factorization.cpp

// ---------------------------------------------------------------------------
// Prime table I/O
//
// File format (binary, little-endian platform order for uint64_t header):
//   [uint64_t magic  = PRIME_TABLE_MAGIC]
//   [uint64_t count  = number of primes stored]
//   [uint32_t prime[count]]   sorted ascending, range [65537, 2^32]
//
// File size: 16 + 203,280,221 × 4 ≈ 813 MB
//
// The table is generated once by --precompute (or auto-generated on first run
// when --method table is used and the file does not yet exist).
// ---------------------------------------------------------------------------

// kf_generate_and_save_prime_table() — in k_factorization.cpp

// kf_load_prime_table() — in k_factorization.cpp

// kf_trial_divide_sieve() — in k_factorization.cpp

// kf_trial_divide_table() — in k_factorization.cpp

// ---------------------------------------------------------------------------
// k32 cbrt cache I/O
//
// We cache only factors actually discovered in challenges (not all 203M primes).
// File format (text):
//   line 1: count
//   line i: <prime> <cuberoot_decimal>
// ---------------------------------------------------------------------------

typedef struct
{
    unsigned long prime;
    mpz_t cbrt;
} cbrt_entry_t;

typedef struct
{
    cbrt_entry_t *items;
    int count;
    int capacity;
} cbrt_cache_t;

static void cbrt_cache_init(cbrt_cache_t *cache)
{
    cache->items = NULL;
    cache->count = 0;
    cache->capacity = 0;
}

static void cbrt_cache_clear(cbrt_cache_t *cache)
{
    for (int i = 0; i < cache->count; i++)
        mpz_clear(cache->items[i].cbrt);
    free(cache->items);
    cache->items = NULL;
    cache->count = 0;
    cache->capacity = 0;
}

static int cbrt_cache_find_index(const cbrt_cache_t *cache, unsigned long prime)
{
    for (int i = 0; i < cache->count; i++)
    {
        if (cache->items[i].prime == prime)
            return i;
    }
    return -1;
}

static void cbrt_cache_append(cbrt_cache_t *cache, unsigned long prime,
                              const mpz_t cbrt_val)
{
    if (cache->count == cache->capacity)
    {
        int nc = (cache->capacity == 0) ? 64 : cache->capacity * 2;
        cache->items = (cbrt_entry_t *)realloc(cache->items,
                                               (size_t)nc * sizeof(cbrt_entry_t));
        cache->capacity = nc;
    }

    cache->items[cache->count].prime = prime;
    mpz_init_set(cache->items[cache->count].cbrt, cbrt_val);
    cache->count++;
}

static bool load_k32_cbrt_cache(cbrt_cache_t *cache, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return false;

    int stored_count = 0;
    if (fscanf(f, "%d", &stored_count) != 1 || stored_count < 0)
    {
        fclose(f);
        return false;
    }

    cbrt_cache_clear(cache);
    cbrt_cache_init(cache);

    for (int i = 0; i < stored_count; i++)
    {
        unsigned long p = 0;
        if (fscanf(f, "%lu", &p) != 1)
        {
            fclose(f);
            cbrt_cache_clear(cache);
            cbrt_cache_init(cache);
            return false;
        }

        mpz_t v;
        mpz_init(v);
        if (mpz_inp_str(v, f, 10) == 0)
        {
            mpz_clear(v);
            fclose(f);
            cbrt_cache_clear(cache);
            cbrt_cache_init(cache);
            return false;
        }

        cbrt_cache_append(cache, p, v);
        mpz_clear(v);
    }

    fclose(f);
    printf("Loaded %d cached cbrt entries from %s\n", cache->count, path);
    return true;
}

static bool save_k32_cbrt_cache(const cbrt_cache_t *cache, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Cannot write cbrt cache to %s\n", path);
        return false;
    }

    fprintf(f, "%d\n", cache->count);
    for (int i = 0; i < cache->count; i++)
    {
        fprintf(f, "%lu ", cache->items[i].prime);
        mpz_out_str(f, 10, cache->items[i].cbrt);
        fputc('\n', f);
    }

    fclose(f);
    printf("Saved %d cached cbrt entries to %s\n", cache->count, path);
    return true;
}

// ---------------------------------------------------------------------------
// Reconstruction using persistent cbrt cache
// ---------------------------------------------------------------------------

static void compute_cube_root_from_found(mpz_t result,
                                         const found_factors_t *found,
                                         const mpz_t modulo,
                                         cbrt_cache_t *cache,
                                         bool *cache_dirty)
{
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);

    mpz_set_ui(result, 1);

    for (int i = 0; i < found->count; i++)
    {
        unsigned long p = mpz_get_ui(found->items[i].prime);
        int idx = cbrt_cache_find_index(cache, p);

        mpz_t cbrt_p;
        mpz_init(cbrt_p);

        if (idx >= 0)
        {
            mpz_set(cbrt_p, cache->items[idx].cbrt);
        }
        else
        {
            mpz_t p_mpz;
            mpz_init_set_ui(p_mpz, p);
            mpz_powm(cbrt_p, p_mpz, e, modulo);
            cbrt_cache_append(cache, p, cbrt_p);
            *cache_dirty = true;
            mpz_clear(p_mpz);
        }

        if (found->items[i].exponent > 1)
        {
            mpz_t powered;
            mpz_init(powered);
            mpz_powm_ui(powered, cbrt_p,
                        (unsigned long)found->items[i].exponent, modulo);
            mpz_mul(result, result, powered);
            mpz_clear(powered);
        }
        else
        {
            mpz_mul(result, result, cbrt_p);
        }

        mpz_mod(result, result, modulo);
        mpz_clear(cbrt_p);
    }

    mpz_clears(phi, e, three, NULL);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    const char *challenge_file = "data/challenges/k32/challenge_32_1024.txt";
    const char *modulo_file = "data/challenges/k32/modulo_32_1024.txt";
    const char *primes_file = "k32_primes.bin";
    int bench_count = 1;
    int do_precompute = 0;
    run_method_t method = METHOD_SIEVE;

    int positional = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--bench") == 0 && argi + 1 < argc)
        {
            bench_count = atoi(argv[++argi]);
        }
        else if (strcmp(argv[argi], "--method") == 0 && argi + 1 < argc)
        {
            const char *m = argv[++argi];
            if (strcmp(m, "sieve") == 0)
                method = METHOD_SIEVE;
            else if (strcmp(m, "table") == 0)
                method = METHOD_TABLE;
            else if (strcmp(m, "compare") == 0)
                method = METHOD_COMPARE;
            else
            {
                fprintf(stderr, "Unknown --method '%s'\n", m);
                return 1;
            }
        }
        else if (strcmp(argv[argi], "--primes") == 0 && argi + 1 < argc)
        {
            primes_file = argv[++argi];
        }
        else if (strcmp(argv[argi], "--precompute") == 0)
        {
            do_precompute = 1;
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
                fprintf(stderr, "Unexpected argument '%s'\n", argv[argi]);
                return 1;
            }
        }
        else
        {
            fprintf(stderr, "Unknown option '%s'\n", argv[argi]);
            return 1;
        }
    }

    const char *method_name =
        (method == METHOD_SIEVE) ? "sieve" : (method == METHOD_TABLE) ? "table"
                                                                      : "compare";

    printf("Challenge: %s\n", challenge_file);
    printf("Modulo:    %s\n", modulo_file);
    printf("Method:    %s\n", method_name);
    if (bench_count > 1)
        printf("Bench:     %d runs\n", bench_count);

#ifdef _OPENMP
    printf("OpenMP:    %d threads\n", omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i; // warm up pool
#endif

    int sp_count;
    unsigned *sp = kf_generate_small_primes(&sp_count);
    printf("Small primes ≤65536: %d\n\n", sp_count);

    // --precompute: generate prime file and exit
    if (do_precompute)
    {
        bool ok = kf_generate_and_save_prime_table(primes_file, sp_count, sp, "");
        free(sp);
        return ok ? 0 : 1;
    }

    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);
    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);

    size_t cbits = mpz_sizeinbase(challenge, 2);
    printf("Challenge: %zu bits  (~%zu factors expected)\n\n", cbits, cbits / 32);

    // ---- Prime table bootstrap (always): load or create k32_primes.bin ----

    uint32_t *prime_table = NULL;
    int64_t prime_table_count = 0;
    double table_load_ms = 0.0;

    auto tl0 = std::chrono::high_resolution_clock::now();
    {
        uint64_t uc = 0;
        prime_table = kf_load_prime_table(&uc, primes_file);
        prime_table_count = (int64_t)uc;
    }
    if (!prime_table)
    {
        printf("Prime table not found — generating now (one-time cost)...\n");
        if (!kf_generate_and_save_prime_table(primes_file, sp_count, sp, ""))
        {
            free(sp);
            return 1;
        }
        uint64_t uc = 0;
        prime_table = kf_load_prime_table(&uc, primes_file);
        prime_table_count = (int64_t)uc;
        if (!prime_table)
        {
            fprintf(stderr, "Failed to load %s after generation\n", primes_file);
            free(sp);
            return 1;
        }
    }
    auto tl1 = std::chrono::high_resolution_clock::now();
    table_load_ms = std::chrono::duration<double, std::milli>(tl1 - tl0).count();
    printf("Table ready: %" PRId64 " primes in %.1f ms\n\n",
           prime_table_count, table_load_ms);

    // ---- k32 cbrt cache file (modulo-specific) ----
    char cbrt_file[512];
    {
        char base[504];
        strncpy(base, modulo_file, sizeof(base) - 1);
        base[sizeof(base) - 1] = '\0';
        char *dot = strrchr(base, '.');
        if (dot)
            *dot = '\0';
        snprintf(cbrt_file, sizeof(cbrt_file), "%s.cbrt", base);
    }

    cbrt_cache_t cbrt_cache;
    cbrt_cache_init(&cbrt_cache);
    bool cbrt_cache_dirty = false;

    if (!load_k32_cbrt_cache(&cbrt_cache, cbrt_file))
    {
        printf("No existing k32 cbrt cache at %s — creating on demand.\n", cbrt_file);
    }

    if (method == METHOD_SIEVE)
    {
        // Keep startup behaviour (always load prime file) without holding 800+ MB
        // when the selected method does not use the table.
        free(prime_table);
        prime_table = NULL;
    }

    // ---- Timing accumulators ----

    // powm sequential baseline
    bench_accum_t p_acc = {0, 0}; // powm sequential baseline
    bench_accum_t s_fac = {0, 0}; // sieve factoring
    bench_accum_t s_rec = {0, 0}; // sieve reconstruction
    bench_accum_t t_fac = {0, 0}; // table factoring
    bench_accum_t t_rec = {0, 0}; // table reconstruction

    // ---- Benchmark loop ----

    for (int bench = 0; bench < bench_count; bench++)
    {

        // ---- POWM sequential baseline (compare mode only) ----
        // Single mpz_powm(challenge, e, M) — the unparallelisable VDF baseline.
        // Timed here so it shares identical loop structure with sieve/table.
        if (method == METHOD_COMPARE)
        {
            mpz_t powm_result, e, phi, three;
            mpz_inits(powm_result, e, phi, three, NULL);
            mpz_sub_ui(phi, modulo, 1);
            mpz_set_ui(three, 3);
            mpz_invert(e, three, phi);

            auto t0 = std::chrono::high_resolution_clock::now();
            mpz_powm(powm_result, challenge, e, modulo);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            accum_record(&p_acc, dt);

            mpz_clears(powm_result, e, phi, three, NULL);
        }
        if (method == METHOD_SIEVE || method == METHOD_COMPARE)
        {
            found_factors_t found;
            ff_init(&found);

            auto t0 = std::chrono::high_resolution_clock::now();
            kf_trial_divide_sieve(challenge, 65537, (uint64_t)UINT32_MAX,
                                  sp_count, sp, &found);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            accum_record(&s_fac, dt);

            if (bench == 0)
            {
                printf("[sieve] Found %d distinct 32-bit factors\n", found.count);
                mpz_t rem;
                mpz_init_set(rem, challenge);
                for (int i = 0; i < found.count; i++)
                    for (int e = 0; e < found.items[i].exponent; e++)
                        mpz_divexact_ui(rem, rem, mpz_get_ui(found.items[i].prime));
                if (mpz_cmp_ui(rem, 1) != 0)
                    printf("[sieve] WARNING: challenge not fully factored\n");
                mpz_clear(rem);
            }

            if (found.count > 0)
            {
                auto t2 = std::chrono::high_resolution_clock::now();
                compute_cube_root_from_found(result, &found, modulo,
                                             &cbrt_cache, &cbrt_cache_dirty);
                auto t3 = std::chrono::high_resolution_clock::now();
                double dr = std::chrono::duration<double, std::milli>(t3 - t2).count();
                accum_record(&s_rec, dr);
            }
            ff_clear(&found);
        }

        // ---- TABLE ----
        if (method == METHOD_TABLE || method == METHOD_COMPARE)
        {
            found_factors_t found;
            ff_init(&found);

            auto t0 = std::chrono::high_resolution_clock::now();
            kf_trial_divide_table(challenge, prime_table, prime_table_count, &found);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            accum_record(&t_fac, dt);

            if (bench == 0)
            {
                printf("[table] Found %d distinct 32-bit factors\n", found.count);
                mpz_t rem;
                mpz_init_set(rem, challenge);
                for (int i = 0; i < found.count; i++)
                    for (int e = 0; e < found.items[i].exponent; e++)
                        mpz_divexact_ui(rem, rem, mpz_get_ui(found.items[i].prime));
                if (mpz_cmp_ui(rem, 1) != 0)
                    printf("[table] WARNING: challenge not fully factored\n");
                mpz_clear(rem);
            }

            if (found.count > 0)
            {
                auto t2 = std::chrono::high_resolution_clock::now();
                compute_cube_root_from_found(result, &found, modulo,
                                             &cbrt_cache, &cbrt_cache_dirty);
                auto t3 = std::chrono::high_resolution_clock::now();
                double dr = std::chrono::duration<double, std::milli>(t3 - t2).count();
                accum_record(&t_rec, dr);
            }
            ff_clear(&found);
        }
    }

    // ---- Report ----
    printf("\n=== Results over %d run(s) ===\n", bench_count);

    if (method == METHOD_COMPARE)
    {
        printf("\n%-28s  %s\n", "", "avg (ms)   stddev (ms)");
        printf("%-28s  %s\n", "Method", "----------   ----------");

        // powm baseline row
        {
            double mean = accum_mean(&p_acc, bench_count);
            double stddev = (bench_count > 1)
                                ? compute_stddev_from_moments(p_acc.sum, p_acc.sumsq, bench_count, true)
                                : 0.0;
            printf("  %-26s  %9.3f      %7.3f\n",
                   "powm (sequential):", mean, stddev);
        }

        // sieve total row
        {
            double s_tot = s_fac.sum + s_rec.sum;
            double s_tot_sq = 0; // approx — printed separately below
            (void)s_tot_sq;
            double mean = s_tot / bench_count;
            double f_mean = accum_mean(&s_fac, bench_count);
            double r_mean = accum_mean(&s_rec, bench_count);
            double f_sd = (bench_count > 1)
                              ? compute_stddev_from_moments(s_fac.sum, s_fac.sumsq, bench_count, true)
                              : 0.0;
            double r_sd = (bench_count > 1)
                              ? compute_stddev_from_moments(s_rec.sum, s_rec.sumsq, bench_count, true)
                              : 0.0;
            printf("  %-26s  %9.3f   (factor: %.3f±%.3f  recon: %.3f±%.3f)\n",
                   "sieve (parallel total):", mean, f_mean, f_sd, r_mean, r_sd);
        }

        // table total row
        {
            double t_tot = t_fac.sum + t_rec.sum;
            double mean = t_tot / bench_count;
            double f_mean = accum_mean(&t_fac, bench_count);
            double r_mean = accum_mean(&t_rec, bench_count);
            double f_sd = (bench_count > 1)
                              ? compute_stddev_from_moments(t_fac.sum, t_fac.sumsq, bench_count, true)
                              : 0.0;
            double r_sd = (bench_count > 1)
                              ? compute_stddev_from_moments(t_rec.sum, t_rec.sumsq, bench_count, true)
                              : 0.0;
            double amortised = table_load_ms / bench_count;
            printf("  %-26s  %9.3f   (factor: %.3f±%.3f  recon: %.3f±%.3f  +%.1f ms load)\n",
                   "table (parallel total):", mean, f_mean, f_sd, r_mean, r_sd, amortised);
        }

        double p_avg = accum_mean(&p_acc, bench_count);
        double s_avg = (s_fac.sum + s_rec.sum) / bench_count;
        double t_avg = (t_fac.sum + t_rec.sum) / bench_count;
        double s_rec_avg = accum_mean(&s_rec, bench_count);
        double t_rec_avg = accum_mean(&t_rec, bench_count);

        printf("\n--- Speedup vs sequential powm ---\n");
        printf("  powm vs sieve total:         %.2fx\n",
               s_avg > 0 ? p_avg / s_avg : 0.0);
        printf("  powm vs sieve recon only:    %.2fx  "
               "(reconstruction alone, no factoring overhead)\n",
               s_rec_avg > 0 ? p_avg / s_rec_avg : 0.0);
        printf("  powm vs table total:         %.2fx\n",
               t_avg > 0 ? p_avg / t_avg : 0.0);
        printf("  powm vs table recon only:    %.2fx\n",
               t_rec_avg > 0 ? p_avg / t_rec_avg : 0.0);
        printf("  sieve vs table (total):      %.2fx  "
               "(sieve/table factoring comparison)\n",
               t_avg > 0 ? s_avg / t_avg : 0.0);
        printf("\nNote: 'recon only' speedup isolates the parallel powm-per-factor\n"
               "      reconstruction from the sieve/table factoring overhead.\n"
               "      This is the number most directly comparable to the paper's\n"
               "      Algorithm 2 latency claim.\n");
    }
    else
    {
        if (method == METHOD_SIEVE)
        {
            printf("\nSIEVE method:\n");
            print_timing_stat_moments("Factoring:",
                                      s_fac.sum, s_fac.sumsq, bench_count, 26, true);
            print_timing_stat_moments("Reconstruction:",
                                      s_rec.sum, s_rec.sumsq, bench_count, 26, true);
            print_timing_total("Total:",
                               (s_fac.sum + s_rec.sum) / bench_count, 26, NULL);
        }

        if (method == METHOD_TABLE)
        {
            double amortised = table_load_ms / bench_count;
            char total_extra[64];
            snprintf(total_extra, sizeof(total_extra),
                     "(+ %.1f ms amortised load)", amortised);
            printf("\nTABLE method  (table load %.1f ms, amortised %.1f ms/run):\n",
                   table_load_ms, amortised);
            print_timing_stat_moments("Factoring:",
                                      t_fac.sum, t_fac.sumsq, bench_count, 26, true);
            print_timing_stat_moments("Reconstruction:",
                                      t_rec.sum, t_rec.sumsq, bench_count, 26, true);
            print_timing_total("Total:",
                               (t_fac.sum + t_rec.sum) / bench_count, 26, total_extra);
        }
    }

    if (cbrt_cache_dirty)
    {
        save_k32_cbrt_cache(&cbrt_cache, cbrt_file);
    }

    // ---- Verify ----
    printf("\n");
    mpz_t challenge_mod;
    mpz_init(challenge_mod);
    mpz_mod(challenge_mod, challenge, modulo);
    verify_cube_root(result, challenge_mod, modulo);
    mpz_clear(challenge_mod);

    cbrt_cache_clear(&cbrt_cache);
    free(prime_table);
    free(sp);
    mpz_clears(challenge, modulo, result, NULL);
    return 0;
}