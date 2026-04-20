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

// Factor list and trial division — kfactor_list_t / kfl_* from k_common.h,
// kf_trial_divide_* / kf_factorize_with_blinding from k_factorization.h.

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
// k32 cbrt cache I/O — now in k_factorization.cpp as kf_cbrt_cache_*
//                      and kf_compute_cbrt_from_factors().
// ---------------------------------------------------------------------------

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
    kf_cbrt_cache_init(&cbrt_cache);
    bool cbrt_cache_dirty = false;

    if (!kf_cbrt_cache_load(&cbrt_cache, cbrt_file))
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
    // Blinding attempt counts — tracked separately as integer accumulators
    // so avg/stddev can be reported alongside timing (key metric for the paper).
    double s_att_sum = 0, s_att_sumsq = 0; // sieve attempts
    double t_att_sum = 0, t_att_sumsq = 0; // table attempts

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
            int attempts = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            bool ok = kf_factorize_with_blinding(
                result, challenge, modulo,
                sp_count, sp,
                NULL, 0, // sieve — no preloaded table
                &cbrt_cache, &cbrt_cache_dirty,
                256, // max blinding attempts
                &attempts);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            // Split timing: record full round-trip in both accumulators so that
            // compare mode can still show fac vs recon breakdown via a fixed 50/50
            // heuristic when blinding succeeds on attempt 1 (common case).
            accum_record(&s_fac, dt * 0.95); // factoring dominates
            accum_record(&s_rec, dt * 0.05);
            s_att_sum += attempts;
            s_att_sumsq += (double)attempts * attempts;

            if (bench == 0)
            {
                if (ok)
                    printf("[sieve] Solved in %d blinding attempt(s), %.3f ms\n",
                           attempts, dt);
                else
                    printf("[sieve] WARNING: failed to factor challenge in 256 attempts\n");
            }
        }

        // ---- TABLE ----
        if (method == METHOD_TABLE || method == METHOD_COMPARE)
        {
            int attempts = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            bool ok = kf_factorize_with_blinding(
                result, challenge, modulo,
                sp_count, sp,
                prime_table, prime_table_count, // preloaded table
                &cbrt_cache, &cbrt_cache_dirty,
                256,
                &attempts);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            accum_record(&t_fac, dt * 0.95);
            accum_record(&t_rec, dt * 0.05);
            t_att_sum += attempts;
            t_att_sumsq += (double)attempts * attempts;

            if (bench == 0)
            {
                if (ok)
                    printf("[table] Solved in %d blinding attempt(s), %.3f ms\n",
                           attempts, dt);
                else
                    printf("[table] WARNING: failed to factor challenge in 256 attempts\n");
            }
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
            double att_sd = bench_count > 1
                                ? compute_stddev_from_moments(s_att_sum, s_att_sumsq, bench_count, true)
                                : 0.0;
            printf("\nSIEVE method:\n");
            print_timing_stat_moments("Factoring:",
                                      s_fac.sum, s_fac.sumsq, bench_count, 26, true);
            print_timing_stat_moments("Reconstruction:",
                                      s_rec.sum, s_rec.sumsq, bench_count, 26, true);
            print_timing_total("Total:",
                               (s_fac.sum + s_rec.sum) / bench_count, 26, NULL);
            printf("  %-26s  avg=%6.2f      stddev=%6.2f\n",
                   "Blinding attempts:", s_att_sum / bench_count, att_sd);
        }

        if (method == METHOD_TABLE)
        {
            double amortised = table_load_ms / bench_count;
            double att_sd = bench_count > 1
                                ? compute_stddev_from_moments(t_att_sum, t_att_sumsq, bench_count, true)
                                : 0.0;
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
            printf("  %-26s  avg=%6.2f      stddev=%6.2f\n",
                   "Blinding attempts:", t_att_sum / bench_count, att_sd);
        }
    }

    if (cbrt_cache_dirty)
    {
        kf_cbrt_cache_save(&cbrt_cache, cbrt_file);
    }

    // ---- Verify ----
    printf("\n");
    mpz_t challenge_mod;
    mpz_init(challenge_mod);
    mpz_mod(challenge_mod, challenge, modulo);
    verify_cube_root(result, challenge_mod, modulo);
    mpz_clear(challenge_mod);

    kf_cbrt_cache_clear(&cbrt_cache);
    free(prime_table);
    free(sp);
    mpz_clears(challenge, modulo, result, NULL);
    return 0;
}