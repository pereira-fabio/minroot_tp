// =============================================================================
// k16.cpp — Cube root solver for challenges with 16-bit prime factors
//
// Strategy: precomputed table + parallel trial division
//
// The challenge is a product of primes <= 2^16 (at most 65535).
// There are exactly 6,542 such primes. For a fixed modulo M we precompute:
//   cbrt_table[i] = primes[i]^e mod M,   e = 3^-1 mod (M-1)
// Both the prime list and the cbrt table are saved to disk keyed by the
// modulo filename, so neither the sieve nor the powm loop runs twice.
//
// Per-challenge:
//   1. Load prime list + cbrt table from disk (build and save on first run)
//   2. Export challenge as raw bytes (once per run)
//   3. For each prime q: compute challenge mod q via byte arithmetic (parallel)
//   4. For divisors: record exponent
//   5. Reconstruct: prod cbrt_table[i]^exp_i  mod M   (homomorphism)
//
// Methods:
//   --method factor      (default) — factorization + precomputed table, parallel
//   --method powm                  — naive Fermat baseline (one powm on challenge)
//   --method powm-recon            — factor first, then time reconstruction using
//                                    one sequential mpz_powm per factor (no OMP)
//                                    isolates raw powm cost 
//
// Benchmarking:
//   --bench N   runs N iterations, reports mean ± standard deviation
// =============================================================================

#include "k_common.h"

#include <climits>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Prime list disk I/O
//
// Saves/loads the 6,542 k16 primes as raw binary (sizeof(unsigned long) each).
// Avoids re-running the sieve on every execution (~0.05ms vs ~0.5ms).
//
// File format: 4-byte int header (count), then raw unsigned long array.
// Filename convention: modulo_16_2048.txt  →  modulo_16_2048.primes
// ---------------------------------------------------------------------------

static bool save_prime_list(const unsigned long *primes, int count,
                            const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
    {
        fprintf(stderr, "Warning: Cannot write prime list to %s\n", filename);
        return false;
    }
    fwrite(&count, sizeof(int), 1, f);
    fwrite(primes, sizeof(unsigned long), (size_t)count, f);
    fclose(f);
    printf("Saved %d primes to %s\n", count, filename);
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

    unsigned long *primes = (unsigned long *)malloc(
        (size_t)stored_count * sizeof(unsigned long));
    if (!primes)
    {
        fclose(f);
        return nullptr;
    }

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
        METHOD_POWM_RECON
    };

    const char *challenge_file = "data/challenges/k16/challenge_2_16_2048.txt";
    const char *modulo_file = "data/challenges/k16/modulo_16_2048.txt";
    solver_method_t method = METHOD_FACTOR;
    int bench_count = 1;

    // -------------------------------------------------------------------------
    // Argument parsing
    //   <challenge_file> <modulo_file> [--method factor|powm|powm-recon]
    //                                  [--bench N]
    // -------------------------------------------------------------------------
    int positional_index = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--method") == 0)
        {
            if (argi + 1 >= argc)
            {
                fprintf(stderr, "Error: Missing value for --method\n");
                return 1;
            }
            if (strcmp(argv[argi + 1], "factor") == 0)
                method = METHOD_FACTOR;
            else if (strcmp(argv[argi + 1], "powm") == 0)
                method = METHOD_POWM;
            else if (strcmp(argv[argi + 1], "powm-recon") == 0)
                method = METHOD_POWM_RECON;
            else
            {
                fprintf(stderr, "Error: Invalid --method '%s' "
                                "(use factor, powm, or powm-recon)\n",
                        argv[argi + 1]);
                return 1;
            }
            argi++;
            continue;
        }

        if (strcmp(argv[argi], "--bench") == 0)
        {
            if (argi + 1 >= argc)
            {
                fprintf(stderr, "Error: Missing value for --bench\n");
                return 1;
            }
            bench_count = atoi(argv[++argi]);
            continue;
        }

        if (positional_index == 0)
            challenge_file = argv[argi];
        else if (positional_index == 1)
            modulo_file = argv[argi];
        else
        {
            fprintf(stderr, "Error: Unexpected argument '%s'\n", argv[argi]);
            return 1;
        }
        positional_index++;
    }

    printf("Challenge: %s\n", challenge_file);
    printf("Modulo:    %s\n", modulo_file);
    printf("Method:    %s\n",
           method == METHOD_POWM ? "powm" : method == METHOD_POWM_RECON ? "powm-recon"
                                                                        : "factor");
    if (bench_count > 1)
        printf("Bench:     %d runs\n", bench_count);

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);
    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);

    mpz_t result;
    mpz_init(result);

    // declared here so cleanup at the bottom is always valid
    mpz_t *cbrt_table = nullptr;
    int prime_count = 0;
    unsigned long *primes = nullptr;

#ifdef _OPENMP
    printf("OpenMP:    %d threads\n", omp_get_max_threads());
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    size_t cbits = mpz_sizeinbase(challenge, 2);
    printf("Challenge: %zu bits  (~%zu factors expected)\n\n", cbits, cbits);

    // =========================================================================
    // METHOD_POWM — naive Fermat baseline
    //
    // Computes challenge^e mod M where e = 3^-1 mod (M-1).
    // Cost: O(log M) squarings on the full challenge — no factorization.
    // Used only to measure the "without parallelism" reference time.
    // =========================================================================
    if (method == METHOD_POWM)
    {
        printf("\nComputing cube root using direct modular exponentiation...\n");
        auto t0 = std::chrono::high_resolution_clock::now();
        compute_cube_root_mod_prime(result, challenge, modulo);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("Execution time: %.3f ms\n",
               std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // =========================================================================
    // METHOD_FACTOR and METHOD_POWM_RECON
    //
    // Both methods share the same factorization phase (Steps 1–3).
    // They differ only in Step 4 (reconstruction):
    //
    //   METHOD_FACTOR:
    //     Parallel reconstruction via compute_cube_root_from_factors().
    //     Uses precomputed cbrt_table — no powm calls during reconstruction.
    //
    //   METHOD_POWM_RECON:
    //     Sequential reconstruction — one mpz_powm per factor, no OpenMP,
    //     no table lookup. Purpose: isolate the raw cost of powm so the
    //     supervisor can compare against METHOD_FACTOR and evaluate whether
    //     a lower-level powm (e.g. mpn_powm) would be worth pursuing.
    //     See k_common.cpp for why mpn_powm is not used.
    // =========================================================================
    else
    {
        printf("\nComputing cube root using factorization...\n");

        // -----------------------------------------------------------------------
        // Step 1: load or build prime list and cbrt table
        //
        // Sidecar files are derived from the modulo filename by replacing the
        // extension:
        //   modulo_16_2048.txt  →  modulo_16_2048.primes  (binary)
        //                      →  modulo_16_2048.cbrt     (decimal text)
        //
        // Prime list: raw binary, loaded in ~0.05ms vs ~0.5ms to re-sieve.
        // cbrt table: one mpz_powm per prime, expensive for large M.
        //             Saved once and reloaded on subsequent runs.
        // -----------------------------------------------------------------------

        // strip extension from modulo_file to build modulo-specific cbrt file
        char base[504]; // 504 + 5 (".cbrt") = 509 < 512 — no truncation
        strncpy(base, modulo_file, sizeof(base) - 1);
        base[sizeof(base) - 1] = '\0';
        char *dot = strrchr(base, '.');
        if (dot)
            *dot = '\0';

        // shared prime cache: created once, then reused across all k16 runs
        const char *prime_file = "k16_primes.bin";
        char table_file[512];
        snprintf(table_file, sizeof(table_file), "%s.cbrt", base);

        // load or generate prime list
        auto prime_start = std::chrono::high_resolution_clock::now();
        primes = load_prime_list(&prime_count, prime_file);
        if (!primes)
        {
            primes = generate_primes_up_to(UINT16_MAX, &prime_count);
            save_prime_list(primes, prime_count, prime_file);
        }
        auto prime_end = std::chrono::high_resolution_clock::now();
        double prime_load_ms = std::chrono::duration<double, std::milli>(
                                   prime_end - prime_start)
                                   .count();

        // load or generate cbrt table
        cbrt_table = (mpz_t *)malloc((size_t)prime_count * sizeof(mpz_t));

        if (!load_cbrt_table(cbrt_table, prime_count, table_file))
        {
            // first run for this modulo — build and persist the table
            precompute_cbrt_table(cbrt_table, primes, prime_count, modulo);
            save_cbrt_table(cbrt_table, prime_count, table_file);
        }

        printf("Prime cache ready: %d primes in %.1f ms\n", prime_count,
               prime_load_ms);
        printf("Cbrt table ready: %s\n\n",
               table_file);

        // export challenge as raw bytes once — reused across all bench iterations
        // the byte-arithmetic divisibility check reads these; they never change
        size_t nbytes;
        unsigned char *challenge_bytes = (unsigned char *)mpz_export(
            NULL, &nbytes, 1, 1, 0, 0, challenge);

        // -----------------------------------------------------------------------
        // Per-run timing arrays — needed for mean + stddev reporting
        // -----------------------------------------------------------------------
        double *factor_times = (double *)malloc((size_t)bench_count * sizeof(double));
        double *root_times = (double *)malloc((size_t)bench_count * sizeof(double));
        int last_total_factors = 0;

        for (int bench = 0; bench < bench_count; bench++)
        {
            int *exponents = (int *)calloc(prime_count, sizeof(int));

            auto factor_start = std::chrono::high_resolution_clock::now();

            // -----------------------------------------------------------------------
            // Step 2: parallel trial division
            //
            // Each OpenMP thread handles a static chunk of the 6,542 primes.
            // For prime q: compute challenge mod q via Horner's method on the
            // raw byte array — no GMP allocation for the ~99.9% of primes that
            // don't divide. count_exponent() (GMP) only called on actual divisors.
            // -----------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < prime_count; i++)
            {
                unsigned long rem = 0;
                for (size_t b = 0; b < nbytes; b++)
                    rem = (rem * 256 + challenge_bytes[b]) % primes[i];

                if (rem != 0)
                {
                    exponents[i] = 0;
                    continue;
                }

                // prime divides — count exact multiplicity
                exponents[i] = count_exponent(challenge, primes[i]);
            }

            // -----------------------------------------------------------------------
            // Step 3: verify full factorization
            //
            // The previous implementation used a double for loop:
            //   for each prime i with exponents[i] > 0:
            //     for e in [0, exponents[i]):       ← inner loop
            //       remaining /= primes[i]          ← one division per occurrence
            //
            // This is replaced by computing p^k once with mpz_ui_pow_ui and then
            // dividing remaining by p^k in a single mpz_divexact call.
            // The inner loop is completely eliminated.
            // -----------------------------------------------------------------------
            mpz_t remaining;
            mpz_init_set(remaining, challenge);
            int total_factors = 0;

            for (int i = 0; i < prime_count; i++)
            {
                if (exponents[i] == 0)
                    continue;

                // divide out primes[i]^exponents[i] in one shot — no inner loop
                mpz_t pe;
                mpz_init(pe);
                mpz_ui_pow_ui(pe, primes[i], (unsigned long)exponents[i]);
                mpz_divexact(remaining, remaining, pe);
                mpz_clear(pe);

                total_factors += exponents[i];
                if (mpz_cmp_ui(remaining, 1) == 0)
                    break; // fully factored
            }

            factor_times[bench] = std::chrono::duration<double, std::milli>(
                                      std::chrono::high_resolution_clock::now() - factor_start)
                                      .count();
            last_total_factors = total_factors;

            if (bench == 0)
            {
                printf("[factor] Found %d factors\n", total_factors);
                if (mpz_cmp_ui(remaining, 1) != 0)
                    printf("[factor] WARNING: challenge not fully factored\n");
            }

            // -----------------------------------------------------------------------
            // Step 4: reconstruction
            // -----------------------------------------------------------------------
            if (mpz_cmp_ui(remaining, 1) == 0)
            {
                auto root_start = std::chrono::high_resolution_clock::now();

                if (method == METHOD_POWM_RECON)
                {
                    // ---------------------------------------------------------------
                    // Sequential powm-per-factor — no parallelism, no table
                    //
                    // For each found factor p with exponent k:
                    //   cbrt(p) = p^e mod M   (one mpz_powm call)
                    //
                    // This does NOT use cbrt_table — it recomputes everything from
                    // scratch per factor. Purpose: give the supervisor a direct
                    // measurement of raw powm cost, unaffected by parallelism or
                    // table lookups. This is the function that mpn_powm (if available)
                    // would replace — see k_common.cpp for why it is not used.
                    // ---------------------------------------------------------------
                    mpz_t phi, e, three;
                    mpz_inits(phi, e, three, NULL);
                    mpz_sub_ui(phi, modulo, 1);
                    mpz_set_ui(three, 3);
                    mpz_invert(e, three, phi);

                    mpz_set_ui(result, 1);
                    for (int i = 0; i < prime_count; i++)
                    {
                        if (exponents[i] == 0)
                            continue;

                        mpz_t p_mpz, cbrt_p;
                        mpz_inits(p_mpz, cbrt_p, NULL);
                        mpz_set_ui(p_mpz, primes[i]);

                        // single sequential mpz_powm — this is what we time
                        mpz_powm(cbrt_p, p_mpz, e, modulo);

                        if (exponents[i] > 1)
                        {
                            mpz_t tmp;
                            mpz_init(tmp);
                            mpz_powm_ui(tmp, cbrt_p,
                                        (unsigned long)exponents[i], modulo);
                            mpz_set(cbrt_p, tmp);
                            mpz_clear(tmp);
                        }
                        mpz_mul(result, result, cbrt_p);
                        mpz_mod(result, result, modulo);
                        mpz_clears(p_mpz, cbrt_p, NULL);
                    }
                    mpz_clears(phi, e, three, NULL);
                }
                else
                {
                    // parallel reconstruction using precomputed cbrt table
                    compute_cube_root_from_factors(result, exponents, cbrt_table,
                                                   prime_count, modulo);
                }

                root_times[bench] = std::chrono::duration<double, std::milli>(
                                        std::chrono::high_resolution_clock::now() - root_start)
                                        .count();
            }
            else
            {
                root_times[bench] = 0.0;
            }

            mpz_clear(remaining);
            free(exponents);
        }

        free(challenge_bytes);

        // -----------------------------------------------------------------------
        // Report: mean ± standard deviation
        // -----------------------------------------------------------------------
        if (bench_count == 1)
        {
            printf("Fully factored into %d factors in %.3f ms\n",
                   last_total_factors, factor_times[0]);
            printf("Cube root recovery time: %.3f ms\n", root_times[0]);
            printf("Total: %.3f ms\n", factor_times[0] + root_times[0]);
        }
        else
        {
            double f_sum = 0.0, r_sum = 0.0;
            double f_sumsq = 0.0, r_sumsq = 0.0;
            for (int i = 0; i < bench_count; i++)
            {
                f_sum += factor_times[i];
                r_sum += root_times[i];
                f_sumsq += factor_times[i] * factor_times[i];
                r_sumsq += root_times[i] * root_times[i];
            }

            printf("\n=== Results over %d run(s) ===\n", bench_count);
            printf("\nFACTOR method:\n");
            print_timing_stat_moments("Factoring:",
                                      f_sum, f_sumsq, bench_count,
                                      26, false);
            print_timing_stat_moments("Reconstruction:",
                                      r_sum, r_sumsq, bench_count,
                                      26, false);
            print_timing_total("Total:",
                               (f_sum + r_sum) / bench_count,
                               26, NULL);
        }

        free(factor_times);
        free(root_times);
    }

    // ---- Verify ----
    printf("\n");
    mpz_t challenge_mod;
    mpz_init(challenge_mod);
    mpz_mod(challenge_mod, challenge, modulo);
    verify_cube_root(result, challenge_mod, modulo);

    // ---- Cleanup ----
    mpz_clears(challenge, modulo, result, challenge_mod, NULL);
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