#include "k_common.h"

#include <gmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

void read_mpz_from_file(mpz_t num, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
    {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    // try decimal text first (e.g. challenge files written as plain integers)
    if (mpz_inp_str(num, f, 10) == 0)
    {
        // fall back to GMP binary format (mpz_out_raw / mpz_inp_raw)
        rewind(f);
        if (mpz_inp_raw(num, f) == 0)
        {
            fprintf(stderr, "Error: Failed to read number from %s\n", filename);
            fclose(f);
            exit(1);
        }
    }
    fclose(f);
    printf("Successfully read number from %s\n", filename);
}

// ---------------------------------------------------------------------------
// Prime generation — Sieve of Eratosthenes
// ---------------------------------------------------------------------------

unsigned long *generate_primes_up_to(unsigned long limit, int *count)
{
    char *sieve = (char *)malloc(limit + 1);
    if (!sieve)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    memset(sieve, 1, limit + 1);
    sieve[0] = sieve[1] = 0;

    // mark composites: for each prime p, mark p², p²+p, p²+2p, ...
    for (unsigned long i = 2; i * i <= limit; i++)
        if (sieve[i])
            for (unsigned long j = i * i; j <= limit; j += i)
                sieve[j] = 0;

    *count = 0;
    for (unsigned long i = 2; i <= limit; i++)
        if (sieve[i])
            (*count)++;

    unsigned long *primes = (unsigned long *)malloc(
        (size_t)(*count) * sizeof(unsigned long));
    if (!primes)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(sieve);
        exit(1);
    }

    int idx = 0;
    for (unsigned long i = 2; i <= limit; i++)
        if (sieve[i])
            primes[idx++] = i;

    free(sieve);
    return primes;
}

// ---------------------------------------------------------------------------
// Cube root via Fermat's little theorem — naive baseline
// ---------------------------------------------------------------------------

bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo,
                                 unsigned long exponent)
{
    mpz_t phi, e_inv, e_base;
    mpz_inits(phi, e_inv, e_base, NULL);

    // phi = modulo - 1  (Euler's totient for a prime)
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);

    // compute e_inv = exponent^{-1} mod phi  (exists iff gcd(exponent, phi) = 1)
    if (mpz_invert(e_inv, e_base, phi) == 0)
    {
        // e-th root does not exist for this modulo
        mpz_set_ui(result, 0);
        mpz_clears(phi, e_inv, e_base, NULL);
        return false;
    }

    // result = value^e_inv mod modulo — this is the e-th root
    mpz_powm(result, value, e_inv, modulo);
    mpz_clears(phi, e_inv, e_base, NULL);
    return true;
}

void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod,
                      const mpz_t modulo, unsigned long exponent)
{
    mpz_t powered;
    mpz_init(powered);
    // check: does candidate^exponent mod M equal the original challenge mod M?
    mpz_powm_ui(powered, candidate_cuberoot, exponent, modulo);
    if (mpz_cmp(powered, challenge_mod) == 0)
        printf("SUCCESS: %lu-th root is correct!\n", exponent);
    else
        printf("FAILURE: %lu-th root is incorrect\n", exponent);
    mpz_clear(powered);
}

// ---------------------------------------------------------------------------
// Factoring helper
// ---------------------------------------------------------------------------

int count_exponent(const mpz_t n, unsigned long prime)
{
    // fast early exit: no GMP allocation if prime doesn't divide n
    if (mpz_tdiv_ui(n, prime) != 0)
        return 0;

    mpz_t temp;
    mpz_init_set(temp, n); // work on a private copy

    int exponent = 0;
    while (mpz_divisible_ui_p(temp, prime))
    {
        mpz_divexact_ui(temp, temp, prime);
        exponent++;
    }
    mpz_clear(temp);
    return exponent;
}

// ---------------------------------------------------------------------------
// Benchmark/reporting helpers
// ---------------------------------------------------------------------------

double compute_stddev_from_values(const double *vals, int n, bool sample_stddev)
{
    if (!vals || n <= 0)
        return 0.0;
    if (n == 1)
        return 0.0;

    double sum = 0.0;
    double sumsq = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += vals[i];
        sumsq += vals[i] * vals[i];
    }

    return compute_stddev_from_moments(sum, sumsq, n, sample_stddev);
}

double compute_stddev_from_moments(double sum, double sumsq, int n,
                                   bool sample_stddev)
{
    if (n <= 0)
        return 0.0;
    if (n == 1)
        return 0.0;

    double denom = sample_stddev ? (double)(n - 1) : (double)n;
    double var = (sumsq - (sum * sum) / (double)n) / denom;
    if (var < 0.0)
    {
        // Clamp tiny negative values caused by floating-point roundoff.
        var = 0.0;
    }
    return sqrt(var);
}

void print_timing_stat_moments(const char *label,
                               double sum, double sumsq, int n,
                               int label_width, bool sample_stddev)
{
    double mean = (n > 0) ? (sum / n) : 0.0;
    double stddev = compute_stddev_from_moments(sum, sumsq, n, sample_stddev);
    printf("    %-*s avg=%9.3f ms   stddev=%7.3f ms\n",
           label_width, label, mean, stddev);
}

void print_timing_total(const char *label, double mean,
                        int label_width, const char *extra_text)
{
    if (extra_text && extra_text[0] != '\0')
    {
        printf("    %-*s avg=%9.3f ms  %s\n",
               label_width, label, mean, extra_text);
    }
    else
    {
        printf("    %-*s avg=%9.3f ms\n", label_width, label, mean);
    }
}

// Computes y = (c * r^exponent) mod M
void blind_challenge(mpz_t y, const mpz_t challenge, const mpz_t r, const mpz_t modulo,
                     unsigned long exponent)
{
    mpz_t r_blind;
    mpz_init(r_blind);
    mpz_powm_ui(r_blind, r, exponent, modulo); // r^e mod M
    mpz_mul(y, challenge, r_blind);
    mpz_mod(y, y, modulo);
    mpz_clear(r_blind);
}

// Computes result = (y_root * r^-1) mod M
void unblind_root(mpz_t result, const mpz_t y_root, const mpz_t r, const mpz_t modulo)
{
    mpz_t r_inv;
    mpz_init(r_inv);
    if (mpz_invert(r_inv, r, modulo))
    { // r^-1 mod M
        mpz_mul(result, y_root, r_inv);
        mpz_mod(result, result, modulo);
    }
    mpz_clear(r_inv);
}

// ---------------------------------------------------------------------------
// Sequential powm baseline
// ---------------------------------------------------------------------------

double run_powm_baseline(mpz_t result, const mpz_t challenge,
                         const mpz_t modulo, int bench_count,
                         int label_width, unsigned long exponent)
{
    mpz_t e_inv, phi, e_base;
    mpz_inits(e_inv, phi, e_base, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);
    mpz_invert(e_inv, e_base, phi);
    mpz_clears(phi, e_base, NULL);

    double sum = 0.0, sumsq = 0.0;

    for (int i = 0; i < bench_count; i++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mpz_powm(result, challenge, e_inv, modulo);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum += dt;
        sumsq += dt * dt;
    }

    mpz_clear(e_inv);

    print_timing_stat_moments("powm (sequential baseline):",
                              sum, sumsq, bench_count, label_width, true);
    return (bench_count > 0) ? sum / bench_count : 0.0;
}

// ---------------------------------------------------------------------------
// Unified factor list
// ---------------------------------------------------------------------------

void kfl_init(kfactor_list_t *f)
{
    f->items = NULL;
    f->count = 0;
    f->capacity = 0;
}

static void kfl_grow(kfactor_list_t *f)
{
    if (f->count == f->capacity)
    {
        int nc = (f->capacity == 0) ? 32 : f->capacity * 2;
        f->items = (kfactor_t *)realloc(f->items, (size_t)nc * sizeof(kfactor_t));
        f->capacity = nc;
    }
}

void kfl_append_ul(kfactor_list_t *f, unsigned long prime, int exponent)
{
    kfl_grow(f);
    mpz_init_set_ui(f->items[f->count].prime, prime);
    f->items[f->count].exponent = exponent;
    f->count++;
}

void kfl_append_mpz(kfactor_list_t *f, const mpz_t prime, int exponent)
{
    kfl_grow(f);
    mpz_init_set(f->items[f->count].prime, prime);
    f->items[f->count].exponent = exponent;
    f->count++;
}

void kfl_free(kfactor_list_t *f)
{
    for (int i = 0; i < f->count; i++)
        mpz_clear(f->items[i].prime);
    free(f->items);
    f->items = NULL;
    f->count = 0;
    f->capacity = 0;
}

void write_benchmark_json(const char *filepath, const benchmark_results_t &res)
{
    FILE *f = fopen(filepath, "w");
    if (!f)
    {
        perror("[ERROR] fopen failed for JSON");
        return;
    }

    // ---------------------------------------------------------------------------
    // JSON layout
    //
    // Run 0 is always a cold-start run: instruction cache, data cache, and
    // branch predictor are all cold. It is kept in the output under "cold_run"
    // so it is visible and clearly labelled rather than silently discarded.
    //
    // Runs 1..N are warm steady-state runs. Their values go into "warm_runs".
    // All statistical analysis (avg, stddev, speedup) should use warm_runs only.
    //
    // "runs_total"  = bench_count (includes run 0)
    // "runs_warm"   = bench_count - 1 (excludes run 0)
    // ---------------------------------------------------------------------------

    // Helpers — write a slice [from, to) of a vector
    auto write_slice_dbl = [&](const char *key,
                               const std::vector<double> &vec,
                               size_t from, size_t to,
                               bool is_last = false)
    {
        if (vec.empty() || from >= vec.size())
            return;
        to = (to > vec.size()) ? vec.size() : to;
        fprintf(f, "      \"%s\": [", key);
        for (size_t i = from; i < to; ++i)
            fprintf(f, "%.4f%s", vec[i], (i == to - 1) ? "" : ", ");
        fprintf(f, "]%s\n", is_last ? "" : ",");
    };

    auto write_slice_int = [&](const char *key,
                               const std::vector<int> &vec,
                               size_t from, size_t to,
                               bool is_last = false)
    {
        if (vec.empty() || from >= vec.size())
            return;
        to = (to > vec.size()) ? vec.size() : to;
        fprintf(f, "      \"%s\": [", key);
        for (size_t i = from; i < to; ++i)
            fprintf(f, "%d%s", vec[i], (i == to - 1) ? "" : ", ");
        fprintf(f, "]%s\n", is_last ? "" : ",");
    };

    // Scalar from index helper (returns -1 if out of range)
    auto dbl_at = [&](const std::vector<double> &v, size_t i) -> double
    {
        return (i < v.size()) ? v[i] : -1.0;
    };

    int n_total = res.bench_count;
    int n_warm = (n_total > 1) ? n_total - 1 : 0;

    // Compute warm averages.
    //
    // Two distinct failure modes:
    //   (a) bench_succeeded = false (blinding exhausted all 256 attempts):
    //       Nothing is pushed to any vector for that run.  Use factor_times.size()
    //       as the authoritative count of recorded runs — do NOT loop to n_warm.
    //   (b) bench_succeeded = true but blinding_attempts[i] > 1:
    //       Run completed but needed multiple blinding rounds; timing is skewed.
    //       Recorded in warm_runs but excluded from warm_avg.
    //
    // runs_failed = (a) runs  |  runs_valid = recorded runs that pass filter (b)
    double warm_factor_avg = 0.0, warm_recon_avg = 0.0, warm_total_avg = 0.0;
    int n_recorded = (int)res.factor_times.size(); // runs where bench_succeeded = true
    int n_failed   = n_warm - n_recorded;          // runs where bench exhausted blindings
    int n_valid    = 0;
    for (int i = 0; i < n_recorded; i++)
    {
        bool bad_blinding = (i < (int)res.blinding_attempts.size() &&
                             res.blinding_attempts[i] > 1);
        if (!bad_blinding)
        {
            warm_factor_avg += res.factor_times[i];
            if (i < (int)res.recon_times.size())
                warm_recon_avg += res.recon_times[i];
            if (i < (int)res.total_times.size())
                warm_total_avg += res.total_times[i];
            n_valid++;
        }
    }
    if (n_valid > 0)
    {
        warm_factor_avg /= n_valid;
        warm_recon_avg  /= n_valid;
        warm_total_avg  /= n_valid;
    }

    fprintf(f, "{\n");

    // ---- metadata ----
    fprintf(f, "  \"metadata\": {\n");
    fprintf(f, "    \"k_size\": %d,\n", res.k_size);
    fprintf(f, "    \"challenge_bits\": %zu,\n", res.challenge_bits);
    fprintf(f, "    \"method\": \"%s\",\n", res.method_name);
    fprintf(f, "    \"threads_per_rank\": %d,\n", res.num_threads);
    fprintf(f, "    \"mpi_ranks\": %d,\n", res.mpi_ranks);
    fprintf(f, "    \"runs_total\": %d,\n", n_total);
    fprintf(f, "    \"runs_warm\": %d,\n", n_warm);
    fprintf(f, "    \"runs_valid\": %d,\n", n_valid);
    fprintf(f, "    \"runs_failed\": %d,\n", n_failed);
    fprintf(f, "    \"note\": \"run 0=cold-start; runs_failed=blinding exhausted; runs_valid=recorded with attempts==1\",\n");
    fprintf(f, "    \"powm_baseline_ms\": %.4f\n", res.powm_baseline_ms);
    fprintf(f, "  },\n");

    // ---- warm averages (the numbers to use for analysis) ----
    fprintf(f, "  \"warm_avg\": {\n");
    fprintf(f, "    \"factor_ms\": %.4f,\n", warm_factor_avg);
    fprintf(f, "    \"recon_ms\": %.4f,\n", warm_recon_avg);
    fprintf(f, "    \"total_ms\": %.4f,\n", warm_total_avg);
    if (res.powm_baseline_ms > 0 && warm_recon_avg > 0)
        fprintf(f, "    \"speedup_recon\": %.2f,\n",
                res.powm_baseline_ms / warm_recon_avg);
    if (res.powm_baseline_ms > 0 && warm_total_avg > 0)
        fprintf(f, "    \"speedup_total\": %.2f\n",
                res.powm_baseline_ms / warm_total_avg);
    else
        fprintf(f, "    \"speedup_total\": 0.0\n");
    fprintf(f, "  },\n");

    // ---- cold run (run 0) — kept for transparency ----
    fprintf(f, "  \"cold_run\": {\n");
    fprintf(f, "    \"note\": \"cold cache — do not use for performance comparison\",\n");
    fprintf(f, "    \"factor_ms\": %.4f,\n", dbl_at(res.factor_times, 0));
    fprintf(f, "    \"recon_ms\": %.4f,\n", dbl_at(res.recon_times, 0));
    fprintf(f, "    \"total_ms\": %.4f\n", dbl_at(res.total_times, 0));
    fprintf(f, "  },\n");

    // ---- warm runs (runs 1..N) ----
    fprintf(f, "  \"warm_runs\": {\n");
    write_slice_dbl("factor_ms", res.factor_times, 1, (size_t)n_total);
    write_slice_dbl("recon_ms", res.recon_times, 1, (size_t)n_total);
    write_slice_dbl("local_cbrt_ms", res.local_cbrt_times, 1, (size_t)n_total);
    write_slice_dbl("cbrt_hit_ms", res.cbrt_hit_ms, 1, (size_t)n_total);
    write_slice_dbl("cbrt_miss_ms", res.cbrt_miss_ms, 1, (size_t)n_total);
    write_slice_int("cbrt_hits", res.cbrt_hits, 1, (size_t)n_total);
    write_slice_int("cbrt_misses", res.cbrt_misses, 1, (size_t)n_total);
    write_slice_dbl("gather_ms", res.gather_times, 1, (size_t)n_total);
    write_slice_dbl("combine_ms", res.combine_times, 1, (size_t)n_total);
    write_slice_int("blinding_attempts", res.blinding_attempts, 1, (size_t)n_total);
    // k64-specific fields (empty for k16/k32 — write_slice guards on empty vectors)
    if (!res.pollard_ms.empty())
        write_slice_dbl("pollard_ms", res.pollard_ms, 1, (size_t)n_total);
    if (!res.factors_small.empty())
        write_slice_int("factors_small", res.factors_small, 1, (size_t)n_total);
    if (!res.factors_large.empty())
        write_slice_int("factors_large", res.factors_large, 1, (size_t)n_total);
    write_slice_dbl("total_ms", res.total_times, 1, (size_t)n_total, true);
    fprintf(f, "  },\n");

    // Warmup runs
    if (!res.warmup_total_ms.empty())
    {
        fprintf(f, "  \"warmup_runs\": {\n");
        fprintf(f, "    \"note\": \"2 silent runs before bench loop to prime CPU caches\",\n");
        fprintf(f, "    \"total_ms\": [");
        for (size_t i = 0; i < res.warmup_total_ms.size(); i++)
            fprintf(f, "%.3f%s", res.warmup_total_ms[i],
                    i == res.warmup_total_ms.size() - 1 ? "" : ", ");
        fprintf(f, "]\n  },\n");
    }

    // Prime statistics
    if (!res.primes_tested_per_rank.empty() || !res.primes_found_per_rank.empty())
    {
        fprintf(f, "  \"prime_stats\": {\n");
        if (!res.primes_tested_per_rank.empty())
        {
            fprintf(f, "    \"tested_per_rank\": [");
            for (size_t i = 0; i < res.primes_tested_per_rank.size(); i++)
                fprintf(f, "%" PRId64 "%s", res.primes_tested_per_rank[i],
                        i == res.primes_tested_per_rank.size() - 1 ? "" : ", ");
            fprintf(f, "],\n");
        }
        if (!res.primes_found_per_rank.empty())
        {
            fprintf(f, "    \"found_per_rank\": [");
            for (size_t i = 0; i < res.primes_found_per_rank.size(); i++)
                fprintf(f, "%d%s", res.primes_found_per_rank[i],
                        i == res.primes_found_per_rank.size() - 1 ? "" : ", ");
            fprintf(f, "]\n");
        }
        fprintf(f, "  },\n");
    }

    // alg3/alg4 hit classification (absent for k16/k32/k35)
    if (!res.hit_type.empty())
    {
        int n_smooth = 0, n_almost = 0, n_multi = 0;
        for (int v : res.hit_type)
        {
            if (v == 1) n_smooth++;
            else if (v == 2) n_almost++;
            else n_multi++;
        }
        fprintf(f, "  \"hit_stats\": {\n");
        fprintf(f, "    \"smooth_count\": %d,\n", n_smooth);
        fprintf(f, "    \"almost_smooth_count\": %d,\n", n_almost);
        fprintf(f, "    \"multi_medium_count\": %d,\n", n_multi);
        fprintf(f, "    \"hit_type\": [");
        for (size_t i = 0; i < res.hit_type.size(); i++)
            fprintf(f, "%d%s", res.hit_type[i],
                    i == res.hit_type.size() - 1 ? "" : ", ");
        fprintf(f, "],\n");
        fprintf(f, "    \"cofactor_bits\": [");
        for (size_t i = 0; i < res.cofactor_bits.size(); i++)
            fprintf(f, "%d%s", res.cofactor_bits[i],
                    i == res.cofactor_bits.size() - 1 ? "" : ", ");
        fprintf(f, "]");
        if (!res.medium_factor_count.empty())
        {
            fprintf(f, ",\n    \"medium_factor_count\": [");
            for (size_t i = 0; i < res.medium_factor_count.size(); i++)
                fprintf(f, "%d%s", res.medium_factor_count[i],
                        i == res.medium_factor_count.size() - 1 ? "" : ", ");
            fprintf(f, "]");
        }
        fprintf(f, "\n  },\n");
    }

    // Per-thread load balance
    if (!res.thread_primes_tested.empty())
    {
        fprintf(f, "  \"thread_balance\": {\n");
        fprintf(f, "    \"note\": \"rank 0 OMP threads, first warm run\",\n");
        fprintf(f, "    \"tested_per_thread\": [");
        for (size_t i = 0; i < res.thread_primes_tested.size(); i++)
            fprintf(f, "%" PRId64 "%s", res.thread_primes_tested[i],
                    i == res.thread_primes_tested.size() - 1 ? "" : ", ");
        fprintf(f, "],\n");
        fprintf(f, "    \"found_per_thread\": [");
        for (size_t i = 0; i < res.thread_primes_found.size(); i++)
            fprintf(f, "%d%s", res.thread_primes_found[i],
                    i == res.thread_primes_found.size() - 1 ? "" : ", ");
        fprintf(f, "]\n  }\n");
    }
    else
        fprintf(f, "  \"thread_balance\": {}\n");

    fprintf(f, "}\n");
    fclose(f);
    printf("[INFO] JSON benchmark written to %s\n", filepath);
}

// ---------------------------------------------------------------------------
// Common argument parser
// ---------------------------------------------------------------------------

int parse_common_args(int argc, char **argv, solver_config_t *cfg,
                      int *argi_inout)
{
    int &argi = *argi_inout;
    int positional = 0;

    // Count how many positional args have already been consumed by the caller
    // so we know which slot (challenge vs modulo) we are filling.
    // We infer this from whether the defaults are still set, but to keep it
    // simple we just track within this single call — callers pre-set defaults.

    while (argi < argc)
    {
        if (strcmp(argv[argi], "--bench") == 0)
        {
            if (argi + 1 >= argc)
            {
                fprintf(stderr, "Error: Missing value for --bench\n");
                return 1;
            }
            cfg->bench_count = atoi(argv[++argi]);
            argi++;
        }
        else if (argv[argi][0] != '-')
        {
            // positional: first non-flag = challenge, second = modulo
            if (positional == 0)
            {
                cfg->challenge_file = argv[argi];
                positional++;
                argi++;
            }
            else if (positional == 1)
            {
                cfg->modulo_file = argv[argi];
                positional++;
                argi++;
            }
            else
            {
                // More positional args — caller handles them
                break;
            }
        }
        else if (strcmp(argv[argi], "--json") == 0 && argi + 1 < argc)
        {
            cfg->json_output_file = argv[++argi];
            argi++;
        }
        else if (strcmp(argv[argi], "--force-factor") == 0)
        {
            cfg->force_factor = true;
            argi++;
        }
        else
        {
            // Unknown flag — stop and let caller handle it
            break;
        }
    }
    return 0;
}