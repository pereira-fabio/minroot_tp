#include "common.h"

#include <gmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

// SAFETY: Do not call from inside an MPI job after MPI_Init.  exit(1) on one
// rank will hang all other ranks waiting on MPI collectives.  MPI solvers use
// ksc_broadcast_challenge_modulo() instead, which handles file errors via
// MPI_Abort so the entire job terminates cleanly.
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
// e-th root via Fermat's little theorem — naive baseline
// ---------------------------------------------------------------------------

bool compute_eth_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo,
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

void verify_eth_root(const mpz_t candidate_root, const mpz_t challenge_mod,
                     const mpz_t modulo, unsigned long exponent)
{
    mpz_t powered;
    mpz_init(powered);
    // check: does candidate^exponent mod M equal the original challenge mod M?
    mpz_powm_ui(powered, candidate_root, exponent, modulo);
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

// ---------------------------------------------------------------------------
// Sequential powm baseline
// ---------------------------------------------------------------------------

double run_powm_baseline(mpz_t result, const mpz_t challenge,
                         const mpz_t modulo, int reps,
                         int label_width, unsigned long exponent)
{
    mpz_t e_inv, phi, e_base;
    mpz_inits(e_inv, phi, e_base, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);
    mpz_invert(e_inv, e_base, phi);
    mpz_clears(phi, e_base, NULL);

    double sum = 0.0, sumsq = 0.0;

    for (int i = 0; i < reps; i++)
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
                              sum, sumsq, reps, label_width, true);
    return (reps > 0) ? sum / reps : 0.0;
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

// ---------------------------------------------------------------------------
// JSON helper — write a slice [from, to) of a vector as a JSON array field.
// Skipped silently when the vector is empty or `from` is out of range.
// ---------------------------------------------------------------------------

static void json_write_dbl_array(FILE *f, const char *key,
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
}

static void json_write_int_array(FILE *f, const char *key,
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
}

// Mean of a value vector (0 if empty).
static double vmean(const std::vector<double> &v)
{
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / (double)v.size();
}

// Write mean/std/min/max/median/p25/p75 for a value vector as a nested object.
// Powers the warm_stats block (error bars / box plots across challenges).
static void json_write_stats_obj(FILE *f, const char *key,
                                 std::vector<double> v, bool is_last)
{
    if (v.empty())
    {
        fprintf(f, "    \"%s\": {}%s\n", key, is_last ? "" : ",");
        return;
    }
    int n = (int)v.size();
    double sum = 0.0, sumsq = 0.0;
    for (double x : v) { sum += x; sumsq += x * x; }
    double mean = sum / n;
    double var = (n > 1) ? (sumsq - sum * sum / n) / (n - 1) : 0.0;
    if (var < 0.0) var = 0.0;
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) -> double {
        if (n == 1) return v[0];
        double idx = p * (n - 1);
        size_t lo = (size_t)idx;
        double frac = idx - (double)lo;
        return (lo + 1 < (size_t)n) ? v[lo] + frac * (v[lo + 1] - v[lo]) : v[lo];
    };
    fprintf(f, "    \"%s\": {\"mean\": %.4f, \"std\": %.4f, \"min\": %.4f, "
               "\"max\": %.4f, \"median\": %.4f, \"p25\": %.4f, \"p75\": %.4f}%s\n",
            key, mean, sqrt(var), v.front(), v.back(),
            pct(0.5), pct(0.25), pct(0.75), is_last ? "" : ",");
}

// Per-challenge speedup vector = powm_baseline / total_ms[i].
static std::vector<double> per_challenge_speedup(const benchmark_results_t &res)
{
    std::vector<double> spd;
    double base = res.timing.powm_baseline_ms;
    for (double tm : res.timing.total_times)
        spd.push_back((base > 0 && tm > 0) ? base / tm : 0.0);
    return spd;
}

// ---------------------------------------------------------------------------
// JSON section writers — each produces one complete "{...}," block.
// All sections except the last end with a trailing comma.
// ---------------------------------------------------------------------------

void set_cold_run(benchmark_results_t &dst, const benchmark_results_t &c)
{
    const auto &t = c.timing;
    if (t.total_times.empty())
        return;  // cold run recorded nothing (e.g. non-smooth) — leave present=false
    auto first = [](const std::vector<double> &v) { return v.empty() ? 0.0 : v[0]; };
    dst.cold.present       = true;
    dst.cold.factor_ms     = first(t.factor_times);
    dst.cold.recon_ms      = first(t.recon_times);
    dst.cold.local_cbrt_ms = first(t.local_cbrt_times);
    dst.cold.gather_ms     = first(t.gather_times);
    dst.cold.combine_ms    = first(t.combine_times);
    dst.cold.total_ms      = first(t.total_times);
}

static void json_write_cold_run(FILE *f, const benchmark_results_t &res)
{
    const auto &cr = res.cold;
    if (!cr.present)
    {
        fprintf(f, "  \"cold_run\": {},\n");
        return;
    }
    fprintf(f, "  \"cold_run\": {\n");
    fprintf(f, "    \"note\": \"challenge_0 cold-cache warmup — saved for completeness, "
               "NOT used in warm averages\",\n");
    fprintf(f, "    \"factor_ms\": %.4f,\n",     cr.factor_ms);
    fprintf(f, "    \"recon_ms\": %.4f,\n",      cr.recon_ms);
    fprintf(f, "    \"local_cbrt_ms\": %.4f,\n", cr.local_cbrt_ms);
    fprintf(f, "    \"gather_ms\": %.4f,\n",     cr.gather_ms);
    fprintf(f, "    \"combine_ms\": %.4f,\n",    cr.combine_ms);
    fprintf(f, "    \"total_ms\": %.4f\n",       cr.total_ms);
    fprintf(f, "  },\n");
}

static void json_write_metadata(FILE *f, const benchmark_results_t &res,
                                 int n_warm, int n_valid, int n_failed)
{
    const auto &c = res.config;
    int n_total = n_warm + (res.cold.present ? 1 : 0);
    fprintf(f, "  \"metadata\": {\n");
    fprintf(f, "    \"algorithm\": \"%s\",\n", c.algorithm ? c.algorithm : "");
    fprintf(f, "    \"method\": \"%s\",\n", c.method_name ? c.method_name : "");
    fprintf(f, "    \"exponent\": %lu,\n", c.exponent);
    fprintf(f, "    \"smoothness_bound_bits\": %d,\n", c.smoothness_bits);
    fprintf(f, "    \"challenge_bits\": %zu,\n", c.challenge_bits);
    fprintf(f, "    \"mpi_ranks\": %d,\n", c.mpi_ranks);
    fprintf(f, "    \"nodes\": %d,\n", c.mpi_ranks / 2);
    fprintf(f, "    \"threads_per_rank\": %d,\n", c.num_threads);
    if (c.b0 > 0)
        fprintf(f, "    \"prefilter_b0\": %lu,\n", c.b0);
    if (c.t_threshold_bits > 0)
        fprintf(f, "    \"prefilter_t_bits\": %d,\n", c.t_threshold_bits);
    fprintf(f, "    \"runs_total\": %d,\n", n_total);
    fprintf(f, "    \"runs_warm\": %d,\n", n_warm);
    fprintf(f, "    \"runs_valid\": %d,\n", n_valid);
    fprintf(f, "    \"runs_failed\": %d,\n", n_failed);
    fprintf(f, "    \"note\": \"warm challenges only (cold challenge_0 discarded upstream); "
               "runs_failed = blinding exhausted; one timed run per distinct challenge\",\n");
    fprintf(f, "    \"powm_baseline_ms\": %.4f\n", res.timing.powm_baseline_ms);
    fprintf(f, "  },\n");
}

static void json_write_warm_avg(FILE *f, const benchmark_results_t &res)
{
    const auto &t = res.timing;
    double factor_avg = vmean(t.factor_times);
    double recon_avg  = vmean(t.recon_times);
    double total_avg  = vmean(t.total_times);
    double base = t.powm_baseline_ms;
    fprintf(f, "  \"warm_avg\": {\n");
    fprintf(f, "    \"factor_ms\": %.4f,\n", factor_avg);
    fprintf(f, "    \"recon_ms\": %.4f,\n", recon_avg);
    fprintf(f, "    \"local_cbrt_ms\": %.4f,\n", vmean(t.local_cbrt_times));
    fprintf(f, "    \"gather_ms\": %.4f,\n", vmean(t.gather_times));
    fprintf(f, "    \"combine_ms\": %.4f,\n", vmean(t.combine_times));
    if (!t.prefilter_times.empty())
        fprintf(f, "    \"prefilter_ms\": %.4f,\n", vmean(t.prefilter_times));
    fprintf(f, "    \"total_ms\": %.4f,\n", total_avg);
    fprintf(f, "    \"speedup_recon\": %.4f,\n",
            (base > 0 && recon_avg > 0) ? base / recon_avg : 0.0);
    fprintf(f, "    \"speedup_total\": %.4f\n",
            (base > 0 && total_avg > 0) ? base / total_avg : 0.0);
    fprintf(f, "  },\n");
}

static void json_write_warm_stats(FILE *f, const benchmark_results_t &res)
{
    std::vector<double> spd = per_challenge_speedup(res);
    fprintf(f, "  \"warm_stats\": {\n");
    json_write_stats_obj(f, "factor_ms", res.timing.factor_times, false);
    json_write_stats_obj(f, "recon_ms",  res.timing.recon_times,  false);
    json_write_stats_obj(f, "total_ms",  res.timing.total_times,  false);
    json_write_stats_obj(f, "speedup_total", spd, true);
    fprintf(f, "  },\n");
}

static void json_write_warm_runs(FILE *f, const benchmark_results_t &res)
{
    const auto &t = res.timing;
    size_t n = t.factor_times.size();
    std::vector<double> spd = per_challenge_speedup(res);
    fprintf(f, "  \"warm_runs\": {\n");
    json_write_dbl_array(f, "factor_ms",      t.factor_times,     0, n);
    json_write_dbl_array(f, "recon_ms",       t.recon_times,      0, n);
    json_write_dbl_array(f, "local_cbrt_ms",  t.local_cbrt_times, 0, n);
    json_write_dbl_array(f, "gather_ms",      t.gather_times,     0, n);
    json_write_dbl_array(f, "combine_ms",     t.combine_times,    0, n);
    if (!t.prefilter_times.empty())
        json_write_dbl_array(f, "prefilter_ms", t.prefilter_times, 0, n);
    json_write_dbl_array(f, "cbrt_hit_ms",    t.cbrt_hit_ms,      0, n);
    json_write_dbl_array(f, "cbrt_miss_ms",   t.cbrt_miss_ms,     0, n);
    json_write_int_array(f, "cbrt_hits",      t.cbrt_hits,        0, n);
    json_write_int_array(f, "cbrt_misses",    t.cbrt_misses,      0, n);
    json_write_int_array(f, "blinding_attempts", t.blinding_attempts, 0, n);
    json_write_dbl_array(f, "total_ms",       t.total_times,      0, n);
    json_write_dbl_array(f, "speedup_total",  spd,                0, n, /*is_last=*/true);
    fprintf(f, "  },\n");
}

static void json_write_warmup_runs(FILE *f, const benchmark_results_t &res)
{
    if (res.timing.warmup_total_ms.empty())
        return;
    fprintf(f, "  \"warmup_runs\": {\n");
    fprintf(f, "    \"note\": \"2 silent runs before bench loop to prime CPU caches\",\n");
    fprintf(f, "    \"total_ms\": [");
    for (size_t i = 0; i < res.timing.warmup_total_ms.size(); i++)
        fprintf(f, "%.3f%s", res.timing.warmup_total_ms[i],
                i == res.timing.warmup_total_ms.size() - 1 ? "" : ", ");
    fprintf(f, "]\n  },\n");
}

static void json_write_prime_stats(FILE *f, const benchmark_results_t &res)
{
    if (res.stats.primes_tested_per_rank.empty() && res.stats.primes_found_per_rank.empty())
        return;
    fprintf(f, "  \"prime_stats\": {\n");
    if (!res.stats.primes_tested_per_rank.empty())
    {
        fprintf(f, "    \"tested_per_rank\": [");
        for (size_t i = 0; i < res.stats.primes_tested_per_rank.size(); i++)
            fprintf(f, "%" PRId64 "%s", res.stats.primes_tested_per_rank[i],
                    i == res.stats.primes_tested_per_rank.size() - 1 ? "" : ", ");
        fprintf(f, "],\n");
    }
    if (!res.stats.primes_found_per_rank.empty())
    {
        fprintf(f, "    \"found_per_rank\": [");
        for (size_t i = 0; i < res.stats.primes_found_per_rank.size(); i++)
            fprintf(f, "%d%s", res.stats.primes_found_per_rank[i],
                    i == res.stats.primes_found_per_rank.size() - 1 ? "" : ", ");
        fprintf(f, "]\n");
    }
    fprintf(f, "  },\n");
}

static void json_write_hit_stats(FILE *f, const benchmark_results_t &res)
{
    if (res.hits.hit_type.empty())
        return;
    int n_smooth = 0, n_almost = 0, n_multi = 0;
    for (int v : res.hits.hit_type)
    {
        if (v == 1)      n_smooth++;
        else if (v == 2) n_almost++;
        else             n_multi++;
    }
    int n_hits = (int)res.hits.hit_type.size();
    double inv = n_hits > 0 ? 1.0 / n_hits : 0.0;
    int cof_min = 0, cof_max = 0;
    double cof_mean = 0.0;
    if (!res.hits.cofactor_bits.empty())
    {
        cof_min = cof_max = res.hits.cofactor_bits[0];
        long cof_sum = 0;
        for (int b : res.hits.cofactor_bits)
        {
            if (b < cof_min) cof_min = b;
            if (b > cof_max) cof_max = b;
            cof_sum += b;
        }
        cof_mean = (double)cof_sum / res.hits.cofactor_bits.size();
    }
    fprintf(f, "  \"hit_stats\": {\n");
    fprintf(f, "    \"smooth_count\": %d,\n",        n_smooth);
    fprintf(f, "    \"almost_smooth_count\": %d,\n", n_almost);
    fprintf(f, "    \"multi_medium_count\": %d,\n",  n_multi);
    fprintf(f, "    \"smooth_frac\": %.4f,\n",        n_smooth * inv);
    fprintf(f, "    \"almost_smooth_frac\": %.4f,\n", n_almost * inv);
    fprintf(f, "    \"multi_medium_frac\": %.4f,\n",  n_multi * inv);
    fprintf(f, "    \"cofactor_bits_min\": %d,\n",  cof_min);
    fprintf(f, "    \"cofactor_bits_mean\": %.2f,\n", cof_mean);
    fprintf(f, "    \"cofactor_bits_max\": %d,\n",  cof_max);
    fprintf(f, "    \"hit_type\": [");
    for (size_t i = 0; i < res.hits.hit_type.size(); i++)
        fprintf(f, "%d%s", res.hits.hit_type[i],
                i == res.hits.hit_type.size() - 1 ? "" : ", ");
    fprintf(f, "],\n");
    fprintf(f, "    \"cofactor_bits\": [");
    for (size_t i = 0; i < res.hits.cofactor_bits.size(); i++)
        fprintf(f, "%d%s", res.hits.cofactor_bits[i],
                i == res.hits.cofactor_bits.size() - 1 ? "" : ", ");
    fprintf(f, "]");
    if (!res.hits.medium_factor_count.empty())
    {
        fprintf(f, ",\n    \"medium_factor_count\": [");
        for (size_t i = 0; i < res.hits.medium_factor_count.size(); i++)
            fprintf(f, "%d%s", res.hits.medium_factor_count[i],
                    i == res.hits.medium_factor_count.size() - 1 ? "" : ", ");
        fprintf(f, "]");
    }
    fprintf(f, "\n  },\n");
}

// thread_balance is always the last standard section.
// If extra_sections is provided it needs a trailing comma; otherwise none.
static void json_write_thread_balance(FILE *f, const benchmark_results_t &res,
                                       bool trailing_comma)
{
    const char *tail = trailing_comma ? ",\n" : "\n";
    if (!res.stats.thread_primes_tested.empty())
    {
        fprintf(f, "  \"thread_balance\": {\n");
        fprintf(f, "    \"note\": \"rank 0 OMP threads, first warm run\",\n");
        fprintf(f, "    \"tested_per_thread\": [");
        for (size_t i = 0; i < res.stats.thread_primes_tested.size(); i++)
            fprintf(f, "%" PRId64 "%s", res.stats.thread_primes_tested[i],
                    i == res.stats.thread_primes_tested.size() - 1 ? "" : ", ");
        fprintf(f, "],\n");
        fprintf(f, "    \"found_per_thread\": [");
        for (size_t i = 0; i < res.stats.thread_primes_found.size(); i++)
            fprintf(f, "%d%s", res.stats.thread_primes_found[i],
                    i == res.stats.thread_primes_found.size() - 1 ? "" : ", ");
        fprintf(f, "]\n  }%s", tail);
    }
    else
    {
        fprintf(f, "  \"thread_balance\": {}%s", tail);
    }
}

// ---------------------------------------------------------------------------
// merge_benchmark_results — combine timing vectors from multiple challenges
// ---------------------------------------------------------------------------

void merge_benchmark_results(benchmark_results_t &dst, const benchmark_results_t &src)
{
    auto &d = dst.timing;
    const auto &s = src.timing;
    // Capture sizes before appending so the weighted average uses pre-merge counts.
    int n_dst = (int)d.factor_times.size();
    int n_src = (int)s.factor_times.size();
    d.factor_times.insert(d.factor_times.end(),         s.factor_times.begin(),         s.factor_times.end());
    d.recon_times.insert(d.recon_times.end(),           s.recon_times.begin(),          s.recon_times.end());
    d.total_times.insert(d.total_times.end(),           s.total_times.begin(),          s.total_times.end());
    d.local_cbrt_times.insert(d.local_cbrt_times.end(), s.local_cbrt_times.begin(),     s.local_cbrt_times.end());
    d.gather_times.insert(d.gather_times.end(),         s.gather_times.begin(),         s.gather_times.end());
    d.combine_times.insert(d.combine_times.end(),       s.combine_times.begin(),        s.combine_times.end());
    d.blinding_attempts.insert(d.blinding_attempts.end(), s.blinding_attempts.begin(),  s.blinding_attempts.end());
    d.cbrt_hit_ms.insert(d.cbrt_hit_ms.end(),           s.cbrt_hit_ms.begin(),          s.cbrt_hit_ms.end());
    d.cbrt_miss_ms.insert(d.cbrt_miss_ms.end(),         s.cbrt_miss_ms.begin(),         s.cbrt_miss_ms.end());
    d.cbrt_hits.insert(d.cbrt_hits.end(),               s.cbrt_hits.begin(),            s.cbrt_hits.end());
    d.cbrt_misses.insert(d.cbrt_misses.end(),           s.cbrt_misses.begin(),          s.cbrt_misses.end());
    d.prefilter_times.insert(d.prefilter_times.end(),   s.prefilter_times.begin(),      s.prefilter_times.end());
    d.warmup_total_ms.insert(d.warmup_total_ms.end(),   s.warmup_total_ms.begin(),      s.warmup_total_ms.end());
    // powm_baseline_ms: weighted average across all merged challenges.
    // Guard: sieve/table use --skip-powm-baseline and write 0 — do not corrupt a valid value.
    int n_total = n_dst + n_src;
    if (s.powm_baseline_ms > 0 && n_total > 0)
        d.powm_baseline_ms = (d.powm_baseline_ms * n_dst + s.powm_baseline_ms * n_src)
                             / (double)n_total;
    auto &dh = dst.hits;
    const auto &sh = src.hits;
    dh.hit_type.insert(dh.hit_type.end(),                    sh.hit_type.begin(),             sh.hit_type.end());
    dh.cofactor_bits.insert(dh.cofactor_bits.end(),          sh.cofactor_bits.begin(),        sh.cofactor_bits.end());
    dh.medium_factor_count.insert(dh.medium_factor_count.end(), sh.medium_factor_count.begin(), sh.medium_factor_count.end());
    // Accumulate the count of warm challenges ATTEMPTED (each run_one sets 1,
    // including failures that produced no timing entry) so runs_failed is
    // recoverable as attempted − recorded.
    dst.config.n_warm_attempted += src.config.n_warm_attempted;
}

// ---------------------------------------------------------------------------
// write_benchmark_json — public entry point
//
// JSON layout:
//   metadata → warm_avg → warm_stats → warm_runs → [warmup_runs] →
//   [prime_stats] → [hit_stats] → thread_balance → [extra_sections]
//
// One benchmark covers MULTIPLE distinct challenges, run once each and merged.
// The cold warmup (challenge_0) is discarded upstream, so EVERY recorded entry
// is a warm success — `warm_runs` and the averages cover all of them.
//   runs_warm   = warm challenges attempted (config.n_warm_attempted)
//   runs_valid  = challenges that produced a timed result (factor_times.size())
//   runs_failed = runs_warm − runs_valid (blinding exhausted, recorded nothing)
// ---------------------------------------------------------------------------

void write_benchmark_json(const char *filepath,
                          const benchmark_results_t &res,
                          void (*extra_sections)(FILE*))
{
    FILE *f = fopen(filepath, "w");
    if (!f)
    {
        perror("[ERROR] fopen failed for JSON");
        return;
    }

    int n_recorded = (int)res.timing.factor_times.size();
    int n_warm = (res.config.n_warm_attempted > 0)
                     ? res.config.n_warm_attempted
                     : n_recorded; // fallback for a single non-merged run
    int n_valid  = n_recorded;     // every recorded run is a success post-Bug-17
    int n_failed = n_warm - n_recorded;
    if (n_failed < 0) n_failed = 0;

    fprintf(f, "{\n");
    json_write_metadata(f, res, n_warm, n_valid, n_failed);
    json_write_cold_run(f, res);
    json_write_warm_avg(f, res);
    json_write_warm_stats(f, res);
    json_write_warm_runs(f, res);
    json_write_warmup_runs(f, res);
    json_write_prime_stats(f, res);
    json_write_hit_stats(f, res);
    json_write_thread_balance(f, res, extra_sections != nullptr);
    if (extra_sections)
        extra_sections(f);
    fprintf(f, "}\n");

    fclose(f);
    printf("[INFO] JSON benchmark written to %s\n", filepath);
}



// Edit: 2026-06-12
