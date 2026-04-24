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

bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo)
{
    mpz_t phi, inverse, three;
    mpz_inits(phi, inverse, three, NULL);

    // phi = modulo - 1  (Euler's totient for a prime)
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);

    // compute e = 3^-1 mod phi  (exists iff gcd(3, phi) = 1)
    if (mpz_invert(inverse, three, phi) == 0)
    {
        // cube root does not exist for this modulo
        mpz_set_ui(result, 0);
        mpz_clears(phi, inverse, three, NULL);
        return false;
    }

    // result = value^e mod modulo — this is the cube root
    mpz_powm(result, value, inverse, modulo);
    mpz_clears(phi, inverse, three, NULL);
    return true;
}

void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod,
                      const mpz_t modulo)
{
    mpz_t cubed;
    mpz_init(cubed);
    // check: does candidate^3 mod M equal the original challenge mod M?
    mpz_powm_ui(cubed, candidate_cuberoot, 3, modulo);
    if (mpz_cmp(cubed, challenge_mod) == 0)
        printf("SUCCESS: Cube root is correct!\n");
    else
        printf("FAILURE: Cube root is incorrect\n");
    mpz_clear(cubed);
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

// Computes y = (c * r^3) mod M
void blind_challenge(mpz_t y, const mpz_t challenge, const mpz_t r, const mpz_t modulo)
{
    mpz_t r3;
    mpz_init(r3);
    mpz_powm_ui(r3, r, 3, modulo); // r^3 mod M
    mpz_mul(y, challenge, r3);     // c * r^3
    mpz_mod(y, y, modulo);         // (c * r^3) mod M
    mpz_clear(r3);
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
                         int label_width)
{
    mpz_t e, phi, three;
    mpz_inits(e, phi, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);
    mpz_clears(phi, three, NULL);

    double sum = 0.0, sumsq = 0.0;

    for (int i = 0; i < bench_count; i++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        mpz_powm(result, challenge, e, modulo);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum += dt;
        sumsq += dt * dt;
    }

    mpz_clear(e);

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

    // Helper lambda to write double arrays cleanly
    auto write_dbl_arr = [&](const char *key, const std::vector<double> &vec, bool is_last = false)
    {
        if (vec.empty())
            return;
        fprintf(f, "    \"%s\": [", key);
        for (size_t i = 0; i < vec.size(); ++i)
        {
            fprintf(f, "%.3f%s", vec[i], (i == vec.size() - 1) ? "" : ", ");
        }
        fprintf(f, "]%s\n", is_last ? "" : ",");
    };

    // Helper lambda to write int arrays
    auto write_int_arr = [&](const char *key, const std::vector<int> &vec, bool is_last = false)
    {
        if (vec.empty())
            return;
        fprintf(f, "    \"%s\": [", key);
        for (size_t i = 0; i < vec.size(); ++i)
        {
            fprintf(f, "%d%s", vec[i], (i == vec.size() - 1) ? "" : ", ");
        }
        fprintf(f, "]%s\n", is_last ? "" : ",");
    };

    fprintf(f, "{\n");
    fprintf(f, "  \"metadata\": {\n");
    fprintf(f, "    \"k_size\": %d,\n", res.k_size);
    fprintf(f, "    \"challenge_bits\": %zu,\n", res.challenge_bits);
    fprintf(f, "    \"method\": \"%s\",\n", res.method_name);
    fprintf(f, "    \"threads_per_rank\": %d,\n", res.num_threads);
    fprintf(f, "    \"mpi_ranks\": %d,\n", res.mpi_ranks);
    fprintf(f, "    \"runs\": %d\n", res.bench_count);
    fprintf(f, "  },\n");

    fprintf(f, "  \"powm_baseline_ms\": %.3f,\n", res.powm_baseline_ms);

    fprintf(f, "  \"raw_data\": {\n");

    // Core Timings
    write_dbl_arr("factor_ms", res.factor_times);
    write_dbl_arr("recon_ms", res.recon_times);

    // MPI Timings (Only prints if populated)
    write_dbl_arr("local_cbrt_ms", res.local_cbrt_times);
    write_dbl_arr("gather_ms", res.gather_times);
    write_dbl_arr("combine_ms", res.combine_times);
    write_int_arr("blinding_attempts", res.blinding_attempts);

    // Total always printed last
    write_dbl_arr("total_ms", res.total_times, true);

    fprintf(f, "  }\n");
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