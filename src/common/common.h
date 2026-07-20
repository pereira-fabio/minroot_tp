#pragma once

// =============================================================================
// common.h — Shared utilities for modular e-th root solvers
//
// Mathematical context:
//   We compute challenge^(1/e) mod M, where M is prime and gcd(e, M-1) = 1.
//   e = 3 gives the cube root (original MinRoot VDF), e = 5 gives the 5th root.
//
//   By Fermat's little theorem:
//     x^(1/e) mod M  =  x^e_inv mod M,   e_inv = e^{-1} mod (M-1)
//
//   Homomorphism exploited for factorization-based solvers:
//     eth_root(a * b) = eth_root(a) * eth_root(b) mod M
//
//   So if challenge = p1^k1 * p2^k2 * ... * pN^kN, then:
//     eth_root(challenge) = eth_root(p1)^k1 * ... * eth_root(pN)^kN mod M
//
//   This lets us replace one expensive powm on a large number with many
//   cheap powm calls on small primes — precomputed once per modulo file.
// =============================================================================

#include <gmp.h>
#include <vector>
#include <cstdint>
#include <cinttypes>

// ---------------------------------------------------------------------------
// Factoring method — shared across all solvers
// ---------------------------------------------------------------------------

typedef enum
{
    METHOD_SIEVE = 0,
    METHOD_TABLE,
    METHOD_POWM
} run_method_t;

// Repetitions used to measure the sequential powm baseline per challenge.
// (Replaces the removed --bench repeat count; the baseline is stable so a few
// reps suffice for a robust mean.)
#define KSC_POWM_BASELINE_REPS 5

// ---------------------------------------------------------------------------
// Prime generation
// ---------------------------------------------------------------------------

// Generate all primes up to `limit` using the Sieve of Eratosthenes.
// Returns a heap-allocated array; sets *count. Caller must free().
unsigned long *generate_primes_up_to(unsigned long limit, int *count);

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

// Read a GMP integer from file. Tries decimal text first, then GMP binary
// (mpz_inp_raw) as fallback. Exits on failure.
void read_mpz_from_file(mpz_t num, const char *filename);

// ---------------------------------------------------------------------------
// e-th root — naive baseline (METHOD_POWM in k16)
// ---------------------------------------------------------------------------

// Compute result = value^e_inv mod modulo, where e_inv = exponent^{-1} mod (modulo-1).
// This is O(log modulo) squarings — correct but not parallelizable.
// Used as comparison baseline and for small factor roots in k16.
// Returns false if exponent has no inverse mod (modulo-1).
bool compute_eth_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo,
                                unsigned long exponent);

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

// Verify that candidate^exponent mod modulo == challenge_mod.
// Prints SUCCESS or FAILURE.
void verify_eth_root(const mpz_t candidate_root, const mpz_t challenge_mod,
                     const mpz_t modulo, unsigned long exponent);

// ---------------------------------------------------------------------------
// Factoring helper
// ---------------------------------------------------------------------------

// Return the exact exponent of `prime` in n (i.e. largest k s.t. prime^k | n).
// Fast path: mpz_tdiv_ui checks divisibility without any GMP allocation;
// returns 0 immediately for the ~99.9% of primes that don't divide n.
int count_exponent(const mpz_t n, unsigned long prime);

// ---------------------------------------------------------------------------
// Benchmark/reporting helpers
// ---------------------------------------------------------------------------

// Compute standard deviation from raw timing samples.
// sample_stddev=true uses Bessel correction (n-1), false uses population (n).
double compute_stddev_from_values(const double *vals, int n, bool sample_stddev);

// Compute standard deviation from sum and sum-of-squares accumulators.
// sample_stddev=true uses Bessel correction (n-1), false uses population (n).
double compute_stddev_from_moments(double sum, double sumsq, int n,
                                   bool sample_stddev);

// Print one aligned timing line in the common format:
//   "    <label> avg=... ms   stddev=... ms"
void print_timing_stat_moments(const char *label,
                               double sum, double sumsq, int n,
                               int label_width, bool sample_stddev);

// Convenience wrapper — standard 30-char label width, sample stddev.
// Replaces the per-file print_stat / _ps macros.
static inline void kprint_stat(const char *lbl, double sum, double sumsq, int n)
{
    print_timing_stat_moments(lbl, sum, sumsq, n, 30, true);
}

// ---------------------------------------------------------------------------
// Sequential powm baseline — the comparison target for all parallel solvers
// ---------------------------------------------------------------------------

// Compute challenge^e_inv mod modulo sequentially for `reps` iterations,
// where e_inv = exponent^{-1} mod (modulo-1).  This is the O(log M) squaring
// baseline: every squaring depends on the previous one so it cannot be parallelised.
// In the VDF context this is exactly what the ASIC executes each round.
//
// Prints one timing line using print_timing_stat_moments, then returns the
// average time in ms so callers can compute speedup ratios.
//
// result receives the last computed root (can be used for verification).
double run_powm_baseline(mpz_t result, const mpz_t challenge,
                         const mpz_t modulo, int reps,
                         int label_width, unsigned long exponent);

// ---------------------------------------------------------------------------
// Unified factor list — (prime, exponent) pairs with mpz_t primes
//
// Using mpz_t for the prime field handles all bit widths (16, 32, 64) without
// separate struct definitions in each solver file. The overhead versus a plain
// unsigned long is negligible — mpz_t allocation only happens on append, and
// the factor count is always small (~N_bits / prime_bits, typically < 200).
// ---------------------------------------------------------------------------

typedef struct
{
    mpz_t prime;
    int exponent;
} kfactor_t;

typedef struct
{
    kfactor_t *items;
    int count;
    int capacity;
} kfactor_list_t;

// Initialise an empty list (must be called before first use).
void kfl_init(kfactor_list_t *f);

// Append a (prime, exponent) entry. prime is copied via mpz_init_set.
// Accepts unsigned long for convenience (k32 path).
void kfl_append_ul(kfactor_list_t *f, unsigned long prime, int exponent);

// Append from an existing mpz_t (k64 / Pollard's rho path).
void kfl_append_mpz(kfactor_list_t *f, const mpz_t prime, int exponent);

// Release all mpz_t entries and free the backing array.
void kfl_free(kfactor_list_t *f);

// ---------------------------------------------------------------------------
// Timing accumulator — (sum, sum-of-squares) pair
//
// Wraps the repetitive  sum += dt; sumsq += dt*dt;  pattern.
// Use accum_record() inside the bench loop, then print with
// print_timing_stat_moments(label, a.sum, a.sumsq, n, width, true).
// ---------------------------------------------------------------------------

typedef struct
{
    double sum;
    double sumsq;
} bench_accum_t;

static inline void accum_record(bench_accum_t *a, double dt)
{
    a->sum += dt;
    a->sumsq += dt * dt;
}

static inline double accum_mean(const bench_accum_t *a, int n)
{
    return n > 0 ? a->sum / n : 0.0;
}

// ---------------------------------------------------------------------------
// benchmark_results_t — SRP split into four focused sub-structs
//
// bench_config_t   — immutable description of the experiment (set once)
// bench_timing_t   — per-run timing vectors (grow during bench loop)
// bench_stats_t    — gather statistics (populated once per benchmark run)
// alg_hit_data_t   — cofactor classification (alg3/alg4 only; empty otherwise)
// ---------------------------------------------------------------------------

struct bench_config_t
{
    int           smoothness_bits   = 0;       // B exponent: 16 / 32 / 35
    size_t        challenge_bits    = 0;       // modulus bit-length
    const char   *method_name       = nullptr; // "sieve" / "table" / "powm"
    const char   *algorithm         = "";      // "k16"/"k32"/"k35"/"alg3"/"alg4"/"alg7"
    unsigned long exponent          = 3;       // e-th root exponent (3 or 5)
    unsigned long b0                = 0;        // alg4 pre-filter bound (0 if n/a)
    int           t_threshold_bits  = 0;        // alg4 discard threshold bits (0 if n/a)
    int           num_threads       = 0;
    int           mpi_ranks         = 0;
    int           n_warm_attempted  = 0;       // warm challenges attempted (for runs_failed)
};

struct bench_timing_t
{
    // Per-run timing (one entry per warm run)
    std::vector<double> factor_times;
    std::vector<double> recon_times;
    std::vector<double> total_times;
    std::vector<double> local_cbrt_times;
    std::vector<double> gather_times;
    std::vector<double> combine_times;
    std::vector<int>    blinding_attempts;
    std::vector<double> cbrt_hit_ms;
    std::vector<double> cbrt_miss_ms;
    std::vector<int>    cbrt_hits;
    std::vector<int>    cbrt_misses;
    std::vector<double> prefilter_times; // alg4 pre-filter time per run (empty otherwise)
    std::vector<double> warmup_total_ms;
    double              powm_baseline_ms = 0.0;
};

struct bench_stats_t
{
    // Prime distribution across MPI ranks (recorded from first warm run)
    std::vector<int64_t> primes_tested_per_rank;
    std::vector<int>     primes_found_per_rank;
    // Per-thread load balance within rank 0 (first warm run only)
    std::vector<int64_t> thread_primes_tested;
    std::vector<int>     thread_primes_found;
};

// Algorithm-specific cofactor classification — empty for k16/k32/k35
struct alg_hit_data_t
{
    std::vector<int> hit_type;            // 1=B-smooth, 2=one medium prime, 3=composite
    std::vector<int> cofactor_bits;       // bit-length of cofactor after trial division
    std::vector<int> medium_factor_count; // number of medium primes per run
};

struct benchmark_results_t
{
    bench_config_t config;
    bench_timing_t timing;
    bench_stats_t  stats;
    alg_hit_data_t hits;
};

// Serialise benchmark results to a JSON file.
//
// extra_sections (Change 9 / OCP hook): if non-NULL, called with the open
// FILE* after the standard sections. The callback writes one or more
// complete JSON object members, each ending with ",\n". It must NOT close
// the object (the main function does that). Pass NULL (default) when no
// algorithm-specific sections are needed.
//
// Example usage:
//   write_benchmark_json(path, res, [](FILE *f) {
//       fprintf(f, "  \"prefilter_ms\": %.4f,\n", val);
//   });
void write_benchmark_json(const char *filepath,
                          const benchmark_results_t &results,
                          void (*extra_sections)(FILE*) = nullptr);

// Append all timing vectors from src into dst.
// Call once per warm challenge (challenges 1..N-1) after using challenge_0
// as the cold-warmup run.  dst must be initialised from the first warm
// challenge (copy assignment).  powm_baseline_ms is kept from dst unchanged
// (same modulus across challenges — variance is negligible).
// dst.config.n_warm_attempted accumulates the warm challenges attempted.
void merge_benchmark_results(benchmark_results_t &dst,
                             const benchmark_results_t &src);

// Edit: 2026-07-17
