#pragma once

// =============================================================================
// k_common.h — Shared utilities for modular cube root solvers
//
// Mathematical context:
//   We compute challenge^(1/3) mod M, where M is prime and gcd(3, M-1) = 1.
//
//   By Fermat's little theorem:
//     x^(1/3) mod M  =  x^e mod M,   e = 3^-1 mod (M-1)
//
//   Homomorphism exploited for factorization-based solvers:
//     cbrt(a * b) = cbrt(a) * cbrt(b) mod M
//
//   So if challenge = p1^e1 * p2^e2 * ... * pN^eN, then:
//     cbrt(challenge) = cbrt(p1)^e1 * cbrt(p2)^e2 * ... * cbrt(pN)^eN mod M
//
//   This lets us replace one expensive powm on a large number with many
//   cheap powm calls on small primes — precomputed once per modulo file.
// =============================================================================

#include <gmp.h>
#include <vector>
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
// Cube root — naive baseline (METHOD_POWM in k16)
// ---------------------------------------------------------------------------

// Compute result = value^e mod modulo, where e = 3^-1 mod (modulo-1).
// This is O(log modulo) squarings — correct but not parallelizable.
// Used as comparison baseline and for small factor cube roots in k16.
// Returns false if 3 has no inverse mod (modulo-1).
bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo);

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

// Verify that candidate^3 mod modulo == challenge_mod.
// Prints SUCCESS or FAILURE.
void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod,
                      const mpz_t modulo);

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

// Print one aligned average-only timing line in the common format:
//   "    <label> avg=... ms"
// If extra_text is non-null/non-empty, it is appended after the avg.
void print_timing_total(const char *label, double mean,
                        int label_width, const char *extra_text);

void blind_challenge(mpz_t y, const mpz_t challenge, const mpz_t r, const mpz_t modulo);
void unblind_root(mpz_t result, const mpz_t y_root, const mpz_t r, const mpz_t modulo);

// ---------------------------------------------------------------------------
// Sequential powm baseline — the comparison target for all parallel solvers
// ---------------------------------------------------------------------------

// Compute challenge^e mod modulo sequentially for bench_count iterations,
// where e = 3^-1 mod (modulo-1).  This is the O(log M) squaring baseline:
// every squaring depends on the previous one so it cannot be parallelised.
// In the VDF context this is exactly what the ASIC executes each round.
//
// Prints one timing line using print_timing_stat_moments, then returns the
// average time in ms so callers can compute speedup ratios.
//
// result receives the last computed cube root (can be used for verification).
double run_powm_baseline(mpz_t result, const mpz_t challenge,
                         const mpz_t modulo, int bench_count,
                         int label_width);

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
// Solver config — shared CLI state parsed by parse_common_args()
// ---------------------------------------------------------------------------

typedef struct
{
    const char *challenge_file;
    const char *modulo_file;
    int bench_count;
    const char *json_output_file;
    bool force_factor;
} solver_config_t;

// Parse positional <challenge> <modulo> and --bench N from argv.
// File defaults are applied before this call; this function overwrites them
// only when the caller explicitly provides them on the command line.
// Returns 0 on success, 1 if an unrecognised flag is encountered (caller
// should then handle it and advance argi past any consumed tokens).
// Unknown flags are NOT consumed — the caller's loop must handle them.
int parse_common_args(int argc, char **argv, solver_config_t *cfg,
                      int *argi_inout);

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

static inline void accum_reset(bench_accum_t *a)
{
    a->sum = 0.0;
    a->sumsq = 0.0;
}

static inline void accum_record(bench_accum_t *a, double dt)
{
    a->sum += dt;
    a->sumsq += dt * dt;
}

static inline double accum_mean(const bench_accum_t *a, int n)
{
    return n > 0 ? a->sum / n : 0.0;
}

typedef struct
{
    int k_size;
    size_t challenge_bits;
    const char *method_name;
    int num_threads;
    int mpi_ranks; 
    int bench_count;

    // Detailed timing lists (raw data)
    std::vector<double> factor_times;
    std::vector<double> recon_times;
    std::vector<double> total_times;

    // MPI-Specific timing lists (left empty for k16/k32 non-MPI)
    std::vector<double> local_cbrt_times;
    std::vector<double> gather_times;
    std::vector<double> combine_times;
    std::vector<int> blinding_attempts;

    double powm_baseline_ms;
} benchmark_results_t;

void write_benchmark_json(const char *filepath, const benchmark_results_t &results);