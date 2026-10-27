// =============================================================================
// gpu_common.h — Shared CUDA device utilities for Algorithm 2 GPU tools
//
// Included by every CUDA solver and GPU experiment.
//
// Device functions defined here:
//   mod_big_by_u32 — Horner reduction mod a prime q < 2^32; pure uint64_t arithmetic
//   mod_big_by_u64 — Horner reduction mod a prime q < 2^63; uses __uint128_t
//                    Required for k35 upper-range primes in (2^32, 2^35].
//   These return the remainder and are kept for the callers that need it
//   (alg7's sieve setup, the CRT experiment). Trial division only needs a
//   yes/no: use divides_big_u32/u64 from gpu_divides.h, which avoid the
//   software 64/128-bit division these spend nearly all their time in.
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "gpu_divides.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

// ---------------------------------------------------------------------------
// CUDA error-checking macro (identical to alg7_cuda.cu's version)
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t _err = (call);                                 \
        if (_err != cudaSuccess)                                   \
        {                                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(_err)); \
            exit(1);                                               \
        }                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Size constants
// ---------------------------------------------------------------------------

// MAX_LIMBS: max uint64_t limbs per big number uploaded to GPU.
// 512 limbs × 64 bits = 32768-bit numbers — well above this project's 24576-bit ceiling.
// Two MAX_LIMBS arrays fit in the 64 KB CUDA constant-memory budget.
#define MAX_LIMBS 512

// MAX_GPU_FACTORS: output buffer size for dividing primes per kernel launch.
// A 24576-bit k35-smooth number has at most log_{65537}(2^24576) ≈ 1536 prime factors
// (with multiplicity); in practice a smooth factorization has far fewer distinct primes.
// 2048 is a generous ceiling that avoids ever hitting the buffer limit.
#define MAX_GPU_FACTORS 2048

// ---------------------------------------------------------------------------
//
// This matters because the reported factor_ms measures the KERNEL ONLY: the
// cudaEventRecord bracket sits inside launch_all, after the challenge upload and
// before the factor download, so neither transfer is currently counted anywhere.
// Splitting setup from per-attempt traffic separates the one-time cost of
// staging the prime table (amortised over every challenge in a run) from the
// recurring per-attempt cost that a real attack would pay on every blind.
//
// All the copies measured here are synchronous with respect to the host
// (pageable memory, default stream), so host wall-clock is an accurate measure
// and no event bracketing is needed.
// ---------------------------------------------------------------------------

typedef struct
{
    double setup_h2d_ms; // one-time: prime / delta table upload
    size_t setup_h2d_bytes;
    long setup_h2d_calls;

    double challenge_h2d_ms; // per-attempt: blinded challenge limbs -> device
    size_t challenge_h2d_bytes;
    long challenge_h2d_calls;

    double factors_d2h_ms; // per-attempt: factor count + factor list -> host
    size_t factors_d2h_bytes;
    long factors_d2h_calls;
} gpu_xfer_prof_t;

static inline double gpu_now_ms(void)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

static inline void gpu_xfer_reset(gpu_xfer_prof_t *p)
{
    p->setup_h2d_ms = p->challenge_h2d_ms = p->factors_d2h_ms = 0.0;
    p->setup_h2d_bytes = p->challenge_h2d_bytes = p->factors_d2h_bytes = 0;
    p->setup_h2d_calls = p->challenge_h2d_calls = p->factors_d2h_calls = 0;
}

static inline double gpu_xfer_total_ms(const gpu_xfer_prof_t *p)
{
    return p->setup_h2d_ms + p->challenge_h2d_ms + p->factors_d2h_ms;
}

// Human table + greppable "XFERPROF|" lines. `tag` labels the run; factor_ms is
// the measured kernel time so the report can state transfer cost as a fraction
// of the work it brackets (pass 0 to omit that column).
static inline void gpu_xfer_report(const gpu_xfer_prof_t *p, const char *tag,
                                   double factor_ms)
{
    const char *t = tag ? tag : "";
    double tot = gpu_xfer_total_ms(p);
    double recur = p->challenge_h2d_ms + p->factors_d2h_ms;

    printf("\n--- host<->device transfer profile  (%s) ---\n", t);
    printf("  %-14s %10s %12s %14s %12s\n",
           "phase", "calls", "MiB", "ms", "GB/s");
    struct
    {
        const char *nm;
        double ms;
        size_t by;
        long ca;
    } rows[3] = {
        {"setup H2D", p->setup_h2d_ms, p->setup_h2d_bytes, p->setup_h2d_calls},
        {"challenge H2D", p->challenge_h2d_ms, p->challenge_h2d_bytes, p->challenge_h2d_calls},
        {"factors D2H", p->factors_d2h_ms, p->factors_d2h_bytes, p->factors_d2h_calls},
    };
    for (int i = 0; i < 3; i++)
    {
        double mib = (double)rows[i].by / (1024.0 * 1024.0);
        double gbs = rows[i].ms > 0.0
                         ? (double)rows[i].by / (rows[i].ms * 1e-3) / 1e9
                         : 0.0;
        printf("  %-14s %10ld %12.3f %14.3f %12.2f\n",
               rows[i].nm, rows[i].ca, mib, rows[i].ms, gbs);
    }
    printf("  %-14s %10s %12s %14.3f\n", "TOTAL", "", "", tot);
    printf("  recurring (per-attempt) transfer: %.3f ms", recur);
    if (factor_ms > 0.0)
        printf("  = %.2f%% of kernel time (%.3f ms)",
               recur / factor_ms * 100.0, factor_ms);
    printf("\n");

    printf("XFERPROF|%s|setup_h2d|%ld|%zu|%.4f\n",
           t, p->setup_h2d_calls, p->setup_h2d_bytes, p->setup_h2d_ms);
    printf("XFERPROF|%s|challenge_h2d|%ld|%zu|%.4f\n",
           t, p->challenge_h2d_calls, p->challenge_h2d_bytes, p->challenge_h2d_ms);
    printf("XFERPROF|%s|factors_d2h|%ld|%zu|%.4f\n",
           t, p->factors_d2h_calls, p->factors_d2h_bytes, p->factors_d2h_ms);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Device function: mod_big_by_u32
//
// Reduce a multi-limb big number (little-endian uint64_t, index 0 = least
// significant) modulo a small prime q < 2^32, using Horner's method from
// the most-significant limb downward:
//
//   acc = 0
//   for i = nlimbs-1 downto 0:
//     acc = (acc * 2^64 + limbs[i]) mod q
//
// Since q < 2^32, acc < q < 2^32 after every step. Products stay in uint64_t:
//   acc * b64  : both factors < 2^32 → product < 2^64  ✓
//   acc + limb_mod : both < q < 2^32 → sum < 2^33 < 2^64  ✓
// ---------------------------------------------------------------------------

__device__ static uint64_t mod_big_by_u32(const uint64_t *__restrict__ limbs,
                                          int nlimbs, uint64_t q)
{
    // b64 = 2^64 mod q: split as (2^32 mod q)^2 mod q to stay within 64 bits
    uint64_t b32 = ((uint64_t)1 << 32) % q;
    uint64_t b64 = (b32 * b32) % q; // b32 < q < 2^32 → b32*b32 < 2^64  ✓

    uint64_t acc = 0;
    for (int i = nlimbs - 1; i >= 0; i--)
    {
        acc = (acc * b64) % q; // acc < q after this
        acc += limbs[i] % q;   // sum < 2*q < 2^33
        if (acc >= q)
            acc -= q; // single correction: restores acc < q
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Device function: mod_big_by_u64
//
// Same Horner reduction as mod_big_by_u32 but for prime q up to 2^63.
// Needed for k35 upper-range primes in (2^32, 2^35] where q > 2^32 and
// products (acc * b64) may exceed uint64_t.  Uses __uint128_t for those
// multiplications, which CUDA supports natively on sm_70 (V100) and above.
// ---------------------------------------------------------------------------

__device__ static uint64_t mod_big_by_u64(const uint64_t *__restrict__ limbs,
                                          int nlimbs, uint64_t q)
{
    typedef unsigned __int128 u128;
    // b64 = 2^64 mod q
    uint64_t b32 = (uint64_t)(((u128)1 << 32) % (u128)q);
    uint64_t b64 = (uint64_t)(((u128)b32 * (u128)b32) % (u128)q);

    uint64_t acc = 0;
    for (int i = nlimbs - 1; i >= 0; i--)
    {
        acc = (uint64_t)(((u128)acc * (u128)b64) % (u128)q);
        uint64_t limb_mod = (uint64_t)((u128)(limbs[i]) % (u128)q);
        acc += limb_mod;
        if (acc >= q)
            acc -= q;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Host-side benchmark helpers
// (static inline — host only; nvcc passes these to g++, not the device compiler)
// ---------------------------------------------------------------------------
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <gmp.h>

typedef enum
{
    GPU_METHOD_SIEVE = 0,
    GPU_METHOD_TABLE = 1,
    GPU_METHOD_POWM = 2
} gpu_method_t;

// Hit statistics collected per warm run — mirrors the CPU solver's hit_stats JSON section.
struct GpuHitStats
{
    std::vector<int> hit_type;            // 0=B-smooth, 1=prime cofactor, 2=composite
    std::vector<int> cofactor_bits;       // bit-length of remaining cofactor (0 if B-smooth)
    std::vector<int> medium_factor_count; // 0=B-smooth, 1=prime cofactor, ≥2=composite factors
};

// GPU memory transfer timing per warm run.
// h2d_ms: host→device transfer (challenge value / blinding candidate upload).
// d2h_ms: device→host transfer (factor list or log-sum array download).
// table_mb: prime table size in MB (constant per binary — used for effective-BW analysis).
// For k32/alg3/alg4: h2d ≈ 4 KB (tiny); d2h ≈ 1–4 KB (factor list).
// For alg7: h2d ≈ 4 KB; d2h ≈ 4 MB (R=2^20 log-sum floats — may be significant).
struct GpuTransferStats
{
    std::vector<double> h2d_ms; // per warm run — H2D wall time minus kernel event time
    std::vector<double> d2h_ms; // per warm run — collect_factors / log-sum download
    double table_mb = 0.0;      // prime table bytes / (1024*1024)
};

// ---------------------------------------------------------------------------
// Warm-up pass support
//
// Reconstruction looks up cbrt(p) for every prime factor it finds; a miss costs
// one full-width mpz_powm, whose price is set by |M| and not by p (~1.8 s at
// 24576 bits).  A challenge set of N distinct challenges therefore pays for
// several hundred exponentiations the first time it is factored, and only runs
// at cache speed afterwards.
//
// The CPU solvers avoid this by building the cache eagerly before timing
// (Plan A).  The GPU solvers instead run the whole warm loop twice and time
// only the second pass, which needs no separate factoring path.  This call
// discards whatever the untimed pass recorded.  It deliberately does NOT touch
// the cube-root cache — populating that is the entire point of the first pass.
// ---------------------------------------------------------------------------
static inline void gpu_reset_warm_accumulators(std::vector<double> &factor_ms_v,
                                               std::vector<double> &recon_ms_v,
                                               GpuHitStats &hits,
                                               GpuTransferStats &xfer)
{
    factor_ms_v.clear();
    recon_ms_v.clear();
    hits.hit_type.clear();
    hits.cofactor_bits.clear();
    hits.medium_factor_count.clear();
    const double table_mb = xfer.table_mb; // set once at setup, not per run
    xfer.h2d_ms.clear();
    xfer.d2h_ms.clear();
    xfer.table_mb = table_mb;
}

// Per-GPU device state for multi-GPU prime-range splitting.
struct GpuDev
{
    uint32_t *d_primes;
    int64_t n_slice;
    uint64_t *d_out_factors;
    int *d_out_count;
    cudaEvent_t ev0, ev1;
};

static inline const char *gpu_method_name(gpu_method_t m)
{
    if (m == GPU_METHOD_TABLE)
        return "table";
    if (m == GPU_METHOD_POWM)
        return "powm";
    return "sieve";
}

static inline gpu_method_t gpu_parse_method(const char *s)
{
    if (!s)
        return GPU_METHOD_TABLE;
    if (strcmp(s, "sieve") == 0)
        return GPU_METHOD_SIEVE;
    if (strcmp(s, "powm") == 0)
        return GPU_METHOD_POWM;
    return GPU_METHOD_TABLE;
}

// Returns average time in ms of 5 sequential mpz_powm_ui calls (1 warmup).
static inline double gpu_measure_powm_baseline(const mpz_t chal,
                                               const mpz_t mod,
                                               unsigned long exp_e)
{
    mpz_t tmp;
    mpz_init(tmp);
    mpz_powm_ui(tmp, chal, exp_e, mod); // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; i++)
        mpz_powm_ui(tmp, chal, exp_e, mod);
    auto t1 = std::chrono::high_resolution_clock::now();
    mpz_clear(tmp);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
}

// ---- Stats helpers ---------------------------------------------------------
static inline double _gpu_vmean(const std::vector<double> &v)
{
    if (v.empty())
        return 0.0;
    double s = 0;
    for (auto x : v)
        s += x;
    return s / v.size();
}
static inline double _gpu_vstd(const std::vector<double> &v, double mean)
{
    if (v.size() < 2)
        return 0.0;
    double s = 0;
    for (auto x : v)
        s += (x - mean) * (x - mean);
    return sqrt(s / (v.size() - 1));
}
static inline double _gpu_vpct(std::vector<double> v, double p)
{
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo = (size_t)idx, hi = lo + 1;
    if (hi >= v.size())
        return v.back();
    return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}
static inline void _gpu_write_int_arr(FILE *f, const char *name,
                                      const std::vector<int> &v, bool last)
{
    fprintf(f, "    \"%s\": [", name);
    for (size_t i = 0; i < v.size(); i++)
        fprintf(f, "%d%s", v[i], i + 1 < v.size() ? ", " : "");
    fprintf(f, "]%s\n", last ? "" : ",");
}
static inline void _gpu_write_dbl_arr(FILE *f, const char *name,
                                      const std::vector<double> &v, bool last)
{
    fprintf(f, "    \"%s\": [", name);
    for (size_t i = 0; i < v.size(); i++)
        fprintf(f, "%.4f%s", v[i], i + 1 < v.size() ? ", " : "");
    fprintf(f, "]%s\n", last ? "" : ",");
}
static inline void _gpu_write_stats_obj(FILE *f, const char *name,
                                        const std::vector<double> &v, bool last)
{
    if (v.empty())
    {
        fprintf(f, "    \"%s\": {}%s\n", name, last ? "" : ",");
        return;
    }
    double mean = _gpu_vmean(v);
    double std_v = _gpu_vstd(v, mean);
    std::vector<double> vs = v;
    std::sort(vs.begin(), vs.end());
    fprintf(f,
            "    \"%s\": {\"mean\": %.4f, \"std\": %.4f, \"min\": %.4f, "
            "\"max\": %.4f, \"median\": %.4f, \"p25\": %.4f, \"p75\": %.4f}%s\n",
            name, mean, std_v, vs.front(), vs.back(),
            _gpu_vpct(v, 0.5), _gpu_vpct(v, 0.25), _gpu_vpct(v, 0.75),
            last ? "" : ",");
}

// Name of the active CUDA device (e.g. "Tesla V100-SXM2-16GB", "NVIDIA L40"),
// queried once from the driver.  Replaces the old hard-coded "V100" so the JSON
// metadata reflects whatever GPU actually ran (V100 / L40 / ...).  Falls back to
// "GPU" if the query fails.
static inline const char *gpu_device_name()
{
    static char name[256] = "";
    if (name[0])
        return name; // cached after first call
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        snprintf(name, sizeof(name), "%s", prop.name);
    else
        snprintf(name, sizeof(name), "GPU");
    return name;
}

// Print human-readable benchmark summary to stdout.
static inline void gpu_print_benchmark(const char *label, int n_warm,
                                       double powm_ms,
                                       double avg_factor_ms, double avg_recon_ms,
                                       int n_gpus = 1)
{
    double total_ms = avg_factor_ms + avg_recon_ms;
    double speedup = total_ms > 0.0 ? powm_ms / total_ms : 0.0;
    printf("\n=== %s — GPU Benchmark (%s, %d GPU%s) ===\n",
           label, gpu_device_name(), n_gpus, n_gpus == 1 ? "" : "s");
    printf("  powm_baseline_ms : %8.4f  (5-rep avg, sequential mpz_powm)\n", powm_ms);
    printf("  factor_ms  (GPU) : %8.4f  (warm avg over %d runs)\n", avg_factor_ms, n_warm);
    printf("  recon_ms   (CPU) : %8.4f  (warm avg)\n", avg_recon_ms);
    printf("  total_ms         : %8.4f\n", total_ms);
    printf("  speedup_total    : %8.2f x  (powm_baseline / total_ms)\n", speedup);
    printf("=================================================\n");
}

// Write full warm-run JSON matching CPU benchmark_results format.
// factor_ms_v / recon_ms_v: one entry per warm run (empty for powm method).
// powm_baseline_ms: 0.0 for sieve/table (CPU convention); actual value for powm method.
static inline void gpu_write_json_warm(const char *path,
                                       const char *algorithm,
                                       gpu_method_t method,
                                       unsigned long exp_e,
                                       int smoothness_bound_bits,
                                       size_t challenge_bits,
                                       double powm_baseline_ms,
                                       double cold_factor_ms,
                                       double cold_recon_ms,
                                       const std::vector<double> &factor_ms_v,
                                       const std::vector<double> &recon_ms_v,
                                       int solved,
                                       int n_gpus = 1,
                                       const GpuHitStats &hit_stats = GpuHitStats{},
                                       const GpuTransferStats &xfer = GpuTransferStats{})
{
    if (!path || path[0] == '\0')
        return;
    const char *mname = gpu_method_name(method);
    int n = (int)factor_ms_v.size();

    std::vector<double> total_ms_v(n), spd_v(n);
    for (int i = 0; i < n; i++)
    {
        total_ms_v[i] = factor_ms_v[i] + recon_ms_v[i];
        spd_v[i] = (powm_baseline_ms > 0 && total_ms_v[i] > 0)
                       ? powm_baseline_ms / total_ms_v[i]
                       : 0.0;
    }
    double f_avg = _gpu_vmean(factor_ms_v);
    double r_avg = _gpu_vmean(recon_ms_v);
    double t_avg = _gpu_vmean(total_ms_v);
    double sp_avg = (powm_baseline_ms > 0 && t_avg > 0) ? powm_baseline_ms / t_avg : 0.0;

    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Warning: cannot write JSON to %s\n", path);
        return;
    }

    fprintf(f,
            "{\n  \"metadata\": {\n"
            "    \"algorithm\": \"%s\",\n    \"method\": \"%s\",\n"
            "    \"exponent\": %lu,\n    \"smoothness_bound_bits\": %d,\n"
            "    \"challenge_bits\": %zu,\n    \"mpi_ranks\": 1,\n"
            "    \"nodes\": 1,\n    \"gpu\": \"%s\",\n    \"platform\": \"IRIS\",\n"
            "    \"n_gpus\": %d,\n"
            "    \"runs_total\": %d,\n    \"runs_warm\": %d,\n"
            "    \"powm_baseline_ms\": %.4f\n  },\n",
            algorithm, mname, exp_e, smoothness_bound_bits, challenge_bits,
            gpu_device_name(), n_gpus, n + 1, n, powm_baseline_ms);

    if (method != GPU_METHOD_POWM)
    {
        fprintf(f,
                "  \"cold_run\": {\n"
                "    \"factor_ms\": %.4f,\n    \"recon_ms\": %.4f,\n"
                "    \"total_ms\": %.4f\n  },\n",
                cold_factor_ms, cold_recon_ms, cold_factor_ms + cold_recon_ms);
    }
    else
    {
        fprintf(f, "  \"cold_run\": {},\n");
    }

    if (method == GPU_METHOD_POWM || n == 0)
    {
        fprintf(f,
                "  \"warm_avg\": {\n"
                "    \"factor_ms\": 0.0000,\n    \"recon_ms\": 0.0000,\n"
                "    \"local_cbrt_ms\": 0.0000,\n    \"gather_ms\": 0.0000,\n"
                "    \"combine_ms\": 0.0000,\n    \"total_ms\": 0.0000,\n"
                "    \"speedup_recon\": 0.0000,\n    \"speedup_total\": 0.0000\n  },\n");
    }
    else
    {
        fprintf(f,
                "  \"warm_avg\": {\n"
                "    \"factor_ms\": %.4f,\n    \"recon_ms\": %.4f,\n"
                "    \"local_cbrt_ms\": 0.0000,\n    \"gather_ms\": 0.0000,\n"
                "    \"combine_ms\": 0.0000,\n    \"total_ms\": %.4f,\n"
                "    \"speedup_recon\": 0.0000,\n    \"speedup_total\": %.4f\n  },\n",
                f_avg, r_avg, t_avg, sp_avg);
    }

    fprintf(f, "  \"warm_stats\": {\n");
    _gpu_write_stats_obj(f, "factor_ms", factor_ms_v, false);
    _gpu_write_stats_obj(f, "recon_ms", recon_ms_v, false);
    _gpu_write_stats_obj(f, "total_ms", total_ms_v, false);
    _gpu_write_stats_obj(f, "speedup_total", spd_v, true);
    fprintf(f, "  },\n");

    fprintf(f, "  \"warm_runs\": {\n");
    if (n > 0)
    {
        _gpu_write_dbl_arr(f, "factor_ms", factor_ms_v, false);
        _gpu_write_dbl_arr(f, "recon_ms", recon_ms_v, false);
        _gpu_write_dbl_arr(f, "total_ms", total_ms_v, false);
        _gpu_write_dbl_arr(f, "speedup_total", spd_v, true);
    }
    fprintf(f, "  },\n");

    fprintf(f, "  \"hit_stats\": {\n");
    if (!hit_stats.hit_type.empty())
    {
        _gpu_write_int_arr(f, "hit_type", hit_stats.hit_type, false);
        _gpu_write_int_arr(f, "cofactor_bits", hit_stats.cofactor_bits, false);
        _gpu_write_int_arr(f, "medium_factor_count", hit_stats.medium_factor_count, true);
    }
    fprintf(f, "  },\n");

    fprintf(f, "  \"gpu_timing\": {\n");
    fprintf(f, "    \"table_mb\": %.3f,\n", xfer.table_mb);
    _gpu_write_dbl_arr(f, "h2d_ms", xfer.h2d_ms, false);
    _gpu_write_dbl_arr(f, "d2h_ms", xfer.d2h_ms, false);
    _gpu_write_stats_obj(f, "h2d_stats", xfer.h2d_ms, false);
    _gpu_write_stats_obj(f, "d2h_stats", xfer.d2h_ms, true);
    fprintf(f, "  },\n");

    fprintf(f, "  \"gpu_result\": {\"solved\": %s}\n}\n", solved ? "true" : "false");
    fclose(f);
    printf("JSON written to %s\n", path);
}

// ---------------------------------------------------------------------------
// Multi-challenge helper
// Requires: common.h included before or after this header by the translation unit.
// ---------------------------------------------------------------------------
void read_mpz_from_file(mpz_t out, const char *path); // defined in common.cpp

// Load challenge_idx from data_dir using naming: challenge_<idx>_<bound>_<size>.txt
// Reduces y mod modulo in-place. Returns false (and prints a warning) if missing.
static inline bool gpu_load_challenge(mpz_t y_out, const char *data_dir,
                                      int idx, const char *bound_str, int size_bits,
                                      const mpz_t modulo)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/challenge_%d_%s_%d.txt",
             data_dir, idx, bound_str, size_bits);
    FILE *chk = fopen(path, "rb");
    if (!chk)
    {
        fprintf(stderr, "  SKIP (missing: %s)\n", path);
        return false;
    }
    fclose(chk);
    read_mpz_from_file(y_out, path);
    mpz_mod(y_out, y_out, modulo);
    return true;
}