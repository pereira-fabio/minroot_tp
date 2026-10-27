// =============================================================================
// alg7_cuda.cu — GPU-accelerated Algorithm 7 (inverted sieve), Phase 1 on CUDA
//
// WHY THIS EXISTS
// ----------------
// Algorithm 7 (Biryukov et al. CRYPTO 2024, Sec. 3.5) inverts the usual
// blind-and-test loop: instead of testing ONE candidate against many primes,
// it tests MANY candidates against each prime simultaneously. For a fixed
// challenge x and modulus M, it considers R lifted integer candidates
//     y_r = x + r * M ,   r = 0 .. R-1
// and for each prime q <= B, marks every r for which q | y_r by computing
//     r_start = (-x * M^{-1}) mod q      (period q among the R candidates)
// then striding r_start, r_start+q, r_start+2q, ... and adding log2(q) into a
// shared log_sum[R] array. This is, structurally, a classic sieve-of-
// Eratosthenes-style "same tiny operation, millions of independent data
// points" workload -- exactly what a GPU is built for, and a much better fit
// than the multi-thousand-bit modular exponentiations elsewhere in this
// project (see wiki/results/ — the "would GPU help" discussion, 2026-06-25).
//
// This file implements ONLY that Phase 1 marking step as a CUDA kernel.
// Everything else (file I/O, prime generation, Phase 2 full trial division,
// reconstruction, verification) reuses the EXISTING MPI-free utilities in
// src/common/common.cpp and src/common/factorization.cpp (now at src/common/),
// linked in by the
// Makefile -- no MPI dependency anywhere in this tool, by design (mirrors
// tools/timelock/'s "standalone, not integrated" precedent).
//
// SCOPE
// -----
// Single-process, single-GPU prototype. No MPI, no multi-GPU. It reads the
// SAME challenge/modulo file format already produced by tools/challenges.c
// (GMP mpz_out_raw binary), so it can be pointed directly at real generated
// files, e.g.:
//     ./alg7_cuda data/challenges/alg3/challenge_0_alg3_1024.txt
//                 data/challenges/alg3/modulo_alg3_1024.txt
// (Algorithm 7 reuses alg3-style "smooth_base * q0" challenges, same as the
// existing CPU alg7.cpp -- see wiki/algorithms/algorithm3.md.)
//
// NOT YET VALIDATED ON REAL HARDWARE — there is no CUDA toolkit or GPU
// available in the environment this file was written in. It was written
// carefully and reasoned through by hand, but treat it as an unverified
// first draft until it's been built and run on an actual GPU node. See
// gpu/README.md for the build/test plan.
//
// A note on why Algorithm 7 needs no "unblinding" step, unlike Algo 2/3/4:
// y_r = x + r*M as an INTEGER is congruent to x mod M for every r (the r*M
// term vanishes mod M). So reconstructing the e-th root of y_r from its
// integer factorization directly gives root(y_r) mod M = root(x) mod M --
// no multiplication by r^{-1} is needed anywhere. (Algorithms 2-4 multiply
// by a blinding factor r^e mod M, which DOES need undoing with r^{-1}; this
// algorithm instead additively *lifts* the candidate, which is why it
// doesn't.)
// =============================================================================

#include "gpu_common.h"
#include "gpu_recon.cuh" // --recon host|device
#include "common.h"
#include "factorization.h"

#include <cuda_runtime.h>
#include <gmp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

__constant__ uint64_t d_x_limbs[MAX_LIMBS];
__constant__ uint64_t d_m_limbs[MAX_LIMBS];
__constant__ int d_nlimbs;

// Limbs of the single winning candidate y = x + best_r*M, uploaded before the
// Phase-2 GPU factoring kernel (see factor_candidate_kernel / gpu_factor_candidate).
__constant__ uint64_t d_y_limbs[MAX_LIMBS];
__constant__ int d_y_nlimbs;

// ---------------------------------------------------------------------------
// Device helpers — alg7-specific (mulmod_u32 used in inverted_sieve_kernel;
// mod_big_by_u32 provided by gpu_common.h)
// ---------------------------------------------------------------------------

// (a * b) mod q, valid whenever a < q and b < q and q < 2^32 (so a*b < 2^64).
__device__ __forceinline__ uint64_t mulmod_u32(uint64_t a, uint64_t b, uint64_t q)
{
    return (a * b) % q;
}

// Modular inverse of a mod m via the extended Euclidean algorithm.
// Safe for a, m < 2^32 using signed 64-bit intermediates (Bezout
// coefficients for inputs this size never approach int64_t's range).
// Precondition: gcd(a, m) == 1 (guaranteed here since m=q is prime and
// a = M mod q is nonzero — M is prime and q != M, so q never divides M).
__device__ uint64_t modinv_u64(uint64_t a, uint64_t m)
{
    int64_t old_r = (int64_t)a, r = (int64_t)m;
    int64_t old_s = 1, s = 0;
    while (r != 0)
    {
        int64_t quotient = old_r / r;
        int64_t tmp_r = old_r - quotient * r;
        old_r = r;
        r = tmp_r;
        int64_t tmp_s = old_s - quotient * s;
        old_s = s;
        s = tmp_s;
    }
    int64_t result = old_s % (int64_t)m;
    if (result < 0)
        result += (int64_t)m;
    return (uint64_t)result;
}

// ---------------------------------------------------------------------------
// The Phase 1 kernel: one grid-stride loop over the rank's prime slice.
// Each thread independently computes r_start for its prime(s) and marks
// log_sum[] via atomicAdd — modern GPUs handle float atomicAdd on global
// memory efficiently, and since different primes mark different stride
// patterns, true collisions (two threads writing the exact same r at the
// exact same instant) are infrequent relative to total work. A per-block
// shared-memory partial-sum scheme would reduce atomic traffic further but
// is deliberately deferred (see README) to keep this first version simple
// and easier to verify correct.
// ---------------------------------------------------------------------------

__global__ void inverted_sieve_kernel(const uint32_t *primes, int64_t nprimes,
                                      int64_t R, float *log_sum)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < nprimes; i += stride)
    {
        uint64_t q = primes[i];

        uint64_t x_mod_q = mod_big_by_u32(d_x_limbs, d_nlimbs, q);
        uint64_t M_mod_q = mod_big_by_u32(d_m_limbs, d_nlimbs, q);
        uint64_t M_inv_q = modinv_u64(M_mod_q, q);

        uint64_t neg_x_mod_q = (x_mod_q == 0) ? 0 : (q - x_mod_q);
        uint64_t r_start = mulmod_u32(neg_x_mod_q, M_inv_q, q);

        float log_q = log2f((float)q);
        for (int64_t r = (int64_t)r_start; r < R; r += (int64_t)q)
        {
            atomicAdd(&log_sum[r], log_q);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase-2 kernel: trial-divide the SINGLE winning candidate y (in d_y_limbs)
// by every sieve prime, recording each prime q with q | y into out_primes[].
// This replaces the old serial host re-scan (kf_trial_divide_table over all
// ~200M primes, which dominated recon_ms) with the same on-device trial
// division that k32 uses — so alg7 factors on the GPU like the other solvers.
// Multiplicities are recovered on the host afterwards (see gpu_factor_candidate).
// ---------------------------------------------------------------------------

__global__ void factor_candidate_kernel(const uint32_t *primes, int64_t nprimes,
                                        uint32_t *out_primes, int *out_count,
                                        int max_out)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < nprimes; i += stride)
    {
        uint64_t q = primes[i];
        if (divides_big_u32(d_y_limbs, d_y_nlimbs, (uint32_t)q))
        {
            int slot = atomicAdd(out_count, 1);
            if (slot < max_out)
                out_primes[slot] = (uint32_t)q;
        }
    }
}

// ---------------------------------------------------------------------------
// Host: export an mpz_t's limbs (little-endian uint64_t words) into a
// fixed-size buffer suitable for upload to __constant__ memory.
// ---------------------------------------------------------------------------

static int export_limbs(const mpz_t value, uint64_t *out, int max_limbs)
{
    memset(out, 0, sizeof(uint64_t) * max_limbs);
    size_t count = 0;
    mpz_export(out, &count, -1 /* least significant word first */, sizeof(uint64_t),
               0 /* native endian within a word */, 0 /* nails */, value);
    if (count > (size_t)max_limbs)
    {
        fprintf(stderr, "Error: modulus/challenge exceeds MAX_LIMBS=%d (%zu limbs needed)\n",
                max_limbs, count);
        exit(1);
    }
    return (int)count;
}

// ---------------------------------------------------------------------------
// GPU trial-division of a single candidate y over the sieve primes.
// Drop-in replacement for kf_trial_divide_table(y, primes, nprimes, found):
// produces the identical factor list (every sieve prime dividing y, with its
// full multiplicity), but does the ~200M-prime scan on the GPU instead of one
// host thread. Multiplicity is recovered on the host by dividing y out fully
// per distinct prime the kernel reports (the kernel reports each prime once).
// ---------------------------------------------------------------------------

// Per-device state for the multi-GPU prime-range split. Each device owns a
// contiguous slice of the sieve primes plus its own Phase-1 log_sum[R] and
// Phase-2 factor-output buffers.
struct Alg7Dev
{
    int64_t p_lo;           // first prime index this device owns
    int64_t p_count;        // number of primes in this device's slice
    uint32_t *d_primes;     // this device's prime slice
    float *d_log_sum;       // R floats — this device's PARTIAL Phase-1 sum
    uint32_t *d_out_primes; // Phase-2 dividing-prime output
    int *d_out_count;
    cudaEvent_t ev0, ev1;
};

// Multi-GPU Phase-2: trial-divide the winning candidate y across ALL devices'
// prime slices, then merge the dividing primes. The slices are disjoint, so the
// union has no duplicates; multiplicities are recovered on the host by dividing
// y out fully per reported prime. Produces the identical factor list to a single
// device scanning the whole prime table.
static void gpu_factor_candidate(const mpz_t y, kfactor_list_t *found,
                                 const Alg7Dev *devs, int n_gpus,
                                 int tpb, int max_out)
{
    uint64_t y_limbs[MAX_LIMBS];
    int y_nl = export_limbs(y, y_limbs, MAX_LIMBS);

    // Launch the factoring kernel on every device over its prime slice.
    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpyToSymbol(d_y_limbs, y_limbs, sizeof(uint64_t) * MAX_LIMBS));
        CUDA_CHECK(cudaMemcpyToSymbol(d_y_nlimbs, &y_nl, sizeof(int)));
        CUDA_CHECK(cudaMemset(devs[g].d_out_count, 0, sizeof(int)));
        int64_t blocks = (devs[g].p_count + tpb - 1) / tpb;
        if (blocks < 1)
            blocks = 1;
        factor_candidate_kernel<<<(unsigned)blocks, tpb>>>(devs[g].d_primes, devs[g].p_count,
                                                           devs[g].d_out_primes, devs[g].d_out_count, max_out);
        CUDA_CHECK(cudaGetLastError());
    }

    // Gather + merge the dividing primes from all devices (disjoint slices → no dups).
    mpz_t t;
    mpz_init_set(t, y);
    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        int h_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_count, devs[g].d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_count > max_out)
        {
            // Overflow: more distinct factors on this device than the buffer holds.
            // Not silent — missing primes leave a composite cofactor, so the
            // downstream cofactor_ok check fails and the run is flagged unsolved.
            fprintf(stderr, "Warning: factor_candidate overflow on GPU %d (%d > %d) — increase MAX_OUT_FACTORS\n",
                    g, h_count, max_out);
            h_count = max_out;
        }
        if (h_count <= 0)
            continue;
        std::vector<uint32_t> h_primes(h_count);
        CUDA_CHECK(cudaMemcpy(h_primes.data(), devs[g].d_out_primes,
                              sizeof(uint32_t) * h_count, cudaMemcpyDeviceToHost));
        for (int i = 0; i < h_count; i++)
        {
            unsigned long q = h_primes[i];
            int e = 0;
            while (mpz_divisible_ui_p(t, q))
            {
                mpz_divexact_ui(t, t, q);
                e++;
            }
            if (e > 0)
                kfl_append_ul(found, q, e);
        }
    }
    mpz_clear(t);
}

// ---------------------------------------------------------------------------
// Host: simple Sieve of Eratosthenes producing a uint32_t prime list in
// (lo, hi], matching the project's k32_primes.bin convention (uint32_t,
// since B=2^32 - 1 still fits) and the [65537, B] range used throughout this
// project (Gap 8: crafted challenges never carry factors <= 65536, by
// construction in tools/challenges.c, so the sieve doesn't need to test
// them — see wiki/planning/gap-analysis.md Gap 8).
// ---------------------------------------------------------------------------

static uint32_t *sieve_primes_u32(uint32_t lo, uint64_t hi, int64_t *out_count)
{
    std::vector<bool> is_composite(hi + 1, false);
    for (uint64_t p = 2; p * p <= hi; p++)
    {
        if (is_composite[p])
            continue;
        for (uint64_t m = p * p; m <= hi; m += p)
            is_composite[m] = true;
    }
    int64_t count = 0;
    for (uint64_t p = lo; p <= hi; p++)
        if (p >= 2 && !is_composite[p])
            count++;
    uint32_t *primes = (uint32_t *)malloc(sizeof(uint32_t) * count);
    int64_t j = 0;
    for (uint64_t p = lo; p <= hi; p++)
        if (p >= 2 && !is_composite[p])
            primes[j++] = (uint32_t)p;
    *out_count = count;
    return primes;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s <challenge_file> <modulo_file> [options]\n"
            "  --method {sieve|powm}  benchmark method (default: sieve)\n"
            "  --R N                  candidates to test, R = 2^N (default: 20)\n"
            "  --bitbound B           sieve up to 2^B (default: 24; use 32 for full k32 range)\n"
            "  --exponent E           root exponent (default: 3)\n"
            "  --medium-bits K        accept cofactor prime up to 2^K (default: 45)\n"
            "  --n-runs N             warm runs (default: 10)\n"
            "  --n-gpus N             split the prime range across N devices (default: 1)\n"
            "  --json FILE            write JSON output\n",
            prog);
}

int main(int argc, char **argv)
{
    gpu_recon_parse_args(&argc, argv); // consumes --recon before the parser below
    const char *challenge_file = nullptr;
    const char *modulo_file = nullptr;
    const char *data_dir = nullptr;
    // Suffix on the modulus filename, e.g. "_e5" for the exponent-5 moduli that
    // challenge_gen_e5.slurm generates with gcd(5, M-1) = 1. Challenge files are
    // exponent-agnostic and shared, so only the modulus changes. Mirrors
    // --modulo-suffix on the MPI solvers.
    const char *modulo_suffix = "";
    int size_bits = 0;
    int r_log2 = 20;
    int bitbound = 24;
    unsigned long exponent = 3;
    int medium_bits = 45;
    int n_challenges = 10;
    int n_gpus = 1;
    gpu_method_t method = GPU_METHOD_SIEVE;
    const char *json_path = nullptr;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--data") && i + 1 < argc)
            data_dir = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc)
            size_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--modulo-suffix") && i + 1 < argc)
            modulo_suffix = argv[++i];
        else if (!strcmp(argv[i], "--n-challenges") && i + 1 < argc)
            n_challenges = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-runs") && i + 1 < argc)
            n_challenges = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-gpus") && i + 1 < argc)
            n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--R") && i + 1 < argc)
            r_log2 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bitbound") && i + 1 < argc)
            bitbound = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exponent") && i + 1 < argc)
            exponent = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--medium-bits") && i + 1 < argc)
            medium_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--method") && i + 1 < argc)
            method = gpu_parse_method(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_path = argv[++i];
        else if (argv[i][0] != '-')
        {
            if (!challenge_file)
                challenge_file = argv[i];
            else if (!modulo_file)
                modulo_file = argv[i];
        }
    }

    // Derive paths from --data/--size when given
    static char _modu_buf[512], _chal_buf[512];
    if (data_dir && size_bits > 0)
    {
        snprintf(_modu_buf, sizeof(_modu_buf), "%s/modulo_alg3_%d%s.txt", data_dir, size_bits, modulo_suffix);
        snprintf(_chal_buf, sizeof(_chal_buf), "%s/challenge_0_alg3_%d.txt", data_dir, size_bits);
        modulo_file = _modu_buf;
        challenge_file = _chal_buf;
    }
    else if (!challenge_file || !modulo_file)
    {
        print_usage(argv[0]);
        return 1;
    }

    if (bitbound > 32)
    {
        fprintf(stderr, "Error: --bitbound > 32 not supported; clamping to 32\n");
        bitbound = 32;
    }
    int64_t R = (int64_t)1 << r_log2;
    uint64_t B = (bitbound == 32) ? 0xFFFFFFFFULL : ((uint64_t)1 << bitbound);

    mpz_t challenge, modulo, challenge_mod;
    mpz_inits(challenge, modulo, challenge_mod, nullptr);
    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);
    mpz_mod(challenge_mod, challenge, modulo);

    size_t challenge_bits = mpz_sizeinbase(modulo, 2);
    printf("Challenge: %zu bits | Modulus: %zu bits | Exponent: %lu | Method: %s\n",
           mpz_sizeinbase(challenge, 2), challenge_bits, exponent, gpu_method_name(method));

    printf("Measuring powm baseline...\n");
    double powm_baseline_ms = gpu_measure_powm_baseline(challenge_mod, modulo, exponent);
    printf("  powm_baseline_ms = %.4f ms\n", powm_baseline_ms);

    if (method == GPU_METHOD_POWM)
    {
        std::vector<double> empty;
        gpu_write_json_warm(json_path, "alg7", method, exponent, 32, challenge_bits,
                            powm_baseline_ms, 0.0, 0.0, empty, empty, 0);
        mpz_clears(challenge, modulo, challenge_mod, nullptr);
        return 0;
    }

    if ((int)challenge_bits / 64 + 1 > MAX_LIMBS)
    {
        fprintf(stderr, "Error: modulus too large for MAX_LIMBS=%d\n", MAX_LIMBS);
        return 1;
    }

    // ---- Generate primes (host sieve) ----
    auto t_sieve0 = std::chrono::high_resolution_clock::now();
    int64_t nprimes = 0;
    uint32_t *primes = sieve_primes_u32(65537, B, &nprimes);
    double sieve_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t_sieve0)
                          .count();
    printf("Generated %ld primes in (65536, 2^%d] in %.1f ms (host)\n", (long)nprimes, bitbound, sieve_ms);

    // ---- Clamp n_gpus to visible devices (mirrors alg2_k32_cuda.cu) ----
    {
        int avail = 0;
        cudaGetDeviceCount(&avail);
        if (n_gpus < 1)
            n_gpus = 1;
        if (avail > 0 && n_gpus > avail)
        {
            fprintf(stderr, "Warning: requested %d GPUs but only %d visible; using %d\n",
                    n_gpus, avail, avail);
            n_gpus = avail;
        }
    }

    // ---- GPU setup ----
    uint64_t x_limbs[MAX_LIMBS], m_limbs[MAX_LIMBS];
    int x_nl = export_limbs(challenge_mod, x_limbs, MAX_LIMBS);
    int m_nl = export_limbs(modulo, m_limbs, MAX_LIMBS);
    int nlimbs = (x_nl > m_nl) ? x_nl : m_nl;

    // Phase-2 GPU factoring output: distinct sieve primes dividing the winner.
    // 4096 slots covers challenges well past the 24576-bit ceiling (a B-smooth
    // n has <= |n|/16 factors with primes > 2^16); overflow is flagged, not silent.
    const int MAX_OUT_FACTORS = 4096;
    int threads_per_block = 256;

    // ---- Per-device setup: split the prime range across n_gpus devices ----
    // INTERLEAVED (strided) assignment: device g owns primes[g], primes[g+n_gpus],
    // primes[g+2*n_gpus], ...  The inverted-sieve marking cost per prime q is
    // O(R/q) — dominated by the SMALL primes (q near 65537 make ~R/65537 marks;
    // q near 2^32 make 0-1). A contiguous count-split would hand the entire
    // small-prime band to GPU 0 and leave the other devices nearly idle, so
    // factor_ms = max(device) would stay pinned to GPU 0 and Phase 1 would barely
    // scale. Striding gives every device an equal share of every magnitude band →
    // Phase 1 scales ~n_gpus. The union is still all primes, disjoint; Phase-1
    // log_sums are summed on the host and Phase-2 factor reports are
    // order-independent, so the reordering does not affect correctness.
    std::vector<Alg7Dev> devs((size_t)n_gpus);
    {
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            int64_t p_count = ((int64_t)g < nprimes)
                                  ? ((nprimes - g + n_gpus - 1) / n_gpus)
                                  : 0;
            devs[g].p_lo = g; // stride start index (informational)
            devs[g].p_count = p_count;
            // Gather this device's strided prime subset into a contiguous host buffer.
            std::vector<uint32_t> h_slice((size_t)p_count);
            for (int64_t k = 0; k < p_count; k++)
                h_slice[(size_t)k] = primes[g + k * n_gpus];
            // Constant memory (challenge x, modulus M, nlimbs) on THIS device.
            CUDA_CHECK(cudaMemcpyToSymbol(d_x_limbs, x_limbs, sizeof(uint64_t) * MAX_LIMBS));
            CUDA_CHECK(cudaMemcpyToSymbol(d_m_limbs, m_limbs, sizeof(uint64_t) * MAX_LIMBS));
            CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &nlimbs, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&devs[g].d_primes, sizeof(uint32_t) * p_count));
            CUDA_CHECK(cudaMemcpy(devs[g].d_primes, h_slice.data(),
                                  sizeof(uint32_t) * p_count, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&devs[g].d_log_sum, sizeof(float) * R));
            CUDA_CHECK(cudaMalloc(&devs[g].d_out_primes, sizeof(uint32_t) * MAX_OUT_FACTORS));
            CUDA_CHECK(cudaMalloc(&devs[g].d_out_count, sizeof(int)));
            CUDA_CHECK(cudaEventCreate(&devs[g].ev0));
            CUDA_CHECK(cudaEventCreate(&devs[g].ev1));
            printf("  GPU %d: %ld primes (interleaved: stride %d from index %d, of %ld)\n",
                   g, (long)p_count, n_gpus, g, (long)nprimes);
        }
    }

    // Combined host log-sum: sum of every device's PARTIAL log_sum[R].
    std::vector<float> h_log_sum((size_t)R);
    std::vector<float> tmp_log_sum((size_t)R);

    // Helper: run Phase 1 on every device (each over its prime slice), sum the
    // partial log_sums into h_log_sum, and return the max kernel time across devices.
    auto run_phase1 = [&]() -> float
    {
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemset(devs[g].d_log_sum, 0, sizeof(float) * R));
            int64_t blocks = (devs[g].p_count + threads_per_block - 1) / threads_per_block;
            if (blocks < 1)
                blocks = 1;
            CUDA_CHECK(cudaEventRecord(devs[g].ev0));
            inverted_sieve_kernel<<<(unsigned)blocks, threads_per_block>>>(
                devs[g].d_primes, devs[g].p_count, R, devs[g].d_log_sum);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(devs[g].ev1));
        }
        float km = 0;
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaEventSynchronize(devs[g].ev1));
            float t = 0;
            CUDA_CHECK(cudaEventElapsedTime(&t, devs[g].ev0, devs[g].ev1));
            if (t > km)
                km = t;
        }
        // Combine: h_log_sum[r] = sum over devices of their partial log_sum[r].
        memset(h_log_sum.data(), 0, sizeof(float) * (size_t)R);
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(tmp_log_sum.data(), devs[g].d_log_sum,
                                  sizeof(float) * R, cudaMemcpyDeviceToHost));
            for (int64_t r = 0; r < R; r++)
                h_log_sum[(size_t)r] += tmp_log_sum[(size_t)r];
        }
        return km;
    };

    // ---- Discovery run: find best_r ----
    float disc_km = run_phase1(); // fills h_log_sum (summed across all devices)

    double total_bits = (double)challenge_bits + r_log2;
    double threshold = total_bits - medium_bits;
    int64_t best_r = -1;
    float best_score = -1.0f;
    for (int64_t r = 0; r < R; r++)
    {
        if (h_log_sum[r] > best_score)
        {
            best_score = h_log_sum[r];
            best_r = r;
        }
    }
    printf("Phase 1: %.3f ms, best_r=%ld, log_sum=%.2f bits (threshold=%.2f)\n",
           disc_km, (long)best_r, best_score, threshold);
    if (best_score < threshold)
    {
        printf("Warning: no almost-smooth candidate found in R=2^%d window. "
               "Try a larger --R or larger --bitbound.\n",
               r_log2);
    }

    // ---- Build y = challenge_mod + best_r * modulo ----
    mpz_t y, r_mpz, cofactor, result;
    mpz_inits(y, r_mpz, cofactor, result, nullptr);
    mpz_set_ui(r_mpz, (unsigned long)best_r);
    mpz_mul(y, r_mpz, modulo);
    mpz_add(y, y, challenge_mod);

    // ---- Phase 2: trial divide y, check cofactor ----
    kfactor_list_t found;
    kfl_init(&found);
    gpu_factor_candidate(y, &found, devs.data(), n_gpus, threads_per_block, MAX_OUT_FACTORS);
    mpz_set(cofactor, y);
    for (int i = 0; i < found.count; i++)
        for (int k = 0; k < found.items[i].exponent; k++)
            mpz_divexact(cofactor, cofactor, found.items[i].prime);
    unsigned long cofactor_bits_ul = mpz_sizeinbase(cofactor, 2);
    bool cofactor_ok = (mpz_cmp_ui(cofactor, 1) == 0) ||
                       ((long)cofactor_bits_ul <= medium_bits && mpz_probab_prime_p(cofactor, 25) > 0);
    bool has_q0 = (mpz_cmp_ui(cofactor, 1) != 0) && cofactor_ok;
    printf("Phase 2 (discovery): %d factors, cofactor %lu bits, ok=%d\n",
           found.count, cofactor_bits_ul, (int)cofactor_ok);

    if (!cofactor_ok)
    {
        fprintf(stderr, "Warning: cofactor check failed — not smooth enough. Try larger --R.\n");
    }

    // ---- cbrt cache (initialized once, outside the warm loop) ----
    cbrt_cache_t cache;
    kf_cbrt_cache_init(&cache);
    bool cache_dirty = false;
    // Room for a 512-byte modulo path plus the ".cbrt_algN" suffix. Sized to
    // 512 a long path truncates, and two moduli differing only past the cut
    // would share one cache file. Matches KSC_CBRT_PATH_MAX on the CPU side,
    // which cannot be included here (it lives in the MPI header).
    char cache_path[600];
    snprintf(cache_path, sizeof(cache_path), "%s.cbrt_alg7", modulo_file);
    kf_cbrt_cache_load(&cache, cache_path);

    int solved = cofactor_ok ? 1 : 0;

    // Helper: run Phase 2 (trial div on y + reconstruction), return recon_ms.
    // Clears and rebuilds found each call so timing is stable across warm runs.
    auto run_recon = [&]() -> double
    {
        kfl_free(&found);
        kfl_init(&found);
        auto tr0 = std::chrono::high_resolution_clock::now();
        gpu_factor_candidate(y, &found, devs.data(), n_gpus, threads_per_block, MAX_OUT_FACTORS);
        kf_compute_cbrt_from_factors_omp(result, &found, modulo, &cache, &cache_dirty, exponent);
        if (has_q0)
        {
            mpz_t cbrt_q;
            mpz_init(cbrt_q);
            compute_eth_root_mod_prime(cbrt_q, cofactor, modulo, exponent);
            mpz_mul(result, result, cbrt_q);
            mpz_mod(result, result, modulo);
            mpz_clear(cbrt_q);
        }
        // no r^{-1} unblinding: y_r ≡ x (mod M) for every r
        return std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - tr0)
            .count();
    };

    // ---- Cold run (warms cbrt cache) ----
    float cold_km = run_phase1();
    double cold_recon_ms = run_recon();
    if (solved)
        verify_eth_root(result, challenge_mod, modulo, exponent);
    printf("Cold run: factor_ms=%.2f, recon_ms=%.4f\n", (double)cold_km, cold_recon_ms);

    // ---- Warm runs: one per challenge file (new x uploaded to constant memory each time) ----
    std::vector<double> factor_ms_v, recon_ms_v;
    factor_ms_v.reserve(n_challenges - 1);
    recon_ms_v.reserve(n_challenges - 1);
    GpuHitStats hit_stats;
    GpuTransferStats xfer;
    xfer.table_mb = (double)nprimes * 4.0 / (1024.0 * 1024.0);

    // Two passes over the challenge set. The first is untimed and exists only
    // to populate the cube-root cache: a cold cache makes reconstruction pay one
    // full-width exponentiation per distinct prime factor, which dominates every
    // other cost at large moduli. Timing the second pass measures the warm steady
    // state, matching how the CPU solvers benchmark.
    for (int pass = 0; pass < 2; pass++)
    {
        if (pass == 1)
            gpu_reset_warm_accumulators(factor_ms_v, recon_ms_v, hit_stats, xfer);

        for (int ci = 1; ci < n_challenges; ci++)
        {
            mpz_t x_w;
            mpz_init(x_w);
            if (data_dir && size_bits > 0)
            {
                if (!gpu_load_challenge(x_w, data_dir, ci, "alg3", size_bits, modulo))
                {
                    mpz_clear(x_w);
                    continue;
                }
            }
            else
            {
                mpz_set(x_w, challenge_mod); // single-challenge compat
            }

            // Upload new x_w to every device's constant memory
            uint64_t x_lims_w[MAX_LIMBS] = {};
            size_t xl_cnt = 0;
            mpz_export(x_lims_w, &xl_cnt, -1, sizeof(uint64_t), 0, 0, x_w);
            auto t_h2d0 = std::chrono::high_resolution_clock::now();
            for (int g = 0; g < n_gpus; g++)
            {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMemcpyToSymbol(d_x_limbs, x_lims_w, sizeof(uint64_t) * MAX_LIMBS));
            }
            auto t_h2d1 = std::chrono::high_resolution_clock::now();

            float km = run_phase1(); // fills h_log_sum (summed across all devices)

            // Find best_r for this x_w from the combined log-sum.
            auto t_d2h0 = std::chrono::high_resolution_clock::now();
            int64_t br_w = 0;
            float bs_w = -1.0f;
            for (int64_t r = 0; r < R; r++)
                if (h_log_sum[r] > bs_w)
                {
                    bs_w = h_log_sum[r];
                    br_w = r;
                }
            auto t_d2h1 = std::chrono::high_resolution_clock::now();
            xfer.h2d_ms.push_back(std::chrono::duration<double, std::milli>(t_h2d1 - t_h2d0).count());
            xfer.d2h_ms.push_back(std::chrono::duration<double, std::milli>(t_d2h1 - t_d2h0).count());

            // Build y_w = x_w + best_r_w * modulo (integer lift)
            mpz_t y_w, r_mpz_w;
            mpz_inits(y_w, r_mpz_w, nullptr);
            mpz_set_ui(r_mpz_w, (unsigned long)br_w);
            mpz_mul(y_w, r_mpz_w, modulo);
            mpz_add(y_w, y_w, x_w);

            // Phase 2 (CPU): trial divide y_w, reconstruct
            auto tr0 = std::chrono::high_resolution_clock::now();
            kfactor_list_t fl_w;
            kfl_init(&fl_w);
            gpu_factor_candidate(y_w, &fl_w, devs.data(), n_gpus, threads_per_block, MAX_OUT_FACTORS);
            mpz_t cof_w;
            mpz_init_set(cof_w, y_w);
            for (int j = 0; j < fl_w.count; j++)
                for (int k = 0; k < fl_w.items[j].exponent; k++)
                    mpz_divexact(cof_w, cof_w, fl_w.items[j].prime);
            kf_compute_cbrt_from_factors_omp(result, &fl_w, modulo, &cache, &cache_dirty, exponent);
            if (mpz_cmp_ui(cof_w, 1) != 0 && (long)mpz_sizeinbase(cof_w, 2) <= medium_bits && mpz_probab_prime_p(cof_w, 25) > 0)
            {
                mpz_t cbrt_q_w;
                mpz_init(cbrt_q_w);
                compute_eth_root_mod_prime(cbrt_q_w, cof_w, modulo, exponent);
                mpz_mul(result, result, cbrt_q_w);
                mpz_mod(result, result, modulo);
                mpz_clear(cbrt_q_w);
            }
            // no r_inv: alg7 additive lift y_r ≡ x (mod M)
            auto tr1 = std::chrono::high_resolution_clock::now();

            // alg7 finds one medium prime cofactor (same structure as alg3, type-1)
            {
                int h_type = 0, h_bits = 0, h_med = 0;
                if (mpz_cmp_ui(cof_w, 1) != 0 && (long)mpz_sizeinbase(cof_w, 2) <= medium_bits && mpz_probab_prime_p(cof_w, 25) > 0)
                {
                    h_type = 1;
                    h_bits = (int)mpz_sizeinbase(cof_w, 2);
                    h_med = 1;
                }
                hit_stats.hit_type.push_back(h_type);
                hit_stats.cofactor_bits.push_back(h_bits);
                hit_stats.medium_factor_count.push_back(h_med);
            }

            factor_ms_v.push_back((double)km);
            recon_ms_v.push_back(std::chrono::duration<double, std::milli>(tr1 - tr0).count());
            kfl_free(&fl_w);
            mpz_clears(y_w, r_mpz_w, cof_w, x_w, nullptr);
        }
    }
    // Restore original x to every device's constant memory for consistency
    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpyToSymbol(d_x_limbs, x_limbs, sizeof(uint64_t) * MAX_LIMBS));
    }

    gpu_print_benchmark("alg7", (int)factor_ms_v.size(), powm_baseline_ms,
                        _gpu_vmean(factor_ms_v), _gpu_vmean(recon_ms_v), n_gpus);
    gpu_write_json_warm(json_path, "alg7", method, exponent, 32, challenge_bits,
                        0.0, cold_km, cold_recon_ms, factor_ms_v, recon_ms_v, solved,
                        n_gpus, hit_stats, xfer);

    // ---- Cleanup ----
    if (cache_dirty)
        kf_cbrt_cache_save(&cache, cache_path);
    kf_cbrt_cache_clear(&cache);
    kfl_free(&found);
    mpz_clears(y, r_mpz, cofactor, result, nullptr);
    mpz_clears(challenge, modulo, challenge_mod, nullptr);
    free(primes);
    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(devs[g].d_primes));
        CUDA_CHECK(cudaFree(devs[g].d_log_sum));
        CUDA_CHECK(cudaFree(devs[g].d_out_primes));
        CUDA_CHECK(cudaFree(devs[g].d_out_count));
        CUDA_CHECK(cudaEventDestroy(devs[g].ev0));
        CUDA_CHECK(cudaEventDestroy(devs[g].ev1));
    }

    return solved ? 0 : 2;
}