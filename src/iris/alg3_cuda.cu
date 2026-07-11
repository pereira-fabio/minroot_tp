// =============================================================================
// alg3_cuda.cu — Algorithm 3 GPU: k32 trial division + medium prime cofactor
//
// Mirrors the CPU alg3 structure: --method {powm|sieve|table}, cold + N warm runs,
// same JSON format (warm_avg / warm_stats / warm_runs).
//
// GPU kernel: identical to alg2_k32_cuda — all ~203 M k32 primes tested.
//   With --n-gpus N, the prime range is split across N devices; CPU merges.
// CPU cofactor check (alg3-specific):
//   Type 0: cofactor == 1        -> fully B-smooth
//   Type 1: cofactor is prime <= 2^65 -> almost-smooth (one medium prime q0)
//
// Challenge file stores y directly; attempt 0 (unblinded) is always accepted.
// Warm runs re-execute the same kernel + reconstruction for N timing samples.
//
// CLI:
//   ./alg3_cuda <challenge_file> <modulo_file> [options]
//     --method {powm|sieve|table}  benchmark method (default: table)
//     --exponent E            root exponent (default: 3)
//     --primes FILE           k32 prime table (default: data/k32_primes.bin)
//     --n-runs N              warm runs (default: 10)
//     --n-gpus N              split prime range across N CUDA devices (default: 1)
//     --json FILE             write JSON output
// =============================================================================

#include "gpu_common.h"
#include "common.h"
#include "factorization.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <chrono>
#include <vector>
// ---------------------------------------------------------------------------
// Parallel segmented sieve helper for --method sieve.
// Each OpenMP thread owns its own is_comp[] segment and output vector.
// Results are merged in order after the parallel section.
// Requires -Xcompiler -fopenmp at build time.
// ---------------------------------------------------------------------------

static uint32_t *sieve_primes_u32(uint64_t lo, uint64_t hi, int64_t *out_count) {
    int n_small = 0;
    unsigned long *small = generate_primes_up_to((uint64_t)(sqrt((double)hi)) + 2, &n_small);

    const uint64_t SEG = (uint64_t)1 << 22;  // 4M candidates per segment (4 MB bool vec)
    int64_t n_segs = (int64_t)((hi - lo + SEG) / SEG);
    std::vector<std::vector<uint32_t>> seg_out((size_t)n_segs);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t s = 0; s < n_segs; s++) {
        uint64_t seg_lo = lo + (uint64_t)s * SEG;
        uint64_t seg_hi = seg_lo + SEG - 1;
        if (seg_hi > hi) seg_hi = hi;
        uint64_t seg_sz = seg_hi - seg_lo + 1;

        std::vector<bool> is_comp(seg_sz, false);
        for (int i = 0; i < n_small; i++) {
            uint64_t p = (uint64_t)small[i];
            if (p * p > seg_hi) break;
            uint64_t r = seg_lo % p;
            uint64_t start = r ? (seg_lo + p - r) : seg_lo;
            for (uint64_t m = start; m <= seg_hi; m += p)
                is_comp[(size_t)(m - seg_lo)] = true;
        }
        for (uint64_t i = 0; i < seg_sz; i++)
            if (!is_comp[i]) seg_out[s].push_back((uint32_t)(seg_lo + i));
    }

    free(small);
    int64_t total = 0;
    for (auto &v : seg_out) total += (int64_t)v.size();
    uint32_t *arr = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)total);
    int64_t j = 0;
    for (auto &v : seg_out) {
        memcpy(arr + j, v.data(), v.size() * sizeof(uint32_t));
        j += (int64_t)v.size();
    }
    *out_count = total;
    return arr;
}

// ---------------------------------------------------------------------------
// GPU constant memory: current blinded value y
// ---------------------------------------------------------------------------
__constant__ uint64_t d_y_limbs[MAX_LIMBS];
__constant__ int       d_nlimbs;

// ---------------------------------------------------------------------------
// k32 trial-division kernel (identical to alg2_k32_cuda)
// ---------------------------------------------------------------------------
__global__ void k32_trial_divide_kernel(
    const uint32_t * __restrict__ primes, int64_t nprimes,
    uint64_t *out_factors, int *out_count)
{
    int64_t idx    = (int64_t)blockIdx.x  * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x   * blockDim.x;
    int nl         = d_nlimbs;

    for (int64_t i = idx; i < nprimes; i += stride) {
        uint64_t p   = (uint64_t)primes[i];
        uint64_t rem = mod_big_by_u32(d_y_limbs, nl, p);
        if (rem == 0) {
            int slot = atomicAdd(out_count, 1);
            if (slot < MAX_GPU_FACTORS) out_factors[slot] = p;
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    const char *challenge_file = nullptr;
    const char *modulo_file    = nullptr;
    const char *data_dir       = nullptr;
    int size_bits              = 0;
    unsigned long exponent     = 3;
    const char *primes_path    = "data/k32_primes.bin";
    const char *json_path      = nullptr;
    gpu_method_t method        = GPU_METHOD_TABLE;
    int n_challenges           = 10;
    int n_gpus                 = 1;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--data")         && i+1<argc) data_dir     = argv[++i];
        else if (!strcmp(argv[i], "--size")         && i+1<argc) size_bits    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-challenges") && i+1<argc) n_challenges = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-runs")       && i+1<argc) n_challenges = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-gpus")       && i+1<argc) n_gpus       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exponent")     && i+1<argc) exponent     = (unsigned long)atol(argv[++i]);
        else if (!strcmp(argv[i], "--primes")       && i+1<argc) primes_path  = argv[++i];
        else if (!strcmp(argv[i], "--json")         && i+1<argc) json_path    = argv[++i];
        else if (!strcmp(argv[i], "--method")       && i+1<argc) method       = gpu_parse_method(argv[++i]);
        else if (argv[i][0] != '-') {
            if (!challenge_file) challenge_file = argv[i];
            else if (!modulo_file) modulo_file  = argv[i];
        }
    }

    // Clamp n_gpus to available devices
    {
        int avail = 1;
        cudaGetDeviceCount(&avail);
        if (n_gpus < 1) n_gpus = 1;
        if (n_gpus > avail) {
            fprintf(stderr, "Warning: only %d GPU(s) available, using %d\n", avail, avail);
            n_gpus = avail;
        }
    }

    // Derive paths from --data/--size when given
    static char _modu_buf[512], _chal_buf[512];
    if (data_dir && size_bits > 0) {
        snprintf(_modu_buf, sizeof(_modu_buf), "%s/modulo_alg3_%d.txt",      data_dir, size_bits);
        snprintf(_chal_buf, sizeof(_chal_buf), "%s/challenge_0_alg3_%d.txt", data_dir, size_bits);
        modulo_file    = _modu_buf;
        challenge_file = _chal_buf;
    } else if (!challenge_file || !modulo_file) {
        fprintf(stderr,
            "Usage: %s [<challenge> <modulo>] [--data DIR --size N] "
            "[--method {powm|sieve|table}] [--exponent E] [--primes FILE] "
            "[--n-challenges N] [--n-gpus N] [--json FILE]\n", argv[0]);
        return 1;
    }

    // ---- Load challenge + modulo ----
    mpz_t challenge, modulo, challenge_mod;
    mpz_inits(challenge, modulo, challenge_mod, NULL);
    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);
    mpz_mod(challenge_mod, challenge, modulo);

    size_t challenge_bits = mpz_sizeinbase(modulo, 2);
    printf("Challenge: %zu bits | Modulus: %zu bits | Exponent: %lu | Method: %s | GPUs: %d\n",
           mpz_sizeinbase(challenge, 2), challenge_bits, exponent, gpu_method_name(method), n_gpus);

    printf("Measuring powm baseline...\n");
    double powm_baseline_ms = gpu_measure_powm_baseline(challenge_mod, modulo, exponent);
    printf("  powm_baseline_ms = %.4f ms\n", powm_baseline_ms);

    // ---- powm method: write JSON and exit ----
    if (method == GPU_METHOD_POWM) {
        std::vector<double> empty;
        gpu_write_json_warm(json_path, "alg3", method, exponent, 32, challenge_bits,
                            powm_baseline_ms, 0.0, 0.0, empty, empty, 0, n_gpus);
        mpz_clears(challenge, modulo, challenge_mod, NULL);
        return 0;
    }

    // ---- Load or generate k32 prime table ----
    int64_t nprimes_i64 = 0;
    uint32_t *h_primes = nullptr;
    if (method == GPU_METHOD_TABLE) {
        uint64_t nprimes_u64 = 0;
        h_primes = kf_load_prime_table(&nprimes_u64, primes_path);
        if (!h_primes) {
            fprintf(stderr, "Failed to load prime table from %s\n", primes_path); return 1;
        }
        nprimes_i64 = (int64_t)nprimes_u64;
        printf("Prime table: %ld primes from %s\n", (long)nprimes_i64, primes_path);
    } else {
        printf("Sieving k32 primes (65537, 2^32] (slow; prefer --method table)...\n");
        h_primes = sieve_primes_u32(65537, 0xFFFFFFFFULL, &nprimes_i64);
        printf("Sieved %ld k32 primes.\n", (long)nprimes_i64);
    }
    uint64_t nprimes = (uint64_t)nprimes_i64;

    int nlimbs = (int)challenge_bits / 64 + 2;
    if (nlimbs > MAX_LIMBS) {
        fprintf(stderr, "Error: modulus too large for MAX_LIMBS=%d\n", MAX_LIMBS); return 1;
    }

    // ---- Per-GPU setup: slice prime table, allocate device memory ----
    printf("Setting up %d GPU(s), %" PRIu64 " primes total...\n", n_gpus, nprimes);
    std::vector<GpuDev> devs((size_t)n_gpus);
    int64_t base_slice = (int64_t)nprimes / n_gpus;
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        int64_t start = (int64_t)g * base_slice;
        int64_t count = (g == n_gpus - 1) ? ((int64_t)nprimes - start) : base_slice;
        devs[g].n_slice = count;
        CUDA_CHECK(cudaMalloc(&devs[g].d_primes, (size_t)count * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(devs[g].d_primes, h_primes + start,
                              (size_t)count * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&devs[g].d_out_factors, MAX_GPU_FACTORS * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&devs[g].d_out_count,   sizeof(int)));
        CUDA_CHECK(cudaEventCreate(&devs[g].ev0));
        CUDA_CHECK(cudaEventCreate(&devs[g].ev1));
        CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &nlimbs, sizeof(int)));
        printf("  GPU %d: %ld primes (offset %ld)\n", g, (long)count, (long)start);
    }

    // ---- Blinding table (255 entries, matching CPU BLIND_TABLE_SIZE) ----
    const int BLIND_TABLE_SIZE = 255;
    blinding_pair_t *btable = kf_blinding_table_alloc(BLIND_TABLE_SIZE, modulo, exponent);
    if (!btable) { fprintf(stderr, "blinding table alloc failed\n"); return 1; }

    // ---- cbrt cache ----
    cbrt_cache_t cbrt_cache;
    kf_cbrt_cache_init(&cbrt_cache);
    bool cache_dirty = false;
    char cache_path[512];
    snprintf(cache_path, sizeof(cache_path), "%s.cbrt_alg3", modulo_file);
    kf_cbrt_cache_load(&cbrt_cache, cache_path);

    // ---- Shared host output buffer ----
    uint64_t *h_out_factors = (uint64_t *)malloc(MAX_GPU_FACTORS * sizeof(uint64_t));

    const int NBLOCKS = 1024;

    // Upload y to all GPUs, launch kernels in parallel, return wall-clock max.
    auto launch_all = [&](const mpz_t y) -> float {
        uint64_t y_limbs[MAX_LIMBS] = {};
        size_t cnt = 0;
        mpz_export(y_limbs, &cnt, -1, 8, 0, 0, y);
        for (int g = 0; g < n_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpyToSymbol(d_y_limbs, y_limbs, MAX_LIMBS * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemset(devs[g].d_out_count, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(devs[g].d_out_factors, 0, MAX_GPU_FACTORS * sizeof(uint64_t)));
            CUDA_CHECK(cudaEventRecord(devs[g].ev0));
            k32_trial_divide_kernel<<<NBLOCKS, 256>>>(devs[g].d_primes, devs[g].n_slice,
                                                      devs[g].d_out_factors, devs[g].d_out_count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(devs[g].ev1));
        }
        float km = 0;
        for (int g = 0; g < n_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaEventSynchronize(devs[g].ev1));
            float t = 0;
            CUDA_CHECK(cudaEventElapsedTime(&t, devs[g].ev0, devs[g].ev1));
            if (t > km) km = t;
        }
        return km;
    };

    // Download factor lists from all devices; build factor list + divide cofactor.
    auto collect_factors = [&](kfactor_list_t *fl, mpz_t cof, const mpz_t y) {
        for (int g = 0; g < n_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            int h_count = 0;
            CUDA_CHECK(cudaMemcpy(&h_count, devs[g].d_out_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            if (h_count <= 0 || h_count > MAX_GPU_FACTORS) continue;
            CUDA_CHECK(cudaMemcpy(h_out_factors, devs[g].d_out_factors,
                                  (size_t)h_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            for (int j = 0; j < h_count; j++) {
                unsigned long p = (unsigned long)h_out_factors[j];
                int e = count_exponent(y, p);
                if (e > 0) {
                    kfl_append_ul(fl, p, e);
                    if (cof) for (int ei = 0; ei < e; ei++) mpz_divexact_ui(cof, cof, p);
                }
            }
        }
    };

    // ---- Find smooth/almost-smooth attempt ----
    mpz_t y_smooth, cofactor_saved, q0_saved, result;
    mpz_inits(y_smooth, cofactor_saved, q0_saved, result, NULL);
    kfactor_list_t solved_factors;
    kfl_init(&solved_factors);
    int solved_attempt = -1;
    bool has_q0 = false;

    for (int attempt = 0; attempt <= BLIND_TABLE_SIZE && solved_attempt < 0; attempt++) {
        mpz_t y; mpz_init(y);
        // Attempt 0: factor the raw challenge (smooth preimage x, possibly >= M).
        // mpz_mod would destroy smoothness when x >= M — matches CPU "Fix E" 2026-06-11.
        if (attempt == 0) mpz_set(y, challenge);
        else { int b = (attempt - 1) % BLIND_TABLE_SIZE; mpz_mul(y, challenge_mod, btable[b].r_blind); mpz_mod(y, y, modulo); }

        float km = launch_all(y);

        kfl_init(&solved_factors);
        mpz_t cofactor; mpz_init_set(cofactor, y);
        collect_factors(&solved_factors, cofactor, y);

        bool accepted = false;
        if (mpz_cmp_ui(cofactor, 1) == 0) {
            accepted = true; has_q0 = false;
            printf("Attempt %d: B-smooth (%d factors, kernel %.2f ms)\n", attempt, solved_factors.count, km);
        } else if (mpz_sizeinbase(cofactor, 2) <= 65 && mpz_probab_prime_p(cofactor, 25) > 0) {
            accepted = true; has_q0 = true;
            mpz_set(q0_saved, cofactor);
            printf("Attempt %d: almost-smooth (%zu bit cofactor, kernel %.2f ms)\n",
                   attempt, mpz_sizeinbase(cofactor, 2), km);
        }

        if (accepted) {
            mpz_set(y_smooth, y);
            solved_attempt = attempt;
        } else {
            kfl_free(&solved_factors);
        }
        mpz_clears(y, cofactor, NULL);
    }

    if (solved_attempt < 0) {
        fprintf(stderr, "Warning: no smooth/almost-smooth attempt found\n");
        std::vector<double> empty;
        gpu_write_json_warm(json_path, "alg3", method, exponent, 32, challenge_bits,
                            0.0, 0.0, 0.0, empty, empty, 0, n_gpus);
        goto cleanup;
    }

    // ---- Cold run: kernel + reconstruction (warms cbrt cache) ----
    {
        float cold_km = launch_all(y_smooth);

        auto tc0 = std::chrono::high_resolution_clock::now();
        kf_compute_cbrt_from_factors_omp(result, &solved_factors, modulo, &cbrt_cache, &cache_dirty, exponent);
        if (has_q0) {
            mpz_t cbrt_q0; mpz_init(cbrt_q0);
            compute_eth_root_mod_prime(cbrt_q0, q0_saved, modulo, exponent);
            mpz_mul(result, result, cbrt_q0);
            mpz_mod(result, result, modulo);
            mpz_clear(cbrt_q0);
        }
        if (solved_attempt > 0) {
            int b = (solved_attempt - 1) % BLIND_TABLE_SIZE;
            mpz_mul(result, result, btable[b].r_inv);
            mpz_mod(result, result, modulo);
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        double cold_factor_ms = cold_km;
        double cold_recon_ms  = std::chrono::duration<double, std::milli>(tc1 - tc0).count();
        verify_eth_root(result, challenge_mod, modulo, exponent);
        printf("Cold run: factor_ms=%.2f, recon_ms=%.4f\n", cold_factor_ms, cold_recon_ms);

        // ---- Warm runs: one per challenge file (different y each time) ----
        std::vector<double> factor_ms_v, recon_ms_v;
        factor_ms_v.reserve(n_challenges - 1);
        recon_ms_v.reserve(n_challenges - 1);
        GpuHitStats hit_stats;
        GpuTransferStats xfer;
        xfer.table_mb = (double)nprimes * 4.0 / (1024.0 * 1024.0);

        for (int ci = 1; ci < n_challenges; ci++) {
            mpz_t y_w; mpz_init(y_w);
            if (data_dir && size_bits > 0) {
                // Load raw challenge (no mpz_mod) — same as discovery attempt 0.
                char _cwpath[512];
                snprintf(_cwpath, sizeof(_cwpath), "%s/challenge_%d_alg3_%d.txt",
                         data_dir, ci, size_bits);
                FILE *_cwf = fopen(_cwpath, "rb");
                if (!_cwf) { mpz_clear(y_w); continue; }
                fclose(_cwf);
                read_mpz_from_file(y_w, _cwpath);
            } else {
                mpz_set(y_w, y_smooth);  // single-challenge compat
            }

            auto t_launch0 = std::chrono::high_resolution_clock::now();
            float km = launch_all(y_w);
            auto t_launch1 = std::chrono::high_resolution_clock::now();
            double h2d_approx = std::max(0.0,
                std::chrono::duration<double, std::milli>(t_launch1 - t_launch0).count() - (double)km);

            kfactor_list_t fl_w; kfl_init(&fl_w);
            mpz_t cof_w; mpz_init_set(cof_w, y_w);
            auto t_d2h0 = std::chrono::high_resolution_clock::now();
            collect_factors(&fl_w, cof_w, y_w);
            auto t_d2h1 = std::chrono::high_resolution_clock::now();
            xfer.h2d_ms.push_back(h2d_approx);
            xfer.d2h_ms.push_back(std::chrono::duration<double, std::milli>(t_d2h1 - t_d2h0).count());

            auto tr0 = std::chrono::high_resolution_clock::now();
            kf_compute_cbrt_from_factors_omp(result, &fl_w, modulo, &cbrt_cache, &cache_dirty, exponent);
            // Type 1: prime cofactor (alg3 challenge always has this structure)
            if (mpz_cmp_ui(cof_w, 1) != 0 && mpz_sizeinbase(cof_w, 2) <= 65
                    && mpz_probab_prime_p(cof_w, 25) > 0) {
                mpz_t cbrt_q_w; mpz_init(cbrt_q_w);
                compute_eth_root_mod_prime(cbrt_q_w, cof_w, modulo, exponent);
                mpz_mul(result, result, cbrt_q_w);
                mpz_mod(result, result, modulo);
                mpz_clear(cbrt_q_w);
            }
            // attempt 0 is unblinded: no r_inv needed
            auto tr1 = std::chrono::high_resolution_clock::now();

            // Track hit type: alg3 challenges always have one prime cofactor (type-1)
            {
                int h_type = 0, h_bits = 0, h_med = 0;
                if (mpz_cmp_ui(cof_w, 1) != 0 && mpz_sizeinbase(cof_w, 2) <= 65
                        && mpz_probab_prime_p(cof_w, 25) > 0) {
                    h_type = 1;
                    h_bits = (int)mpz_sizeinbase(cof_w, 2);
                    h_med  = 1;
                }
                hit_stats.hit_type.push_back(h_type);
                hit_stats.cofactor_bits.push_back(h_bits);
                hit_stats.medium_factor_count.push_back(h_med);
            }

            factor_ms_v.push_back((double)km);
            recon_ms_v.push_back(std::chrono::duration<double, std::milli>(tr1 - tr0).count());
            kfl_free(&fl_w);
            mpz_clears(cof_w, y_w, NULL);
        }

        gpu_print_benchmark("alg3", (int)factor_ms_v.size(), powm_baseline_ms,
                            _gpu_vmean(factor_ms_v), _gpu_vmean(recon_ms_v));
        gpu_write_json_warm(json_path, "alg3", method, exponent, 32, challenge_bits,
                            0.0, cold_factor_ms, cold_recon_ms, factor_ms_v, recon_ms_v, 1,
                            n_gpus, hit_stats, xfer);
    }

cleanup:
    if (cache_dirty) kf_cbrt_cache_save(&cbrt_cache, cache_path);
    kfl_free(&solved_factors);
    mpz_clears(y_smooth, cofactor_saved, q0_saved, result, NULL);
    mpz_clears(challenge, challenge_mod, modulo, NULL);
    free(h_out_factors);
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        cudaFree(devs[g].d_primes);
        cudaFree(devs[g].d_out_factors);
        cudaFree(devs[g].d_out_count);
        cudaEventDestroy(devs[g].ev0);
        cudaEventDestroy(devs[g].ev1);
    }
    kf_blinding_table_free(btable, BLIND_TABLE_SIZE);
    kf_cbrt_cache_clear(&cbrt_cache);
    free(h_primes);

    return solved_attempt >= 0 ? 0 : 2;
}
// Edit: 2026-07-11
