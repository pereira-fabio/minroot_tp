// =============================================================================
// alg2_k16_cuda.cu — GPU-accelerated Algorithm 2, k16 variant
//
// Algorithm 2 (Biryukov et al. CRYPTO 2024, §2.1), smoothness bound B = 2^16.
// Mirrors the CPU alg2_k16 structure: --method {powm|sieve|table}, cold + N
// warm runs, same JSON format (warm_avg / warm_stats / warm_runs).
//
// GPU strategy: 6542 primes ≤ 65536 fit in CUDA constant memory (26 KB).
// A batch of 256 blinded candidates is tested in ONE kernel launch:
//   - Block b tests all 6542 primes against candidate y_b (256 blocks total)
//   - Slot 0 is always the raw unblinded challenge (always smooth)
//
// Methods:
//   sieve — generate 6542 k16 primes in-memory; GPU batch trial division
//   table — same computation as sieve (primes generated in-memory, no file);
//           distinct label for JSON parity with CPU alg2_k16
//   powm  — measure sequential mpz_powm baseline only; no GPU kernel
//
// Warm runs re-launch the same batch kernel (d_y_batch stays on GPU) for N
// timing samples.  factor_ms = kernel_ms / batch_size (amortized per candidate).
//
// CLI:
//   ./alg2_k16_cuda [<challenge_file> <modulo_file>] [options]
//     --data DIR --size N     use challenge directory (replaces positional args)
//     --method {powm|sieve|table}  benchmark method (default: sieve)
//     --exponent E            root exponent (default: 3)
//     --attempts N            batch size = blinded candidates per kernel (default: 256)
//     --n-challenges N        challenge files to run (default: 10)
//     --json FILE             write JSON output to FILE
// =============================================================================

#include "gpu_common.h"

#include <gmp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include "../common/common.h"
#include "../common/factorization.h"

// ---------------------------------------------------------------------------
// Constant memory: all 6,542 k16 primes + current batch of blinded values
// ---------------------------------------------------------------------------

#define K16_MAX_PRIMES 6600
__constant__ uint32_t d_k16_primes[K16_MAX_PRIMES];
__constant__ int d_n_k16_primes;
__constant__ int d_nlimbs;

// ---------------------------------------------------------------------------
// GPU kernel: trial-divide a batch of blinded values by all k16 primes.
// ---------------------------------------------------------------------------

__global__ void k16_trial_divide_kernel(
    const uint64_t *__restrict__ y_batch,
    int batch_size,
    uint64_t *out_factors,
    int *out_counts)
{
    int b = (int)blockIdx.x;
    if (b >= batch_size)
        return;

    const uint64_t *y = y_batch + (int64_t)b * MAX_LIMBS;
    int nprime = d_n_k16_primes;
    int nlimbs = d_nlimbs;

    for (int i = (int)threadIdx.x; i < nprime; i += (int)blockDim.x)
    {
        uint64_t p = (uint64_t)d_k16_primes[i];
        uint64_t rem = mod_big_by_u32(y, nlimbs, p);
        if (rem == 0)
        {
            int slot = atomicAdd(&out_counts[b], 1);
            if (slot < MAX_GPU_FACTORS)
                out_factors[(int64_t)b * MAX_GPU_FACTORS + slot] = p;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-device state for the multi-GPU candidate-batch split. Each device holds
// its own slice of the 256-candidate batch and the FULL prime table in its own
// constant memory (26 KB — nothing to tile, unlike k32's 813 MB table, so the
// batch rather than the primes is what we split across devices).
// ---------------------------------------------------------------------------
struct K16Dev
{
    int cb_lo;               // first global candidate index this device owns
    int n_cand;              // number of candidates this device owns
    uint64_t *d_y_batch;     // n_cand * MAX_LIMBS
    uint64_t *d_out_factors; // n_cand * MAX_GPU_FACTORS
    int *d_out_counts;       // n_cand
    cudaEvent_t ev0, ev1;
};

// ---------------------------------------------------------------------------
// Host utilities
// ---------------------------------------------------------------------------

static int export_limbs(const mpz_t val, uint64_t *out, int max_limbs)
{
    memset(out, 0, sizeof(uint64_t) * max_limbs);
    size_t cnt = 0;
    mpz_export(out, &cnt, -1, sizeof(uint64_t), 0, 0, val);
    if ((int)cnt > max_limbs)
    {
        fprintf(stderr, "Error: number too large for MAX_LIMBS=%d (%zu needed)\n", max_limbs, cnt);
        exit(1);
    }
    return (int)cnt;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [<challenge_file> <modulo_file>] [options]\n"
            "  --data DIR --size N    use challenge directory (replaces positional args)\n"
            "  --method {powm|sieve|table}  benchmark method (default: sieve)\n"
            "  --exponent E           root exponent (default: 3)\n"
            "  --attempts N           batch size — blinded candidates per kernel (default: 256)\n"
            "  --n-challenges N       challenge files to run (default: 10)\n"
            "  --n-gpus N             split the candidate batch across N devices (default: 1)\n"
            "  --json FILE            write JSON output\n",
            prog);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    const char *challenge_file = nullptr;
    const char *modulo_file = nullptr;
    const char *data_dir = nullptr;
    // Suffix on the modulus filename, e.g. "_e5" for the exponent-5 moduli that
    // challenge_gen_e5.slurm generates with gcd(5, M-1) = 1. Challenge files are
    // exponent-agnostic and shared, so only the modulus changes. Mirrors
    // --modulo-suffix on the MPI solvers.
    const char *modulo_suffix = "";
    int size_bits = 0;
    unsigned long exponent = 3;
    int batch_size = 256;
    const char *json_path = nullptr;
    gpu_method_t method = GPU_METHOD_SIEVE;
    int n_challenges = 10;
    int n_gpus = 1;

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
        else if (!strcmp(argv[i], "--exponent") && i + 1 < argc)
            exponent = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--attempts") && i + 1 < argc)
            batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_path = argv[++i];
        else if (!strcmp(argv[i], "--method") && i + 1 < argc)
            method = gpu_parse_method(argv[++i]);
        else if (!strcmp(argv[i], "--n-gpus") && i + 1 < argc)
            n_gpus = atoi(argv[++i]);
        else if (argv[i][0] != '-')
        {
            if (!challenge_file)
                challenge_file = argv[i];
            else if (!modulo_file)
                modulo_file = argv[i];
        }
    }
    if (batch_size < 1 || batch_size > 4096)
    {
        fprintf(stderr, "Error: --attempts must be in [1, 4096]\n");
        return 1;
    }

    // Clamp n_gpus to the number of visible devices (mirrors alg2_k32_cuda.cu).
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

    // Derive paths from --data/--size when given (overrides positional args)
    static char _modu_buf[512], _chal_buf[512];
    if (data_dir && size_bits > 0)
    {
        snprintf(_modu_buf, sizeof(_modu_buf), "%s/modulo_16_%d%s.txt", data_dir, size_bits, modulo_suffix);
        snprintf(_chal_buf, sizeof(_chal_buf), "%s/challenge_0_16_%d.txt", data_dir, size_bits);
        modulo_file = _modu_buf;
        challenge_file = _chal_buf;
    }
    else if (!challenge_file || !modulo_file)
    {
        print_usage(argv[0]);
        return 1;
    }

    // ---- Load challenge_0 + modulus ----
    mpz_t challenge, modulo, challenge_mod;
    mpz_inits(challenge, modulo, challenge_mod, nullptr);
    read_mpz_from_file(challenge, challenge_file);
    read_mpz_from_file(modulo, modulo_file);
    mpz_mod(challenge_mod, challenge, modulo);

    printf("Challenge: %zu bits | Modulus: %zu bits | Exponent: %lu | Method: %s\n",
           mpz_sizeinbase(challenge, 2), mpz_sizeinbase(modulo, 2), exponent, gpu_method_name(method));

    printf("Measuring powm baseline...\n");
    double powm_baseline_ms = gpu_measure_powm_baseline(challenge_mod, modulo, exponent);
    printf("  powm_baseline_ms = %.4f ms\n", powm_baseline_ms);

    size_t challenge_bits = mpz_sizeinbase(modulo, 2);

    // ---- powm method: write JSON and exit ----
    if (method == GPU_METHOD_POWM)
    {
        std::vector<double> empty;
        gpu_write_json_warm(json_path, "k16", method, exponent, 16, challenge_bits,
                            powm_baseline_ms, 0.0, 0.0, empty, empty, 0);
        mpz_clears(challenge, modulo, challenge_mod, nullptr);
        return 0;
    }

    int nlimbs = (int)mpz_sizeinbase(modulo, 2) / 64 + 2;
    if (nlimbs > MAX_LIMBS)
    {
        fprintf(stderr, "Error: modulus too large for MAX_LIMBS=%d\n", MAX_LIMBS);
        return 1;
    }

    // ---- Generate k16 primes (all primes ≤ 65536) ----
    int n_primes = 0;
    unsigned long *primes_ul = generate_primes_up_to(65536, &n_primes);
    printf("k16 primes: %d primes ≤ 65536\n", n_primes);

    if (n_primes > K16_MAX_PRIMES)
    {
        fprintf(stderr, "Error: n_primes=%d exceeds K16_MAX_PRIMES=%d\n", n_primes, K16_MAX_PRIMES);
        return 1;
    }
    uint32_t *primes32 = (uint32_t *)malloc(n_primes * sizeof(uint32_t));
    for (int i = 0; i < n_primes; i++)
        primes32[i] = (uint32_t)primes_ul[i];

    // Prime table + nlimbs are uploaded to EACH device's constant memory in the
    // per-device setup loop below (the batch is split across devices, not the primes).

    // ---- cbrt cache ----
    cbrt_cache_t cache;
    kf_cbrt_cache_init(&cache);
    bool cache_dirty = false;
    // Room for a 512-byte modulo path plus the ".cbrt_algN" suffix. Sized to
    // 512 a long path truncates, and two moduli differing only past the cut
    // would share one cache file. Matches KSC_CBRT_PATH_MAX on the CPU side,
    // which cannot be included here (it lives in the MPI header).
    char cache_path[600];
    snprintf(cache_path, sizeof(cache_path), "%s.cbrt", modulo_file);
    kf_cbrt_cache_load(&cache, cache_path);

    // ---- Blinding table ----
    blinding_pair_t *btable = kf_blinding_table_alloc(batch_size, modulo, exponent);
    printf("Blinding table ready (%d entries).\n", batch_size);

    // ---- Shared host buffers (full batch; filled by gathering all devices) ----
    uint64_t *h_y_batch;
    CUDA_CHECK(cudaMallocHost(&h_y_batch, (size_t)batch_size * MAX_LIMBS * sizeof(uint64_t)));
    uint64_t *h_out_factors;
    CUDA_CHECK(cudaMallocHost(&h_out_factors, (size_t)batch_size * MAX_GPU_FACTORS * sizeof(uint64_t)));
    int *h_out_counts = (int *)calloc(batch_size, sizeof(int));

    // ---- Per-device setup: split the candidate batch across n_gpus devices ----
    std::vector<K16Dev> devs((size_t)n_gpus);
    {
        int base = batch_size / n_gpus;
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            int cb_lo = g * base;
            int n_cand = (g == n_gpus - 1) ? (batch_size - cb_lo) : base;
            devs[g].cb_lo = cb_lo;
            devs[g].n_cand = n_cand;
            // Full prime table + nlimbs into THIS device's constant memory.
            CUDA_CHECK(cudaMemcpyToSymbol(d_k16_primes, primes32, n_primes * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_n_k16_primes, &n_primes, sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &nlimbs, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&devs[g].d_y_batch, (size_t)n_cand * MAX_LIMBS * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&devs[g].d_out_factors, (size_t)n_cand * MAX_GPU_FACTORS * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&devs[g].d_out_counts, (size_t)n_cand * sizeof(int)));
            CUDA_CHECK(cudaEventCreate(&devs[g].ev0));
            CUDA_CHECK(cudaEventCreate(&devs[g].ev1));
            printf("  GPU %d: candidates [%d, %d) (%d of %d)\n",
                   g, cb_lo, cb_lo + n_cand, n_cand, batch_size);
        }
    }

    // ---- Build batch of blinded candidates (slot 0 = unblinded) ----
    memset(h_y_batch, 0, (size_t)batch_size * MAX_LIMBS * sizeof(uint64_t));
    {
        mpz_t y_tmp;
        mpz_init(y_tmp);
        for (int b = 0; b < batch_size; b++)
        {
            if (b == 0)
            {
                export_limbs(challenge_mod, h_y_batch + (size_t)b * MAX_LIMBS, MAX_LIMBS);
            }
            else
            {
                mpz_mul(y_tmp, challenge_mod, btable[b].r_blind);
                mpz_mod(y_tmp, y_tmp, modulo);
                export_limbs(y_tmp, h_y_batch + (size_t)b * MAX_LIMBS, MAX_LIMBS);
            }
        }
        mpz_clear(y_tmp);
    }
    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpy(devs[g].d_y_batch,
                              h_y_batch + (size_t)devs[g].cb_lo * MAX_LIMBS,
                              (size_t)devs[g].n_cand * MAX_LIMBS * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
    }

    // Helper: launch the batch kernel on every device (each over its candidate
    // slice, against the full prime table), then gather all device outputs into
    // the shared host buffers. Returns the wall-clock kernel time (max across
    // devices = effective per-launch cost). d_y_batch stays resident on each GPU.
    auto launch_batch = [&]() -> float
    {
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemset(devs[g].d_out_factors, 0,
                                  (size_t)devs[g].n_cand * MAX_GPU_FACTORS * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemset(devs[g].d_out_counts, 0, (size_t)devs[g].n_cand * sizeof(int)));
            CUDA_CHECK(cudaEventRecord(devs[g].ev0));
            k16_trial_divide_kernel<<<devs[g].n_cand, 256>>>(devs[g].d_y_batch, devs[g].n_cand,
                                                             devs[g].d_out_factors, devs[g].d_out_counts);
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
        // Gather each device's candidate slice into the shared host buffers.
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMemcpy(h_out_counts + devs[g].cb_lo, devs[g].d_out_counts,
                                  (size_t)devs[g].n_cand * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_out_factors + (size_t)devs[g].cb_lo * MAX_GPU_FACTORS,
                                  devs[g].d_out_factors,
                                  (size_t)devs[g].n_cand * MAX_GPU_FACTORS * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost));
        }
        return km;
    };

    // ---- Cold run: find smooth slot + reconstruct (warms cbrt cache) ----
    // launch_batch() already gathered all devices' outputs into h_out_* buffers.
    float cold_kernel_ms = launch_batch();

    printf("Cold kernel: %.3f ms for %d candidates × %d primes (%d GPU(s))\n",
           cold_kernel_ms, batch_size, n_primes, n_gpus);

    // Find smooth slot (always slot 0 since challenge file stores smooth y)
    int solved_b = -1;
    kfactor_list_t solved_factors;
    kfl_init(&solved_factors);
    mpz_t y_mpz, cofactor, result;
    mpz_inits(y_mpz, cofactor, result, nullptr);

    for (int b = 0; b < batch_size && solved_b < 0; b++)
    {
        int gpu_count = h_out_counts[b];
        if (gpu_count <= 0)
            continue;
        if (gpu_count > MAX_GPU_FACTORS)
        {
            fprintf(stderr, "Warning: k16 factor overflow slot %d (%d > %d) — "
                            "processing first %d\n",
                    b, gpu_count, MAX_GPU_FACTORS, MAX_GPU_FACTORS);
            gpu_count = MAX_GPU_FACTORS;
        }

        mpz_import(y_mpz, MAX_LIMBS, -1, sizeof(uint64_t), 0, 0,
                   h_y_batch + (size_t)b * MAX_LIMBS);

        kfl_init(&solved_factors);
        const uint64_t *fp = h_out_factors + (size_t)b * MAX_GPU_FACTORS;
        mpz_set(cofactor, y_mpz);
        for (int j = 0; j < gpu_count; j++)
        {
            unsigned long p = (unsigned long)fp[j];
            int e = count_exponent(y_mpz, p);
            if (e > 0)
            {
                kfl_append_ul(&solved_factors, p, e);
                for (int k = 0; k < e; k++)
                    mpz_divexact_ui(cofactor, cofactor, p);
            }
        }
        if (mpz_cmp_ui(cofactor, 1) == 0)
        {
            printf("Slot %d: k16-smooth (%d factors)\n", b, solved_factors.count);
            solved_b = b;
        }
        else
        {
            kfl_free(&solved_factors);
        }
    }

    double cold_factor_ms = cold_kernel_ms / batch_size; // amortized per candidate
    double cold_recon_ms = 0.0;

    if (solved_b < 0)
    {
        printf("No k16-smooth slot found in cold batch. Challenge may not be k16-smooth.\n");
        std::vector<double> empty;
        gpu_write_json_warm(json_path, "k16", method, exponent, 16, challenge_bits,
                            0.0, cold_factor_ms, 0.0, empty, empty, 0, n_gpus);
        goto cleanup;
    }

    {
        auto tc0 = std::chrono::high_resolution_clock::now();
        kf_compute_cbrt_from_factors_omp(result, &solved_factors, modulo, &cache, &cache_dirty, exponent);
        if (solved_b > 0)
        {
            mpz_mul(result, result, btable[solved_b].r_inv);
            mpz_mod(result, result, modulo);
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        cold_recon_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();
        verify_eth_root(result, challenge_mod, modulo, exponent);
        printf("Cold run: factor_ms=%.4f (amortized), recon_ms=%.4f\n",
               cold_factor_ms, cold_recon_ms);
    }

    // ---- Warm runs: one per challenge file (slot 0 updated with new y each time) ----
    {
        std::vector<double> factor_ms_v, recon_ms_v;
        factor_ms_v.reserve(n_challenges - 1);
        recon_ms_v.reserve(n_challenges - 1);
        GpuHitStats hit_stats;
        GpuTransferStats xfer;
        xfer.table_mb = (double)n_primes * 4.0 / (1024.0 * 1024.0);

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
                mpz_t y_w;
                mpz_init(y_w);
                if (data_dir && size_bits > 0)
                {
                    if (!gpu_load_challenge(y_w, data_dir, ci, "16", size_bits, modulo))
                    {
                        mpz_clear(y_w);
                        continue;
                    }
                }
                else
                {
                    mpz_set(y_w, challenge_mod); // single-challenge compat
                }
                // Update slot 0 only (challenge_i is smooth at slot 0 by generator)
                export_limbs(y_w, h_y_batch + 0 * MAX_LIMBS, MAX_LIMBS);
                auto t_h2d0 = std::chrono::high_resolution_clock::now();
                // Global slot 0 lives on device 0 (local slot 0). Only slot 0 is the
                // real challenge; the other slots stay resident for timing amortization.
                CUDA_CHECK(cudaSetDevice(0));
                CUDA_CHECK(cudaMemcpy(devs[0].d_y_batch, h_y_batch,
                                      MAX_LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice));
                auto t_h2d1 = std::chrono::high_resolution_clock::now();

                float km = launch_batch();

                // Read slot 0 result (always smooth at attempt 0). launch_batch()
                // already gathered device 0's slot-0 output into h_out_*.
                auto t_d2h0 = std::chrono::high_resolution_clock::now();
                int gpu_count_w = h_out_counts[0];
                if (gpu_count_w > MAX_GPU_FACTORS)
                {
                    fprintf(stderr, "Warning: k16 warm factor overflow (%d > %d) — "
                                    "processing first %d\n",
                            gpu_count_w, MAX_GPU_FACTORS, MAX_GPU_FACTORS);
                    gpu_count_w = MAX_GPU_FACTORS;
                }
                kfactor_list_t fl_w;
                kfl_init(&fl_w);
                if (gpu_count_w > 0)
                {
                    mpz_t cof_w;
                    mpz_init_set(cof_w, y_w);
                    for (int j = 0; j < gpu_count_w; j++)
                    {
                        unsigned long p = (unsigned long)h_out_factors[j];
                        int e = count_exponent(y_w, p);
                        if (e > 0)
                        {
                            kfl_append_ul(&fl_w, p, e);
                            for (int ek = 0; ek < e; ek++)
                                mpz_divexact_ui(cof_w, cof_w, p);
                        }
                    }
                    mpz_clear(cof_w);
                }
                auto t_d2h1 = std::chrono::high_resolution_clock::now();

                auto tr0 = std::chrono::high_resolution_clock::now();
                kf_compute_cbrt_from_factors_omp(result, &fl_w, modulo, &cache, &cache_dirty, exponent);
                // slot 0 = unblinded (attempt 0): no r_inv needed
                auto tr1 = std::chrono::high_resolution_clock::now();

                // k16 challenges are always B-smooth (slot 0 guaranteed smooth by generator)
                hit_stats.hit_type.push_back(0);
                hit_stats.cofactor_bits.push_back(0);
                hit_stats.medium_factor_count.push_back(0);

                xfer.h2d_ms.push_back(std::chrono::duration<double, std::milli>(t_h2d1 - t_h2d0).count());
                xfer.d2h_ms.push_back(std::chrono::duration<double, std::milli>(t_d2h1 - t_d2h0).count());

                factor_ms_v.push_back((double)km / batch_size); // amortized per candidate
                recon_ms_v.push_back(std::chrono::duration<double, std::milli>(tr1 - tr0).count());
                kfl_free(&fl_w);
                mpz_clear(y_w);
            }
        }

        gpu_print_benchmark("k16", (int)factor_ms_v.size(), powm_baseline_ms,
                            _gpu_vmean(factor_ms_v), _gpu_vmean(recon_ms_v), n_gpus);
        gpu_write_json_warm(json_path, "k16", method, exponent, 16, challenge_bits,
                            0.0,
                            cold_factor_ms, cold_recon_ms,
                            factor_ms_v, recon_ms_v, 1, n_gpus, hit_stats, xfer);
    }

cleanup:
    if (cache_dirty)
        kf_cbrt_cache_save(&cache, cache_path);
    kfl_free(&solved_factors);
    mpz_clears(y_mpz, cofactor, result, nullptr);
    mpz_clears(challenge, modulo, challenge_mod, nullptr);
    kf_blinding_table_free(btable, batch_size);
    kf_cbrt_cache_clear(&cache);
    free(primes_ul);
    free(primes32);
    free(h_out_counts);
    CUDA_CHECK(cudaFreeHost(h_y_batch));
    CUDA_CHECK(cudaFreeHost(h_out_factors));
    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(devs[g].d_y_batch));
        CUDA_CHECK(cudaFree(devs[g].d_out_factors));
        CUDA_CHECK(cudaFree(devs[g].d_out_counts));
        CUDA_CHECK(cudaEventDestroy(devs[g].ev0));
        CUDA_CHECK(cudaEventDestroy(devs[g].ev1));
    }

    return solved_b >= 0 ? 0 : 2;
}