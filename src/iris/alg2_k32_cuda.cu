// =============================================================================
// alg2_k32_cuda.cu — GPU-accelerated Algorithm 2, k32 variant
//
// Algorithm 2 (Biryukov et al. CRYPTO 2024, §2.1), smoothness bound B = 2^32.
// Mirrors the CPU alg2_k32.cpp structure: --method {powm|sieve|table}, cold
// run + N warm runs, same JSON format (warm_avg / warm_stats / warm_runs).
//
// Methods:
//   powm  — measure sequential mpz_powm baseline only; no GPU kernel
//   table — load prime table from disk (flat K32PRIMS or delta K32DLTA1,
//           auto-detected); GPU trial-divides y
//   sieve — generate primes in (65536, 2^32] in-memory (slow startup)
//
// Phase 1: GPU trial-divides y by all ~203 M primes in (65536, 2^32].
//          With --n-gpus N, the prime range is split evenly across N devices;
//          each device checks a disjoint [p_start, p_end] slice; CPU merges.
// Phase 2: CPU cofactor check + cbrt reconstruction.
//
// Challenge files store y directly (smooth value), so attempt 0 (unblinded)
// is always smooth.  Warm runs re-execute the same kernel for timing stats.
//
// CLI:
//   ./alg2_k32_cuda [<challenge_file> <modulo_file>] [options]
//     --data DIR                    challenge directory (with --size, replaces positional args)
//     --size N                      modulus size in bits (required with --data)
//     --method {powm|sieve|table}   benchmark method (default: table)
//     --exponent E                  root exponent (default: 3)
//     --primes FILE                 k32 prime table (flat .bin or delta .dlt; default data/k32_primes.bin)
//     --table-format {auto|flat|delta}  format (default auto = detect by file magic).
//                                   delta = 1 byte/prime, decoded on-GPU per block —
//                                   the L2-residency experiment (gpu-delta-decode-kernel-plan.md)
//     --n-challenges N              number of challenge files to run (default: 10)
//     --n-gpus N                    split prime range across N CUDA devices (default: 1)
//     --json FILE                   write JSON output to FILE
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
#include "gpu_k32_kernels.cuh"
#include "delta_table_gpu.cuh" // experiments/delta — L2-residency experiment

// ---------------------------------------------------------------------------
// Host utilities
// ---------------------------------------------------------------------------

static uint32_t *sieve_primes_u32(uint64_t lo, uint64_t hi, int64_t *out_count)
{
    std::vector<bool> is_comp(hi + 1, false);
    for (uint64_t p = 2; p * p <= hi; p++)
    {
        if (is_comp[p])
            continue;
        for (uint64_t m = p * p; m <= hi; m += p)
            is_comp[m] = true;
    }
    int64_t cnt = 0;
    for (uint64_t p = lo; p <= hi; p++)
        if (!is_comp[p])
            cnt++;
    uint32_t *arr = (uint32_t *)malloc(sizeof(uint32_t) * cnt);
    int64_t j = 0;
    for (uint64_t p = lo; p <= hi; p++)
        if (!is_comp[p])
            arr[j++] = (uint32_t)p;
    *out_count = cnt;
    return arr;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [<challenge_file> <modulo_file>] [options]\n"
            "  --data DIR --size N           use challenge directory (replaces positional args)\n"
            "  --method {powm|sieve|table}   benchmark method (default: table)\n"
            "  --exponent E                  root exponent (default: 3)\n"
            "  --primes FILE                 k32 prime table (default: data/k32_primes.bin)\n"
            "  --n-challenges N              challenge files to run (default: 10)\n"
            "  --n-gpus N                    split prime range across N CUDA devices (default: 1)\n"
            "  --json FILE                   write JSON output\n",
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
    const char *primes_path = "data/k32_primes.bin";
    const char *table_format = "auto"; // auto | flat | delta (auto = detect by file magic)
    const char *json_path = nullptr;
    gpu_method_t method = GPU_METHOD_TABLE;
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
        else if (!strcmp(argv[i], "--n-gpus") && i + 1 < argc)
            n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exponent") && i + 1 < argc)
            exponent = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--primes") && i + 1 < argc)
            primes_path = argv[++i];
        else if (!strcmp(argv[i], "--table-format") && i + 1 < argc)
            table_format = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_path = argv[++i];
        else if (!strcmp(argv[i], "--method") && i + 1 < argc)
            method = gpu_parse_method(argv[++i]);
        else if (argv[i][0] != '-')
        {
            if (!challenge_file)
                challenge_file = argv[i];
            else if (!modulo_file)
                modulo_file = argv[i];
        }
    }

    // Clamp n_gpus to available devices
    {
        int avail = 1;
        cudaGetDeviceCount(&avail);
        if (n_gpus < 1)
            n_gpus = 1;
        if (n_gpus > avail)
        {
            fprintf(stderr, "Warning: only %d GPU(s) available, using %d\n", avail, avail);
            n_gpus = avail;
        }
    }

    // Derive paths from --data/--size when given (overrides positional args)
    static char _modu_buf[512], _chal_buf[512];
    if (data_dir && size_bits > 0)
    {
        snprintf(_modu_buf, sizeof(_modu_buf), "%s/modulo_32_%d%s.txt", data_dir, size_bits, modulo_suffix);
        snprintf(_chal_buf, sizeof(_chal_buf), "%s/challenge_0_32_%d.txt", data_dir, size_bits);
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

    size_t challenge_bits = mpz_sizeinbase(modulo, 2);
    int nlimbs = (int)challenge_bits / 64 + 2;
    if (nlimbs > MAX_LIMBS)
    {
        fprintf(stderr, "Error: modulus too large for MAX_LIMBS=%d\n", MAX_LIMBS);
        return 1;
    }

    printf("Challenge: %zu bits | Modulus: %zu bits | Exponent: %lu | Method: %s | GPUs: %d\n",
           mpz_sizeinbase(challenge, 2), challenge_bits, exponent, gpu_method_name(method), n_gpus);

    // ---- powm baseline ----
    printf("Measuring powm baseline...\n");
    double powm_baseline_ms = gpu_measure_powm_baseline(challenge_mod, modulo, exponent);
    printf("  powm_baseline_ms = %.4f ms\n", powm_baseline_ms);

    // ---- powm method: write JSON and exit ----
    if (method == GPU_METHOD_POWM)
    {
        std::vector<double> empty;
        gpu_write_json_warm(json_path, "k32", method, exponent, 32, challenge_bits,
                            powm_baseline_ms, 0.0, 0.0, empty, empty, 0, n_gpus);
        mpz_clears(challenge, modulo, challenge_mod, nullptr);
        return 0;
    }

    // ---- Load prime table (flat or delta) ----
    // Delta is auto-detected by the 8-byte file magic (K32DLTA1); --table-format
    // {flat|delta} can force/assert a format. Delta = the L2-residency experiment
    // (see gpu-delta-decode-kernel-plan.md); sieve method is always flat/in-memory.
    int64_t nprimes = 0;
    uint32_t *primes = nullptr; // flat host table; nullptr in delta mode

    // Delta-mode state: whole delta file's block bytes (host) + per-GPU block split.
    bool use_delta = false;
    uint8_t *all_blocks = nullptr;
    size_t block_bytes_total = 0;
    int64_t n_blocks_total = 0;
    std::vector<uint8_t *> d_delta((size_t)n_gpus, nullptr);
    std::vector<uint32_t> dev_nblocks((size_t)n_gpus, 0);
    std::vector<uint64_t> dev_first_block((size_t)n_gpus, 0);

    if (method == GPU_METHOD_TABLE)
    {
        FILE *pf = fopen(primes_path, "rb");
        if (!pf)
        {
            fprintf(stderr,
                    "Error: --method table requires %s (not found).\n"
                    "Generate flat with `alg2_k32 --precompute`, delta with `--convert-delta`, "
                    "or use --method sieve.\n",
                    primes_path);
            return 1;
        }
        // Peek the 8-byte magic to auto-detect flat vs delta.
        uint64_t magic = 0;
        if (fread(&magic, 8, 1, pf) != 1)
        {
            fprintf(stderr, "Error: cannot read header of %s\n", primes_path);
            fclose(pf);
            return 1;
        }
        bool file_is_delta = (magic == KF_DELTA_MAGIC_U32);
        bool file_is_flat = (magic == KF_PRIME_TABLE_MAGIC);
        if (strcmp(table_format, "delta") == 0 && !file_is_delta)
        {
            fprintf(stderr, "Error: --table-format delta but %s is not a K32DLTA1 file\n", primes_path);
            fclose(pf);
            return 1;
        }
        if (strcmp(table_format, "flat") == 0 && !file_is_flat)
        {
            fprintf(stderr, "Error: --table-format flat but %s is not a flat K32PRIMS file\n", primes_path);
            fclose(pf);
            return 1;
        }
        use_delta = file_is_delta;

        if (use_delta)
        {
            // Header: [u64 magic][u64 count][u32 B][u32 width]; validate, slurp blocks.
            uint64_t count = 0;
            uint32_t Bhdr = 0, width = 0;
            if (fread(&count, 8, 1, pf) != 1 || fread(&Bhdr, 4, 1, pf) != 1 ||
                fread(&width, 4, 1, pf) != 1)
            {
                fprintf(stderr, "Error: truncated delta header in %s\n", primes_path);
                fclose(pf);
                return 1;
            }
            if (Bhdr != KF_DELTA_BLOCK || width != 4)
            {
                fprintf(stderr, "Error: unsupported delta table (B=%u width=%u; expected B=%d width=4)\n",
                        Bhdr, width, KF_DELTA_BLOCK);
                fclose(pf);
                return 1;
            }
            nprimes = (int64_t)count;
            n_blocks_total = (int64_t)((count + Bhdr - 1) / Bhdr);
            fseek(pf, 0, SEEK_END);
            long fsz = ftell(pf);
            block_bytes_total = (size_t)fsz - KF_DELTA_HDR;
            fseek(pf, KF_DELTA_HDR, SEEK_SET);
            all_blocks = (uint8_t *)malloc(block_bytes_total);
            if (!all_blocks || fread(all_blocks, 1, block_bytes_total, pf) != block_bytes_total)
            {
                fprintf(stderr, "Error: cannot read delta blocks from %s\n", primes_path);
                free(all_blocks);
                fclose(pf);
                return 1;
            }
            fclose(pf);
            printf("Loaded delta k32 table: %ld primes, %ld blocks, %.0f MB (%s)\n",
                   (long)nprimes, (long)n_blocks_total, block_bytes_total / 1e6, primes_path);
        }
        else
        {
            fclose(pf);
            printf("Loading flat k32 prime table from %s...\n", primes_path);
            auto t0 = std::chrono::high_resolution_clock::now();
            uint64_t cnt64 = 0;
            primes = kf_load_prime_table(&cnt64, primes_path);
            nprimes = (int64_t)cnt64;
            printf("Loaded %ld primes in %.1f ms\n", (long)nprimes,
                   std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - t0)
                       .count());
        }
    }
    else
    {
        printf("Sieving primes in (65536, 2^32] (slow; prefer --method table)...\n");
        auto t0 = std::chrono::high_resolution_clock::now();
        primes = sieve_primes_u32(65537, 0xFFFFFFFFULL, &nprimes);
        printf("Sieved %ld primes in %.1f ms\n", (long)nprimes,
               std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count());
    }

    // ---- Per-GPU setup: slice the table (by prime index for flat / by whole
    // block for delta), allocate device memory, upload nlimbs. ----
    printf("Setting up %d GPU(s), %ld primes total (%s)...\n",
           n_gpus, (long)nprimes, use_delta ? "delta table" : "flat table");
    std::vector<GpuDev> devs((size_t)n_gpus);
    gpu_xfer_prof_t xprof;
    gpu_xfer_reset(&xprof);
    int64_t base_slice = nprimes / n_gpus;                        // flat: primes/GPU
    int64_t base_blk = use_delta ? (n_blocks_total / n_gpus) : 0; // delta: blocks/GPU
    const size_t full = K32_DELTA_FULL;                           // 1027 bytes/full block
    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        if (use_delta)
        {
            // Split by WHOLE blocks — blocks are fixed-size + self-anchoring, so
            // the union is all primes with no mid-block boundary to track.
            int64_t b_lo = (int64_t)g * base_blk;
            int64_t b_hi = (g == n_gpus - 1) ? n_blocks_total : (int64_t)(g + 1) * base_blk;
            dev_first_block[g] = (uint64_t)b_lo;
            dev_nblocks[g] = (uint32_t)(b_hi - b_lo);
            size_t byte_start = (size_t)b_lo * full;
            // last device ends at the (possibly short) final block → clamp to file.
            size_t byte_end = (b_hi == n_blocks_total) ? block_bytes_total : (size_t)b_hi * full;
            size_t dev_bytes = byte_end - byte_start;
            devs[g].n_slice = 0;
            devs[g].d_primes = nullptr;
            CUDA_CHECK(cudaMalloc(&d_delta[g], dev_bytes));
            double t_x = gpu_now_ms();
            CUDA_CHECK(cudaMemcpy(d_delta[g], all_blocks + byte_start, dev_bytes,
                                  cudaMemcpyHostToDevice));
            xprof.setup_h2d_ms += gpu_now_ms() - t_x;
            xprof.setup_h2d_bytes += dev_bytes;
            xprof.setup_h2d_calls++;
            printf("  GPU %d: blocks [%ld, %ld) = %u blocks, %.0f MB\n",
                   g, (long)b_lo, (long)b_hi, dev_nblocks[g], dev_bytes / 1e6);
        }
        else
        {
            int64_t start = (int64_t)g * base_slice;
            int64_t count = (g == n_gpus - 1) ? (nprimes - start) : base_slice;
            devs[g].n_slice = count;
            CUDA_CHECK(cudaMalloc(&devs[g].d_primes, (size_t)count * sizeof(uint32_t)));
            double t_x = gpu_now_ms();
            CUDA_CHECK(cudaMemcpy(devs[g].d_primes, primes + start,
                                  (size_t)count * sizeof(uint32_t), cudaMemcpyHostToDevice));
            xprof.setup_h2d_ms += gpu_now_ms() - t_x;
            xprof.setup_h2d_bytes += (size_t)count * sizeof(uint32_t);
            xprof.setup_h2d_calls++;
            printf("  GPU %d: %ld primes (offset %ld)\n", g, (long)count, (long)start);
        }
        CUDA_CHECK(cudaMalloc(&devs[g].d_out_factors, MAX_GPU_FACTORS * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&devs[g].d_out_count, sizeof(int)));
        CUDA_CHECK(cudaEventCreate(&devs[g].ev0));
        CUDA_CHECK(cudaEventCreate(&devs[g].ev1));
        CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &nlimbs, sizeof(int)));
    }

    // ---- Shared host output buffer ----
    uint64_t *h_out_factors = (uint64_t *)malloc(MAX_GPU_FACTORS * sizeof(uint64_t));

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

    // ---- Blinding table (255 entries, matching CPU BLIND_TABLE_SIZE) ----
    const int BLIND_TABLE_SIZE = 255;
    blinding_pair_t *btable = kf_blinding_table_alloc(BLIND_TABLE_SIZE, modulo, exponent);
    printf("Blinding table ready (%d entries).\n", BLIND_TABLE_SIZE);

    const int THREADS = 256, BLOCKS = 1024;

    // Upload y to all GPUs, launch kernel on each in parallel, sync and return
    // wall-clock kernel time (max across all devices = effective per-challenge cost).
    auto launch_all = [&](const mpz_t y) -> float
    {
        uint64_t y_limbs[MAX_LIMBS] = {};
        size_t cnt = 0;
        mpz_export(y_limbs, &cnt, -1, sizeof(uint64_t), 0, 0, y);
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            // Counted separately from the kernel: the event bracket below starts
            // AFTER this copy, so factor_ms has never included it.
            double t_x = gpu_now_ms();
            CUDA_CHECK(cudaMemcpyToSymbol(d_y_limbs, y_limbs, sizeof(uint64_t) * MAX_LIMBS));
            xprof.challenge_h2d_ms += gpu_now_ms() - t_x;
            xprof.challenge_h2d_bytes += sizeof(uint64_t) * MAX_LIMBS;
            xprof.challenge_h2d_calls++;
            CUDA_CHECK(cudaMemset(devs[g].d_out_factors, 0, MAX_GPU_FACTORS * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemset(devs[g].d_out_count, 0, sizeof(int)));
            CUDA_CHECK(cudaEventRecord(devs[g].ev0));
            if (use_delta)
            {
                uint32_t grid = dev_nblocks[g] ? dev_nblocks[g] : 1; // one CUDA block per delta block
                k32_trial_divide_delta_kernel<<<grid, THREADS>>>(
                    d_delta[g], dev_nblocks[g], dev_first_block[g], (uint64_t)nprimes,
                    devs[g].d_out_factors, devs[g].d_out_count);
            }
            else
            {
                k32_trial_divide_kernel<<<BLOCKS, THREADS>>>(devs[g].d_primes, devs[g].n_slice,
                                                             devs[g].d_out_factors, devs[g].d_out_count);
            }
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
        return km;
    };

    // Download factor lists from all devices; build factor list + divide cofactor.
    // Pass cof=nullptr to skip cofactor division.
    auto collect_factors = [&](kfactor_list_t *fl, mpz_t cof, const mpz_t y)
    {
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            int h_count = 0;
            double t_x = gpu_now_ms();
            CUDA_CHECK(cudaMemcpy(&h_count, devs[g].d_out_count,
                                  sizeof(int), cudaMemcpyDeviceToHost));
            xprof.factors_d2h_ms += gpu_now_ms() - t_x;
            xprof.factors_d2h_bytes += sizeof(int);
            xprof.factors_d2h_calls++;
            if (h_count <= 0)
                continue;
            if (h_count > MAX_GPU_FACTORS)
            {
                fprintf(stderr, "Warning: factor overflow on GPU %d (%d > %d) — "
                                "processing first %d; increase MAX_GPU_FACTORS\n",
                        g, h_count, MAX_GPU_FACTORS, MAX_GPU_FACTORS);
                h_count = MAX_GPU_FACTORS;
            }
            double t_f = gpu_now_ms();
            CUDA_CHECK(cudaMemcpy(h_out_factors, devs[g].d_out_factors,
                                  (size_t)h_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            xprof.factors_d2h_ms += gpu_now_ms() - t_f;
            xprof.factors_d2h_bytes += (size_t)h_count * sizeof(uint64_t);
            xprof.factors_d2h_calls++;
            for (int j = 0; j < h_count; j++)
            {
                unsigned long p = (unsigned long)h_out_factors[j];
                int e = count_exponent(y, p);
                if (e > 0)
                {
                    kfl_append_ul(fl, p, e);
                    if (cof)
                        for (int k = 0; k < e; k++)
                            mpz_divexact_ui(cof, cof, p);
                }
            }
        }
    };

    // ---- Find the smooth attempt (attempt 0 = unblinded, always smooth) ----
    mpz_t y_smooth, cofactor, result;
    mpz_inits(y_smooth, cofactor, result, nullptr);
    kfactor_list_t solved_factors;
    kfl_init(&solved_factors);
    int solved_attempt = -1;

    for (int attempt = 0; attempt <= BLIND_TABLE_SIZE && solved_attempt < 0; attempt++)
    {
        if (attempt == 0)
            mpz_set(y_smooth, challenge_mod);
        else
        {
            mpz_mul(y_smooth, challenge_mod, btable[attempt - 1].r_blind);
            mpz_mod(y_smooth, y_smooth, modulo);
        }

        float km = launch_all(y_smooth);

        kfl_init(&solved_factors);
        mpz_set(cofactor, y_smooth);
        collect_factors(&solved_factors, cofactor, y_smooth);

        if (mpz_cmp_ui(cofactor, 1) == 0)
        {
            printf("Attempt %d: k32-smooth (%d factors, kernel %.2f ms)\n",
                   attempt, solved_factors.count, km);
            solved_attempt = attempt;
        }
        else
        {
            kfl_free(&solved_factors);
        }
    }

    if (solved_attempt < 0)
    {
        fprintf(stderr, "Warning: no smooth attempt found within blinding table\n");
        std::vector<double> empty;
        gpu_write_json_warm(json_path, "k32", method, exponent, 32, challenge_bits,
                            0.0, 0.0, 0.0, empty, empty, 0, n_gpus);
        goto cleanup;
    }

    // ---- Cold run: kernel + reconstruction (warms cbrt cache) ----
    {
        // Cold and warm are counted separately: the cold run is where the cache
        // misses land (every cbrt is a full mpz_powm), the warm runs are the
        // steady state the speedup figures are taken from.
        kf_cbrt_stats_reset();
        kf_recon_prof_reset();
        float cold_km = launch_all(y_smooth);
        auto tc0 = std::chrono::high_resolution_clock::now();
        kf_compute_cbrt_from_factors_omp(result, &solved_factors, modulo, &cache, &cache_dirty, exponent);
        if (solved_attempt > 0)
        {
            mpz_mul(result, result, btable[solved_attempt - 1].r_inv);
            mpz_mod(result, result, modulo);
        }
        auto tc1 = std::chrono::high_resolution_clock::now();
        double cold_factor_ms = cold_km;
        double cold_recon_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();
        verify_eth_root(result, challenge_mod, modulo, exponent);
        printf("Cold run: factor_ms=%.2f, recon_ms=%.4f\n", cold_factor_ms, cold_recon_ms);
        {
            char tg[128];
            snprintf(tg, sizeof(tg), "k32 %s %zub e%lu cold",
                     gpu_method_name(method), challenge_bits, exponent);
            kf_cbrt_stats_report(tg);
            const kf_recon_prof_t *rp = kf_recon_prof_get();
            printf("RECONPROF|%s|%.4f|%.4f|%ld|%ld\n", tg,
                   rp->parallel_ms, rp->serial_ms, rp->factors, rp->calls);
        }

        // ---- Warm runs: one per challenge file (different y each time) ----
        kf_cbrt_stats_reset();
        kf_recon_prof_reset();
        std::vector<double> factor_ms_v, recon_ms_v;
        factor_ms_v.reserve(n_challenges - 1);
        recon_ms_v.reserve(n_challenges - 1);
        GpuHitStats hit_stats;
        GpuTransferStats xfer;
        xfer.table_mb = use_delta ? (double)block_bytes_total / (1024.0 * 1024.0)
                                  : (double)nprimes * 4.0 / (1024.0 * 1024.0);

        for (int ci = 1; ci < n_challenges; ci++)
        {
            mpz_t y_w;
            mpz_init(y_w);
            if (data_dir && size_bits > 0)
            {
                if (!gpu_load_challenge(y_w, data_dir, ci, "32", size_bits, modulo))
                {
                    mpz_clear(y_w);
                    continue;
                }
            }
            else
            {
                mpz_set(y_w, y_smooth); // single-challenge compat: repeat same y
            }

            // H2D approx = wall-clock minus pure GPU kernel time (CUDA events)
            auto t_launch0 = std::chrono::high_resolution_clock::now();
            float km = launch_all(y_w);
            auto t_launch1 = std::chrono::high_resolution_clock::now();
            double h2d_approx = std::max(0.0,
                                         std::chrono::duration<double, std::milli>(t_launch1 - t_launch0).count() - (double)km);

            kfactor_list_t fl_w;
            kfl_init(&fl_w);
            mpz_t cof_w;
            mpz_init_set(cof_w, y_w);
            auto t_d2h0 = std::chrono::high_resolution_clock::now();
            collect_factors(&fl_w, cof_w, y_w);
            auto t_d2h1 = std::chrono::high_resolution_clock::now();
            double d2h_ms = std::chrono::duration<double, std::milli>(t_d2h1 - t_d2h0).count();

            xfer.h2d_ms.push_back(h2d_approx);
            xfer.d2h_ms.push_back(d2h_ms);

            auto tr0 = std::chrono::high_resolution_clock::now();
            kf_compute_cbrt_from_factors_omp(result, &fl_w, modulo, &cache, &cache_dirty, exponent);
            // attempt 0 is always unblinded: no r_inv multiplication needed
            auto tr1 = std::chrono::high_resolution_clock::now();

            // k32 challenges are always B-smooth (cofactor == 1 guaranteed by generator)
            hit_stats.hit_type.push_back(0);
            hit_stats.cofactor_bits.push_back(0);
            hit_stats.medium_factor_count.push_back(0);

            factor_ms_v.push_back((double)km);
            recon_ms_v.push_back(std::chrono::duration<double, std::milli>(tr1 - tr0).count());
            kfl_free(&fl_w);
            mpz_clears(cof_w, y_w, nullptr);
        }

        gpu_print_benchmark("k32", (int)factor_ms_v.size(), powm_baseline_ms,
                            _gpu_vmean(factor_ms_v), _gpu_vmean(recon_ms_v), n_gpus);
        {
            char tg[128];
            snprintf(tg, sizeof(tg), "k32 %s %zub e%lu warm",
                     gpu_method_name(method), challenge_bits, exponent);
            kf_cbrt_stats_report(tg);
            const kf_recon_prof_t *rp = kf_recon_prof_get();
            double rtot = rp->parallel_ms + rp->serial_ms;
            printf("\n--- reconstruction phase split  (%s) ---\n", tg);
            printf("  parallel (per-factor cbrt) %10.4f ms  %5.1f%%\n",
                   rp->parallel_ms, rtot > 0 ? rp->parallel_ms / rtot * 100.0 : 0.0);
            printf("  serial   (product mod M)   %10.4f ms  %5.1f%%\n",
                   rp->serial_ms, rtot > 0 ? rp->serial_ms / rtot * 100.0 : 0.0);
            printf("  %ld factors over %ld calls\n", rp->factors, rp->calls);
            printf("RECONPROF|%s|%.4f|%.4f|%ld|%ld\n", tg,
                   rp->parallel_ms, rp->serial_ms, rp->factors, rp->calls);
            gpu_xfer_report(&xprof, tg, _gpu_vmean(factor_ms_v));
        }
        gpu_write_json_warm(json_path, "k32", method, exponent, 32, challenge_bits,
                            0.0, // CPU convention: powm_baseline_ms=0 in sieve/table file
                            cold_factor_ms, cold_recon_ms,
                            factor_ms_v, recon_ms_v, 1, n_gpus, hit_stats, xfer);
    }

cleanup:
    if (cache_dirty)
        kf_cbrt_cache_save(&cache, cache_path);
    kfl_free(&solved_factors);
    mpz_clears(y_smooth, cofactor, result, nullptr);
    mpz_clears(challenge, modulo, challenge_mod, nullptr);
    kf_blinding_table_free(btable, BLIND_TABLE_SIZE);
    kf_cbrt_cache_clear(&cache);
    free(primes);     // nullptr in delta mode — safe
    free(all_blocks); // nullptr in flat mode  — safe
    free(h_out_factors);
    for (int g = 0; g < n_gpus; g++)
    {
        cudaSetDevice(g);
        cudaFree(devs[g].d_primes); // nullptr in delta mode — safe
        cudaFree(d_delta[g]);       // nullptr in flat mode  — safe
        cudaFree(devs[g].d_out_factors);
        cudaFree(devs[g].d_out_count);
        cudaEventDestroy(devs[g].ev0);
        cudaEventDestroy(devs[g].ev1);
    }

    return solved_attempt >= 0 ? 0 : 2;
}