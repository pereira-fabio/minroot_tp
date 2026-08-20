// =============================================================================
// alg2_ksweep_cuda.cu — Algorithm 2 with a plug-and-play smoothness bound
//
// WHAT THIS IS
// ------------
// The same attack as alg2_k32_cuda.cu -- identical trial-division core, identical
// reconstruction path -- with the smoothness bound B = 2^k as a run-time argument
// instead of a compile-time 2^32. It exists to answer one question the three
// shipped variants can only sample at three widely-spaced points:
//
//     how far can k rise before the prime table stops fitting in GPU cache,
//     and does crossing that boundary actually show up in the timings?
//
// k16 (26 KB, fits L1) wins decisively on a GPU; k32 (813 MB) loses to the CPU
// cluster. analysis/cache_residency.py predicts the boundary analytically -- the
// largest L2-resident bound is k30 on an L40 at 4 GPUs, k26 on a V100 -- but a
// prediction about residency is not a measurement of speed. This binary produces
// the measurement.
//
// WHY IT IS A SEPARATE BINARY (do not merge this back into alg2_k32_cuda.cu)
// --------------------------------------------------------------------------//
// So this binary has its OWN identity end to end:
//     algorithm tag   "ksweep"        (never "k32")
//     JSON filename   ksweep_table_<job>_<size>_e<N>.json
//     results dir     results/results/iris/ksweep/
// It shares the *core* -- mod_big_by_u32, the table loader, the cbrt cache, the
// blinding table -- through the existing headers, and changes nothing in them.
//
// SCOPE
// -----
// Table method only. The sieve generates primes on the fly and so has no table to
// be resident or not; including it would answer a different question. Flat tables
// only for the same reason -- delta encoding is an orthogonal axis already
// measured in sub6:delta, and mixing the two would confound this one.
//
// PREREQUISITE
//     tools/gen_prime_table <k>        -> data/k<k>_primes.bin   (~0.1 s for k20)
//     tools/gen_prime_table --all      -> k17..k31
//
// CLI
//   ./alg2_ksweep_cuda [<challenge_file> <modulo_file>] [options]
//     --k N                         smoothness bound 2^N, 17..32  [REQUIRED]
//     --data DIR                    challenge directory (with --size)
//     --size N                      modulus size in bits (required with --data)
//     --exponent E                  root exponent (default: 3)
//     --primes FILE                 override the table path (default data/kN_primes.bin)
//     --n-challenges N              number of challenge files (default: 10)
//     --n-gpus N                    physical devices to use (default: 1)
//     --split N                     divide the table N ways and time ONE shard
//                                   (default: = n-gpus, the ordinary split)
//     --json FILE                   write JSON results to FILE
//
// THE TWO AXES, AND WHY --split IS THE IMPORTANT ONE
// --------------------------------------------------
// Sweeping k varies the table size, but it varies TOTAL WORK at the same time:
// k31 has twice the primes of k30, so it is slower for a reason that has nothing
// to do with cache. The two effects are confounded and no normalisation fully
// separates them.
//
// Sweeping --split at FIXED k does not have that problem. Total work is constant;
// only the shard size moves, sliding the working set down the memory hierarchy.
// Per-prime cost then isolates residency directly: it should be flat while the
// shard streams from GDDR6 and drop when the shard starts fitting in L2.
//
// It also lifts the 4-GPU ceiling. The interesting shard sizes for k32 are 1/64
// and 1/256 of the table; IRIS will not give you 256 cards, but one card can
// process a 1/256 shard and report exactly what one of those 256 would cost.
// That is the measurement analysis/gpu_scaling_sim.py currently has to
// extrapolate, and the reason its projections past the residency threshold carry
// an "optimistic" caveat.
//
//   ./alg2_ksweep_cuda --k 32 --split 256 --n-gpus 1 --data data --size 24576
//     -> times one 1/256 shard of the k32 table: 0.79 MB, comfortably L2-resident
// =============================================================================

#include "gpu_common.h"

#include <gmp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

#include "../common/common.h"
#include "../common/factorization.h"

// ---------------------------------------------------------------------------
// Constant memory: blinded value y (one limb array, updated per attempt)
// ---------------------------------------------------------------------------

__constant__ uint64_t d_y_limbs[MAX_LIMBS];
__constant__ int d_nlimbs;

// ---------------------------------------------------------------------------
// Trial-division kernel.
//
// Deliberately identical in body to k32_trial_divide_kernel: the point of the
// sweep is that ONLY the table changes between runs, so the kernel must not.
// mod_big_by_u32 is the shared primitive from gpu_common.h -- that is the "same
// core as Algorithm 2" this binary is built on.
// ---------------------------------------------------------------------------

__global__ void ksweep_trial_divide_kernel(
    const uint32_t *__restrict__ primes, int64_t nprimes,
    uint64_t *out_factors, int *out_count)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int nlimbs = d_nlimbs;

    for (int64_t i = idx; i < nprimes; i += stride)
    {
        uint64_t p = (uint64_t)primes[i];
        uint64_t rem = mod_big_by_u32(d_y_limbs, nlimbs, p);
        if (rem == 0)
        {
            int slot = atomicAdd(out_count, 1);
            if (slot < MAX_GPU_FACTORS)
                out_factors[slot] = p;
        }
    }
}

// ---------------------------------------------------------------------------
// Device L2 capacity, for the residency verdict printed alongside the timing.
// Queried rather than hard-coded so the verdict stays right on hardware this
// was not written against.
// ---------------------------------------------------------------------------

static double device_l2_mb(int dev)
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return 0.0;
    return (double)prop.l2CacheSize / 1e6;
}

// read_mpz_from_file prints an error and returns void, so a missing file leaves
// the mpz at zero and the run limps on producing nonsense. Check first and die
// with the path, which is the only thing the operator needs to see.
static void require_file(const char *path, const char *what)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "ERROR: %s not found: %s\n", what, path);
        fprintf(stderr, "       challenge sets live in data/challenges/k<K>/ --\n"
                        "       pass e.g. --data data/challenges/k32\n");
        exit(1);
    }
    fclose(f);
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [<challenge_file> <modulo_file>] [options]\n"
            "  --k N                         smoothness bound 2^N, 17..32  [REQUIRED]\n"
            "  --data DIR                    challenge directory (use with --size)\n"
            "  --size N                      modulus size in bits\n"
            "  --exponent E                  root exponent (default: 3)\n"
            "  --primes FILE                 table path (default: data/kN_primes.bin)\n"
            "  --n-challenges N              challenge files to run (default: 10)\n"
            "  --n-gpus N                    devices to split the table across (default: 1)\n"
            "  --json FILE                   write JSON results\n"
            "\n"
            "  --chal-k K                    bound the challenge set was crafted for\n"
            "                                (default: --k; use 32 to sweep tables\n"
            "                                against the existing k32 challenges)\n"
            "\nBuild the table first:  tools/gen_prime_table <k>\n"
            "Challenge sets:         --data data/challenges/k32\n",
            prog);
}

int main(int argc, char **argv)
{
    const char *challenge_file = nullptr;
    const char *modulo_file = nullptr;
    const char *data_dir = nullptr;
    int size_bits = 0;
    unsigned long exponent = 3;
    const char *primes_path = nullptr; // resolved from k unless overridden
    const char *json_path = nullptr;
    int n_challenges = 10;
    int n_gpus = 1;
    int k_bound = 0; // no default: the sweep must be explicit
    int chal_k = 0;  // challenge-set bound; defaults to k_bound
    // --split N: divide the table into N shards and process only the first
    // n_gpus of them, so ONE physical device measures what one of N would see.
    // This is what makes the residency question answerable: IRIS caps a job at
    // 4 GPUs, but the interesting slices are 1/64 or 1/256 of the table. 0 =
    // "same as n_gpus", i.e. the ordinary all-devices-cover-the-table split.
    int split = 0;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--k") && i + 1 < argc)
            k_bound = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--data") && i + 1 < argc)
            data_dir = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc)
            size_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-challenges") && i + 1 < argc)
            n_challenges = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-runs") && i + 1 < argc)
            n_challenges = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-gpus") && i + 1 < argc)
            n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--split") && i + 1 < argc)
            split = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exponent") && i + 1 < argc)
            exponent = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--primes") && i + 1 < argc)
            primes_path = argv[++i];
        else if (!strcmp(argv[i], "--chal-k") && i + 1 < argc)
            chal_k = atoi(argv[++i]);
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

    // k is mandatory and bounded. Below 17 the streamed range is empty or nearly
    // so (the dense k16 cbrt table is the right structure there); above 32 a
    // prime no longer fits the u32 the table stores.
    if (k_bound < 17 || k_bound > 32)
    {
        fprintf(stderr, "ERROR: --k is required and must be 17..32 (got %d)\n\n", k_bound);
        print_usage(argv[0]);
        return 1;
    }

    static char kpath[256];
    if (!primes_path)
    {
        snprintf(kpath, sizeof(kpath), "data/k%d_primes.bin", k_bound);
        primes_path = kpath;
    }

    // Resolve challenge/modulo from --data + --size if not given positionally.
    // data/challenges/k<K>/{modulo_<K>_<bits>.txt, challenge_<n>_<K>_<bits>.txt}
    // per tools/challenges.c. --chal-k names the bound the SET was crafted for,
    // which is independent of --k: sweeping the table bound against a fixed
    // challenge set is exactly the point (see the sweep script).
    static char cbuf[512], mbuf[512];
    if (chal_k <= 0)
        chal_k = k_bound;
    if (data_dir && size_bits > 0 && !challenge_file)
    {
        snprintf(cbuf, sizeof(cbuf), "%s/challenge_0_%d_%d.txt", data_dir, chal_k, size_bits);
        snprintf(mbuf, sizeof(mbuf), "%s/modulo_%d_%d.txt", data_dir, chal_k, size_bits);
        challenge_file = cbuf;
        modulo_file = mbuf;
    }
    if (!challenge_file || !modulo_file)
    {
        print_usage(argv[0]);
        return 1;
    }
    require_file(modulo_file, "modulo");
    require_file(challenge_file, "challenge");

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

    if (split <= 0)
        split = n_gpus; // default: cover the whole table
    if (split < n_gpus)
    {
        fprintf(stderr, "ERROR: --split %d < --n-gpus %d; a shard per device is the minimum\n",
                split, n_gpus);
        return 1;
    }

    mpz_t challenge, modulo, e_inv, result;
    mpz_inits(challenge, modulo, e_inv, result, NULL);
    read_mpz_from_file(challenge, challenge_file); // aborts internally on failure
    read_mpz_from_file(modulo, modulo_file);
    const size_t challenge_bits = mpz_sizeinbase(modulo, 2);
    {
        mpz_t phi, e;
        mpz_inits(phi, e, NULL);
        mpz_sub_ui(phi, modulo, 1);
        mpz_set_ui(e, exponent);
        if (!mpz_invert(e_inv, e, phi))
        {
            fprintf(stderr, "Error: e is not invertible mod (M-1)\n");
            return 1;
        }
        mpz_clears(phi, e, NULL);
    }
    int nlimbs = (int)((mpz_sizeinbase(modulo, 2) + 63) / 64);
    if (nlimbs > MAX_LIMBS)
    {
        fprintf(stderr, "Error: modulus needs %d limbs, MAX_LIMBS=%d\n", nlimbs, MAX_LIMBS);
        return 1;
    }

    // ---- Load the k-specific table ----
    printf("=== Algorithm 2 k-sweep — B = 2^%d ===\n", k_bound);
    printf("Table: %s\n", primes_path);
    FILE *pf = fopen(primes_path, "rb");
    if (!pf)
    {
        fprintf(stderr, "Error: cannot open %s\n"
                        "       build it first:  tools/gen_prime_table %d\n",
                primes_path, k_bound);
        return 1;
    }
    fclose(pf);
    uint32_t *primes = nullptr;
    int64_t nprimes = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t cnt64 = 0;
        primes = kf_load_prime_table(&cnt64, primes_path);
        if (!primes)
        {
            fprintf(stderr, "Error: cannot load %s\n", primes_path);
            return 1;
        }
        nprimes = (int64_t)cnt64;
        printf("Loaded %ld primes in %.1f ms\n", (long)nprimes,
               std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
                   .count());
    }

    // ---- Residency verdict: the hypothesis this run is testing ----
    const double table_mb = (double)nprimes * sizeof(uint32_t) / 1e6;
    const double per_gpu_mb = table_mb / split; // shard size, not table/devices
    const double l2_mb = device_l2_mb(0);
    printf("Table %.2f MB total; %d-way split -> %.3f MB per shard\n",
           table_mb, split, per_gpu_mb);
    if (split != n_gpus)
        printf("SIMULATED: measuring %d shard(s) of a %d-way split on %d physical device(s).\n"
               "           Reported times are PER SHARD -- what one of %d GPUs would see.\n",
               n_gpus, split, n_gpus, split);
    if (l2_mb > 0)
        printf("Device L2 %.0f MB  ->  shard is %sL2-RESIDENT\n",
               l2_mb, (per_gpu_mb <= l2_mb) ? "" : "NOT ");
    printf("\n");

    // ---- Per-GPU setup ----
    std::vector<GpuDev> devs((size_t)n_gpus);
    gpu_xfer_prof_t xprof;
    gpu_xfer_reset(&xprof);
    // Shard by `split`, not by n_gpus. With split == n_gpus this is the ordinary
    // split and the devices between them cover the table. With split > n_gpus the
    // run covers only n_gpus/split of it -- deliberately, because the quantity
    // being measured is the cost of ONE shard, not of the whole sweep.
    int64_t base_slice = nprimes / split;
    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        int64_t start = (int64_t)g * base_slice;
        int64_t count = (g == split - 1) ? (nprimes - start) : base_slice;
        devs[g].n_slice = count;
        CUDA_CHECK(cudaMalloc(&devs[g].d_primes, (size_t)count * sizeof(uint32_t)));
        double t_x = gpu_now_ms();
        CUDA_CHECK(cudaMemcpy(devs[g].d_primes, primes + start,
                              (size_t)count * sizeof(uint32_t), cudaMemcpyHostToDevice));
        xprof.setup_h2d_ms += gpu_now_ms() - t_x;
        xprof.setup_h2d_bytes += (size_t)count * sizeof(uint32_t);
        xprof.setup_h2d_calls++;
        CUDA_CHECK(cudaMalloc(&devs[g].d_out_factors, MAX_GPU_FACTORS * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&devs[g].d_out_count, sizeof(int)));
        CUDA_CHECK(cudaEventCreate(&devs[g].ev0));
        CUDA_CHECK(cudaEventCreate(&devs[g].ev1));
        CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &nlimbs, sizeof(int)));
        printf("  GPU %d: %ld primes (offset %ld)\n", g, (long)count, (long)start);
    }

    uint64_t *h_out_factors = (uint64_t *)malloc(MAX_GPU_FACTORS * sizeof(uint64_t));

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

    const int THREADS = 256, BLOCKS = 1024;

    auto launch_all = [&](const mpz_t y) -> float
    {
        uint64_t y_limbs[MAX_LIMBS] = {};
        size_t cnt = 0;
        mpz_export(y_limbs, &cnt, -1, sizeof(uint64_t), 0, 0, y);
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            double t_x = gpu_now_ms();
            CUDA_CHECK(cudaMemcpyToSymbol(d_y_limbs, y_limbs, sizeof(uint64_t) * MAX_LIMBS));
            xprof.challenge_h2d_ms += gpu_now_ms() - t_x;
            xprof.challenge_h2d_bytes += sizeof(uint64_t) * MAX_LIMBS;
            xprof.challenge_h2d_calls++;
            CUDA_CHECK(cudaMemset(devs[g].d_out_factors, 0, MAX_GPU_FACTORS * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemset(devs[g].d_out_count, 0, sizeof(int)));
            CUDA_CHECK(cudaEventRecord(devs[g].ev0));
            ksweep_trial_divide_kernel<<<BLOCKS, THREADS>>>(
                devs[g].d_primes, devs[g].n_slice,
                devs[g].d_out_factors, devs[g].d_out_count);
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

    auto collect_factors = [&](kfactor_list_t *fl, mpz_t cof, const mpz_t y)
    {
        for (int g = 0; g < n_gpus; g++)
        {
            CUDA_CHECK(cudaSetDevice(g));
            int h_count = 0;
            double t_x = gpu_now_ms();
            CUDA_CHECK(cudaMemcpy(&h_count, devs[g].d_out_count, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            xprof.factors_d2h_ms += gpu_now_ms() - t_x;
            xprof.factors_d2h_bytes += sizeof(int);
            xprof.factors_d2h_calls++;
            if (h_count <= 0)
                continue;
            if (h_count > MAX_GPU_FACTORS)
                h_count = MAX_GPU_FACTORS;
            t_x = gpu_now_ms();
            CUDA_CHECK(cudaMemcpy(h_out_factors, devs[g].d_out_factors,
                                  (size_t)h_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            xprof.factors_d2h_ms += gpu_now_ms() - t_x;
            xprof.factors_d2h_bytes += (size_t)h_count * sizeof(uint64_t);
            xprof.factors_d2h_calls++;
            for (int i = 0; i < h_count; i++)
            {
                unsigned long p = (unsigned long)h_out_factors[i];
                int k = 0;
                if (cof)
                {
                    while (mpz_divisible_ui_p(cof, p))
                    {
                        mpz_divexact_ui(cof, cof, p);
                        k++;
                    }
                }
                if (k > 0)
                    kfl_append_ul(fl, p, k);
            }
        }
        (void)y;
    };

    // ---- powm baseline (same reference the whole thesis uses) ----
    double powm_baseline_ms = 0.0;
    {
        mpz_t chk;
        mpz_init(chk);
        const int REPS = 5;
        std::vector<double> t(REPS);
        for (int i = 0; i < REPS; i++)
        {
            double t0 = gpu_now_ms();
            mpz_powm(chk, challenge, e_inv, modulo);
            t[i] = gpu_now_ms() - t0;
        }
        std::sort(t.begin(), t.end());
        powm_baseline_ms = t[REPS / 2];
        mpz_clear(chk);
        printf("powm baseline: %.2f ms (median of %d)\n\n", powm_baseline_ms, REPS);
    }

    // ---- Warm runs ----
    std::vector<double> factor_v, recon_v;
    int solved = 0;
    for (int run = 0; run < n_challenges; run++)
    {
        char cf[512];
        // Same naming as the initial load: challenge_<n>_<chal_k>_<bits>.txt.
        // read_mpz_from_file exit(1)s on a missing file, so getting this wrong
        // kills the process on the first warm run -- which is exactly what it
        // did before this was fixed.
        if (data_dir && size_bits > 0)
            snprintf(cf, sizeof(cf), "%s/challenge_%d_%d_%d.txt",
                     data_dir, run, chal_k, size_bits);
        else
            snprintf(cf, sizeof(cf), "%s", challenge_file);
        mpz_t y;
        mpz_init(y);
        read_mpz_from_file(y, cf);

        float kernel_ms = launch_all(y);

        mpz_t cof;
        mpz_init_set(cof, y);
        kfactor_list_t fl;
        kfl_init(&fl);
        collect_factors(&fl, cof, y);

        double t_rec0 = gpu_now_ms();
        bool smooth = (mpz_cmp_ui(cof, 1) == 0);
        if (smooth)
        {
            kf_compute_cbrt_from_factors_omp(result, &fl, modulo, &cache,
                                             &cache_dirty, exponent);
            mpz_t chk;
            mpz_init(chk);
            mpz_powm_ui(chk, result, exponent, modulo);
            if (mpz_cmp(chk, y) == 0)
                solved++;
            mpz_clear(chk);
        }
        double recon_ms = gpu_now_ms() - t_rec0;

        factor_v.push_back((double)kernel_ms);
        recon_v.push_back(recon_ms);
        printf("  run %2d: factor %8.2f ms  recon %7.2f ms  total %8.2f ms  %s\n",
               run, (double)kernel_ms, recon_ms, kernel_ms + recon_ms,
               smooth ? "smooth" : "NOT smooth");

        kfl_free(&fl);
        mpz_clears(cof, y, NULL);
    }
    if (cache_dirty)
        kf_cbrt_cache_save(&cache, cache_path);

    // ---- Summary ----
    if (!factor_v.empty())
    {
        double fs = 0, rs = 0;
        for (size_t i = 0; i < factor_v.size(); i++)
        {
            fs += factor_v[i];
            rs += recon_v[i];
        }
        double fm = fs / factor_v.size(), rm = rs / recon_v.size();
        printf("\n=== k=%d, %d-way split, %.3f MB/shard%s ===\n", k_bound, split, per_gpu_mb,
               (l2_mb > 0 && per_gpu_mb <= l2_mb) ? " [L2-resident]" : "");
        printf("  factor  %8.2f ms    recon %7.2f ms    total %8.2f ms\n", fm, rm, fm + rm);
        // A challenge crafted 2^chal_k-smooth cannot factor under a smaller
        // bound: its factors run up to 2^chal_k and a k<chal_k table simply does
        // not contain them. factor_ms is still exactly right -- the kernel scans
        // its shard either way, which is the quantity the residency question
        // turns on -- but end-to-end speedup is only meaningful when the solve
        // completes, so it is suppressed rather than printed as a wrong number.
        if (solved > 0)
        {
            printf("  speedup %8.2fx   (vs %.2f ms sequential)\n",
                   (fm + rm) > 0 ? powm_baseline_ms / (fm + rm) : 0.0, powm_baseline_ms);
        }
        else
        {
            printf("  speedup       n/a   (challenge is 2^%d-smooth; a 2^%d table\n"
                   "                       cannot factor it -- factor_ms above is\n"
                   "                       still the valid measurement)\n",
                   chal_k, k_bound);
        }
        // Per-prime cost normalises the shard size away, so a change here is a
        // change in the COST OF A PRIME -- which is the residency signal. The
        // divisor is the shard, not the table.
        printf("  ns per prime per pass: %.3f   (shard = %ld primes)\n",
               fm * 1e6 / (double)(nprimes / split), (long)(nprimes / split));
        printf("  solved %d/%zu\n", solved, factor_v.size());
    }

    // Identity is "ksweep", never "k32": these results must never join with the
    // Algorithm 2 corpus (see the header note).
    // n_gpus in the JSON is the SIMULATED split, not the physical device count:
    // the analysis reasons about "what would one of N GPUs cost", and the number
    // of cards that stood in for it is a runtime detail. The console output above
    // states both.
    gpu_write_json_warm(json_path, "ksweep", GPU_METHOD_TABLE, exponent, k_bound,
                        challenge_bits, powm_baseline_ms, 0.0, 0.0,
                        factor_v, recon_v, solved, split);
    if (json_path)
        printf("\nJSON -> %s\n", json_path);

    for (int g = 0; g < n_gpus; g++)
    {
        CUDA_CHECK(cudaSetDevice(g));
        cudaFree(devs[g].d_primes);
        cudaFree(devs[g].d_out_factors);
        cudaFree(devs[g].d_out_count);
        cudaEventDestroy(devs[g].ev0);
        cudaEventDestroy(devs[g].ev1);
    }
    free(h_out_factors);
    free(primes);
    kf_cbrt_cache_clear(&cache);
    mpz_clears(challenge, modulo, e_inv, result, NULL);
    return 0;
}
