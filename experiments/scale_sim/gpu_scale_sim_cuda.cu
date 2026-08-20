// =============================================================================
// gpu_scale_sim_cuda.cu — measure what an N-GPU deployment would actually cost
//
// WHAT IT ANSWERS
// ---------------
// IRIS caps a job at 4 GPUs. At 4 GPUs k32 reaches ~9x on an L40 against the CPU
// cluster's 52x, and the open question is how many GPUs it would take to matter
// -- and whether more GPUs keep helping at all.
//
// The honest way to answer that with 4 cards is to stop projecting and start
// measuring. Split the prime table N ways and run ONE node's worth of shards:
// that node then does exactly what a node of a real N-GPU deployment would do,
// on real silicon, with real cache behaviour. Everything that remains is a
// handful of inter-node terms, and all but one of those can be measured on the
// same host.
//
//     N simulated GPUs, 4 per node  ->  n_nodes = ceil(N/4)
//
//     factor        MEASURED  max over this node's GPUs, each on 1/N of the table
//     local_recon   MEASURED  this host rooting + multiplying its shards' factors
//     combine       MEASURED  (n_nodes - 1) full-width modmuls, timed on this host
//     gather        modelled  n_nodes partials to rank 0 -- see the note below
//     ------------------------------------------------------------------------
//     total = factor + local_recon + gather + combine
//     speedup = powm_baseline / total
//
// This supersedes analysis/gpu_scaling_sim.py, which had to extrapolate factor
// time as W/N below the smallest measured slice. That extrapolation is optimistic
// exactly where it matters -- a shard small enough to sit in L2 is no longer
// memory-bound, so further splitting should stop paying -- and on synthetic data
// the difference between extrapolating and measuring reached 3x at 64 GPUs. The
// point of this binary is that no such guess is made.
//
// THE ONE MODELLED TERM
// ---------------------
// `gather` is an MPI_Gather of n_nodes partial products to rank 0, and a
// single-node job cannot measure a network. It is taken from the AION
// measurement (20 us floor + ~0.43 us per rank at 24576b, from gather_ms across
// the 2..128 rank sweep) and is 0.2% of the solve even at 128 nodes, so the
// projection does not turn on it. Every other term is timed here.
//
// WHY THE COMBINE IS MEASURED RATHER THAN MODELLED
// -------------------------------------------------
// Rank 0's final combine is (n_nodes - 1) multiplications of |M|-bit residues
// modulo M. That is arithmetic, not communication, so this host can simply
// perform it and time it -- no cluster required, no constant to trust. It is
// also the term that decides the answer: it GROWS with node count while every
// other term shrinks, so it is what makes the speedup curve turn over.
//
// USAGE
//   ./gpu_scale_sim_cuda --data data --size 24576 [options]
//     --k N            smoothness bound 2^N (default 32; needs data/kN_primes.bin)
//     --splits "..."   simulated GPU counts (default: 1 2 4 8 16 32 64 128 256 512)
//     --n-gpus G       physical devices per simulated node (default: all, max 4)
//     --n-runs R       timed repeats per split (default: 5)
//     --exponent E     root exponent (default 3)
//     --json FILE      write the projection table as JSON
//
// Build:
//   make gpu-experiments      (or: make gpu_scale_sim_cuda ARCH=sm_89)
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

__constant__ uint64_t d_y_limbs[MAX_LIMBS];
__constant__ int      d_nlimbs;

// Identical body to the k32 kernel: the whole point is that only the shard size
// changes between runs, so the arithmetic must not.
__global__ void scale_trial_divide_kernel(
        const uint32_t * __restrict__ primes, int64_t nprimes,
        uint64_t *out_factors, int *out_count)
{
    int64_t idx    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    int nlimbs     = d_nlimbs;
    for (int64_t i = idx; i < nprimes; i += stride) {
        uint64_t p = (uint64_t)primes[i];
        if (mod_big_by_u32(d_y_limbs, nlimbs, p) == 0) {
            int slot = atomicAdd(out_count, 1);
            if (slot < MAX_GPU_FACTORS) out_factors[slot] = p;
        }
    }
}

// AION MPI_Gather of n partials to rank 0, microseconds. The only modelled term.
static double gather_us(int n_nodes) { return 20.0 + 0.43 * (double)(n_nodes > 1 ? n_nodes - 1 : 0); }

// Measure (n_nodes - 1) full-width modular multiplications on this host: exactly
// rank 0's final combine. Real arithmetic, not a constant.
static double measure_combine_ms(int n_nodes, const mpz_t modulo, gmp_randstate_t rs)
{
    if (n_nodes <= 1) return 0.0;
    mpz_t acc, partial;
    mpz_inits(acc, partial, NULL);
    mpz_urandomm(acc, rs, modulo);
    mpz_urandomm(partial, rs, modulo);
    double t0 = gpu_now_ms();
    for (int i = 0; i < n_nodes - 1; i++) {
        mpz_mul(acc, acc, partial);
        mpz_mod(acc, acc, modulo);
    }
    double dt = gpu_now_ms() - t0;
    mpz_clears(acc, partial, NULL);
    return dt;
}

// read_mpz_from_file prints an error and returns void, so a missing file leaves
// the mpz at zero and the run limps on producing nonsense. Check first and die
// with the path, which is the only thing the operator needs to see.
static void require_file(const char *path, const char *what) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: %s not found: %s\n", what, path);
        fprintf(stderr, "       challenge sets live in data/challenges/k<K>/ --\n"
                        "       pass e.g. --data data/challenges/k32\n");
        exit(1);
    }
    fclose(f);
}

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s --data DIR --size BITS [options]\n"
        "  --k N            smoothness bound 2^N (default 32)\n"
        "  --splits \"...\"   simulated GPU counts (default: 1 2 4 8 16 32 64 128 256 512)\n"
        "  --n-gpus G       physical devices per simulated node (default: all)\n"
        "  --n-runs R       timed repeats per split (default: 5)\n"
        "  --exponent E     root exponent (default 3)\n"
        "  --chal-k K       bound the challenge set was crafted for (default: --k)\n"
        "  --json FILE      write the projection as JSON\n"
        "\nChallenge sets: --data data/challenges/k32\n", p);
}

int main(int argc, char **argv)
{
    const char *data_dir = nullptr, *json_path = nullptr, *primes_path = nullptr;
    // Suffix on the modulus filename, e.g. "_e5" for the exponent-5 moduli that
    // challenge_gen_e5.slurm generates with gcd(5, M-1) = 1. Challenge files are
    // exponent-agnostic and shared, so only the modulus changes. Mirrors
    // --modulo-suffix on the MPI solvers.
    const char *modulo_suffix = "";
    const char *splits_s = "1 2 4 8 16 32 64 128 256 512";
    int size_bits = 0, n_gpus = 0, n_runs = 5, k_bound = 32;
    int chal_k = 0;   // bound the challenge set was crafted for; defaults to k_bound
    unsigned long exponent = 3;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--data")     && i+1<argc) data_dir   = argv[++i];
        else if (!strcmp(argv[i], "--size")     && i+1<argc) size_bits  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k")        && i+1<argc) k_bound    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--modulo-suffix") && i + 1 < argc)
            modulo_suffix = argv[++i];
        else if (!strcmp(argv[i], "--splits")   && i+1<argc) splits_s   = argv[++i];
        else if (!strcmp(argv[i], "--n-gpus")   && i+1<argc) n_gpus     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-runs")   && i+1<argc) n_runs     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exponent") && i+1<argc) exponent   = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--primes")   && i+1<argc) primes_path= argv[++i];
        else if (!strcmp(argv[i], "--chal-k")   && i+1<argc) chal_k     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")     && i+1<argc) json_path  = argv[++i];
    }
    if (!data_dir || size_bits <= 0) { usage(argv[0]); return 1; }
    if (chal_k <= 0) chal_k = k_bound;
    if (k_bound < 17 || k_bound > 32) {
        fprintf(stderr, "ERROR: --k must be 17..32 (got %d)\n", k_bound); return 1;
    }

    std::vector<int> splits;
    { char buf[512]; snprintf(buf, sizeof(buf), "%s", splits_s);
      for (char *t = strtok(buf, " ,"); t; t = strtok(nullptr, " ,")) {
          int v = atoi(t); if (v > 0) splits.push_back(v); } }
    std::sort(splits.begin(), splits.end());
    if (splits.empty()) { fprintf(stderr, "ERROR: no valid --splits\n"); return 1; }

    int avail = 1; cudaGetDeviceCount(&avail);
    if (n_gpus <= 0) n_gpus = avail;
    if (n_gpus > avail) n_gpus = avail;
    const int GPUS_PER_NODE = n_gpus;   // this host stands in for one node

    // Naming follows tools/challenges.c: data/challenges/k<K>/ holds
    // modulo_<K>_<bits>.txt and challenge_<n>_<K>_<bits>.txt. The K in the
    // filename is the bound the challenge was CRAFTED for, which is not
    // necessarily --k (see the note in the sweep script).
    static char cbuf[512], mbuf[512], pbuf[256];
    snprintf(cbuf, sizeof(cbuf), "%s/challenge_0_%d_%d.txt", data_dir, chal_k, size_bits);
    snprintf(mbuf, sizeof(mbuf), "%s/modulo_%d_%d%s.txt", data_dir, chal_k, size_bits, modulo_suffix);
    require_file(mbuf, "modulo");
    require_file(cbuf, "challenge");
    if (!primes_path) { snprintf(pbuf, sizeof(pbuf), "data/k%d_primes.bin", k_bound); primes_path = pbuf; }

    mpz_t challenge, modulo, e_inv, result;
    mpz_inits(challenge, modulo, e_inv, result, NULL);
    read_mpz_from_file(challenge, cbuf);
    read_mpz_from_file(modulo, mbuf);
    { mpz_t phi, e; mpz_inits(phi, e, NULL);
      mpz_sub_ui(phi, modulo, 1); mpz_set_ui(e, exponent);
      if (!mpz_invert(e_inv, e, phi)) { fprintf(stderr, "e not invertible\n"); return 1; }
      mpz_clears(phi, e, NULL); }
    int nlimbs = (int)((mpz_sizeinbase(modulo, 2) + 63) / 64);
    if (nlimbs > MAX_LIMBS) { fprintf(stderr, "modulus too wide\n"); return 1; }

    printf("=== N-GPU scaling: MEASURED, not modelled ===\n");
    printf("k=%d  size=%d bits  physical GPUs per simulated node=%d\n",
           k_bound, size_bits, GPUS_PER_NODE);

    uint64_t cnt64 = 0;
    uint32_t *primes = kf_load_prime_table(&cnt64, primes_path);
    if (!primes) {
        fprintf(stderr, "ERROR: cannot load %s\n"
                        "       build it with: tools/gen_prime_table %d\n", primes_path, k_bound);
        return 1;
    }
    const int64_t nprimes = (int64_t)cnt64;
    const double table_mb = (double)nprimes * 4.0 / 1e6;
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
    const double l2_mb = (double)prop.l2CacheSize / 1e6;
    printf("Table: %ld primes, %.1f MB   Device: %s, L2 %.0f MB\n\n",
           (long)nprimes, table_mb, prop.name, l2_mb);

    // powm baseline — the same reference the whole thesis divides by.
    double baseline_ms = 0.0;
    { mpz_t chk; mpz_init(chk); std::vector<double> t(5);
      for (int i = 0; i < 5; i++) { double t0 = gpu_now_ms();
          mpz_powm(chk, challenge, e_inv, modulo); t[i] = gpu_now_ms() - t0; }
      std::sort(t.begin(), t.end()); baseline_ms = t[2]; mpz_clear(chk);
      printf("powm baseline: %.2f ms (median of 5)\n\n", baseline_ms); }

    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, 12345);

    cbrt_cache_t cache; kf_cbrt_cache_init(&cache);
    bool cache_dirty = false;
    char cpath[600]; snprintf(cpath, sizeof(cpath), "%s.cbrt", mbuf);
    kf_cbrt_cache_load(&cache, cpath);

    uint64_t *h_out = (uint64_t *)malloc(MAX_GPU_FACTORS * sizeof(uint64_t));
    const int THREADS = 256, BLOCKS = 1024;

    struct Row { int N, nodes; double mb, factor, recon, gather, combine, total, speedup; bool resident; };
    std::vector<Row> rows;

    printf("%6s %6s %10s %10s %9s %8s %9s %11s %9s  %s\n",
           "GPUs", "nodes", "MB/GPU", "factor", "l.recon", "gather", "combine",
           "total", "speedup", "notes");
    printf("--------------------------------------------------------------------------------------------\n");

    for (int N : splits) {
        // A simulated deployment of N GPUs has ceil(N/4) nodes, and this host
        // stands in for ONE of them. When N is smaller than the card count we
        // simply use fewer cards -- N=1 must still be measurable on a 4-GPU box,
        // and it is the row every other row is judged against.
        const int use_gpus  = (N < GPUS_PER_NODE) ? N : GPUS_PER_NODE;
        const int n_nodes   = (N + use_gpus - 1) / use_gpus;
        const int64_t shard = nprimes / N;
        if (shard <= 0) continue;

        // ---- upload this node's shards (one per physical GPU in use) ----
        std::vector<GpuDev> devs((size_t)use_gpus);
        for (int g = 0; g < use_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            int64_t start = (int64_t)g * shard;
            CUDA_CHECK(cudaMalloc(&devs[g].d_primes, (size_t)shard * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(devs[g].d_primes, primes + start,
                                  (size_t)shard * sizeof(uint32_t), cudaMemcpyHostToDevice));
            devs[g].n_slice = shard;
            CUDA_CHECK(cudaMalloc(&devs[g].d_out_factors, MAX_GPU_FACTORS * sizeof(uint64_t)));
            CUDA_CHECK(cudaMalloc(&devs[g].d_out_count, sizeof(int)));
            CUDA_CHECK(cudaEventCreate(&devs[g].ev0));
            CUDA_CHECK(cudaEventCreate(&devs[g].ev1));
            CUDA_CHECK(cudaMemcpyToSymbol(d_nlimbs, &nlimbs, sizeof(int)));
        }

        uint64_t y_limbs[MAX_LIMBS] = {}; size_t xc = 0;
        mpz_export(y_limbs, &xc, -1, sizeof(uint64_t), 0, 0, challenge);

        double best_factor = 1e30, best_recon = 1e30;
        for (int run = 0; run < n_runs; run++) {
            // ---- MEASURED: factor, max over this node's GPUs ----
            for (int g = 0; g < use_gpus; g++) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMemcpyToSymbol(d_y_limbs, y_limbs, sizeof(uint64_t) * MAX_LIMBS));
                CUDA_CHECK(cudaMemset(devs[g].d_out_count, 0, sizeof(int)));
                CUDA_CHECK(cudaEventRecord(devs[g].ev0));
                scale_trial_divide_kernel<<<BLOCKS, THREADS>>>(
                    devs[g].d_primes, devs[g].n_slice,
                    devs[g].d_out_factors, devs[g].d_out_count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(devs[g].ev1));
            }
            float fmax = 0;
            for (int g = 0; g < use_gpus; g++) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaEventSynchronize(devs[g].ev1));
                float t = 0; CUDA_CHECK(cudaEventElapsedTime(&t, devs[g].ev0, devs[g].ev1));
                if (t > fmax) fmax = t;
            }

            // ---- MEASURED: this node's local reconstruction ----
            kfactor_list_t fl; kfl_init(&fl);
            mpz_t cof; mpz_init_set(cof, challenge);
            for (int g = 0; g < use_gpus; g++) {
                CUDA_CHECK(cudaSetDevice(g));
                int hc = 0;
                CUDA_CHECK(cudaMemcpy(&hc, devs[g].d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
                if (hc <= 0) continue;
                if (hc > MAX_GPU_FACTORS) hc = MAX_GPU_FACTORS;
                CUDA_CHECK(cudaMemcpy(h_out, devs[g].d_out_factors,
                                      (size_t)hc * sizeof(uint64_t), cudaMemcpyDeviceToHost));
                for (int i = 0; i < hc; i++) {
                    unsigned long p = (unsigned long)h_out[i]; int e = 0;
                    while (mpz_divisible_ui_p(cof, p)) { mpz_divexact_ui(cof, cof, p); e++; }
                    if (e > 0) kfl_append_ul(&fl, p, e);
                }
            }
            double t0 = gpu_now_ms();
            kf_compute_cbrt_from_factors_omp(result, &fl, modulo, &cache, &cache_dirty, exponent);
            double rec = gpu_now_ms() - t0;
            kfl_free(&fl); mpz_clear(cof);

            if (fmax < best_factor) best_factor = fmax;
            if (rec  < best_recon)  best_recon  = rec;
        }

        // ---- MEASURED: rank 0's serial combine over n_nodes partials ----
        double combine = measure_combine_ms(n_nodes, modulo, rs);
        double gather  = gather_us(n_nodes) / 1000.0;
        double total   = best_factor + best_recon + gather + combine;
        double mb      = table_mb / N;
        bool   res     = (mb <= l2_mb);

        rows.push_back({N, n_nodes, mb, best_factor, best_recon, gather, combine,
                        total, baseline_ms / total, res});
        printf("%6d %6d %10.2f %10.2f %9.2f %8.3f %9.2f %11.2f %8.2fx  %s%s\n",
               N, n_nodes, mb, best_factor, best_recon, gather, combine, total,
               baseline_ms / total, res ? "L2-resident " : "",
               combine > best_factor ? "combine-bound" : "");

        for (int g = 0; g < use_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            cudaFree(devs[g].d_primes); cudaFree(devs[g].d_out_factors);
            cudaFree(devs[g].d_out_count);
            cudaEventDestroy(devs[g].ev0); cudaEventDestroy(devs[g].ev1);
        }
    }

    // ---- Verdict ----
    if (!rows.empty()) {
        const Row *peak = &rows[0];
        for (auto &r : rows) if (r.speedup > peak->speedup) peak = &r;
        printf("\n");
        printf("peak speedup      : %.2fx at N = %d GPUs (%d nodes)\n",
               peak->speedup, peak->N, peak->nodes);
        const Row *first1 = nullptr, *firstCPU = nullptr;
        for (auto &r : rows) {
            if (!first1   && r.speedup >= 1.0)   first1   = &r;
            if (!firstCPU && r.speedup >= 52.38) firstCPU = &r;
        }
        if (first1)   printf("clears baseline   : N = %d\n", first1->N);
        if (firstCPU) printf("matches AION N64  : N = %d  (52.38x, k32 table)\n", firstCPU->N);
        else          printf("matches AION N64  : not reached in this sweep\n");
        if (peak != &rows.back())
            printf("\nBeyond N = %d more GPUs make it SLOWER: the serial combine grows with\n"
                   "node count while every other term shrinks. That is the same ceiling the\n"
                   "CPU cluster hits, and it is measured here, not assumed.\n", peak->N);
    }

    if (json_path) {
        FILE *f = fopen(json_path, "w");
        if (f) {
            fprintf(f, "{\n  \"metadata\": {\n");
            fprintf(f, "    \"algorithm\": \"gpu_scale_sim\",\n");
            fprintf(f, "    \"smoothness_bound_bits\": %d,\n", k_bound);
            fprintf(f, "    \"challenge_bits\": %d,\n", size_bits);
            fprintf(f, "    \"exponent\": %lu,\n", exponent);
            fprintf(f, "    \"gpus_per_node\": %d,\n", GPUS_PER_NODE);
            fprintf(f, "    \"device\": \"%s\",\n", prop.name);
            fprintf(f, "    \"l2_mb\": %.1f,\n", l2_mb);
            fprintf(f, "    \"table_mb\": %.2f,\n", table_mb);
            fprintf(f, "    \"powm_baseline_ms\": %.4f,\n", baseline_ms);
            fprintf(f, "    \"note\": \"factor/local_recon/combine MEASURED; gather modelled from AION\"\n");
            fprintf(f, "  },\n  \"rows\": [\n");
            for (size_t i = 0; i < rows.size(); i++) {
                const Row &r = rows[i];
                fprintf(f, "    {\"n_gpus\": %d, \"n_nodes\": %d, \"mb_per_gpu\": %.3f, "
                           "\"factor_ms\": %.4f, \"local_recon_ms\": %.4f, \"gather_ms\": %.4f, "
                           "\"combine_ms\": %.4f, \"total_ms\": %.4f, \"speedup\": %.4f, "
                           "\"l2_resident\": %s}%s\n",
                        r.N, r.nodes, r.mb, r.factor, r.recon, r.gather, r.combine,
                        r.total, r.speedup, r.resident ? "true" : "false",
                        i + 1 < rows.size() ? "," : "");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
            printf("\nJSON -> %s\n", json_path);
        }
    }

    if (cache_dirty) kf_cbrt_cache_save(&cache, cpath);
    kf_cbrt_cache_clear(&cache);
    free(h_out); free(primes);
    gmp_randclear(rs);
    mpz_clears(challenge, modulo, e_inv, result, NULL);
    return 0;
}
