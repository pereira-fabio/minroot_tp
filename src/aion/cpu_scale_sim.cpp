// =============================================================================
// cpu_scale_sim.cpp — what would AION do with more nodes than we can allocate?
//
// The CPU counterpart of src/iris/gpu_scale_sim_cuda.cu, and it exists for the
// same reason: the 64-node ceiling is an account limit, so the rank counts at
// which k32/k35 turn over cannot be run. Until now the CPU projection was a
// power-law FIT to the seven measured rank counts; this tool MEASURES three of
// its four terms at any rank count, on one node.
//
// Why that is legitimate. A solve is
//
//     total(N) = factor(N) + local_recon(N) + gather(N) + combine(N)
//
// and three of those four are per-rank quantities that do not depend on the
// other ranks existing:
//
//   factor(N)      one rank owns [65537,2^k] sliced N ways (the SAME split
//                  alg2_k32/k35.cpp compute, including k35's straddle of 2^32
//                  across two prime tables) and runs the SAME kernel on it. The
//                  slice shrinking into cache as N grows -- the effect that
//                  makes the cluster beat the GPU at all -- is reproduced
//                  exactly, because it is a property of the slice, not of the job.
//   local_recon(N) that rank roots and multiplies the ~803/N factors it found.
//   combine(N)     rank 0 folds N-1 partial products together: N-1 dependent
//                  full-width mpz_mul+mpz_mod. Pure arithmetic on one core, so
//                  it is measured for any N with no cluster at all. This is the
//                  term that creates the turnover, and it was previously the
//                  one derived rather than measured.
//
// gather(N) is the exception and stays modelled -- a single-node job cannot
// measure a network. It is reported separately and never folded in silently.
// At 24576 b it is <1% of the total; at paper-scale moduli it is NOT small
// (~50% at 256 b), so every JSON row carries total_no_gather_ms/speedup_no_gather
// beside the modelled figures -- use those where the model is doing real work.
//
// Run it under the production layout (2 ranks/node, 32 threads each) so the
// memory contention a real rank sees is present:
//     srun -N1 -n2 -c32 ./cpu_scale_sim --size 24576 --k 32 --method table \
//          --splits 1,2,4,...,1024 --json out.json
// Every rank times the SAME slice; the reported figure is the max, matching
// how the solvers report a phase.
//
// Correctness is not claimed here and no root is produced: this measures the
// cost of the phases, not the answer. Use the real solver for that.
// =============================================================================

#include "common.h"
#include "solver_common.h"
#include "factorization.h"
#include "sizes.h"

#include <mpi.h>
#include <gmp.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

// Modelled gather, fitted to the AION k32 sweep (see THESIS tab:cpu-fits).
// Depends on rank count only -- it does NOT scale with |M|, which is why it is
// reported apart from the measured terms rather than blended into them.
static double gather_model_ms(int nranks) { return 0.0113 * pow((double)nranks, 0.357); }

struct SimRow {
    int      nranks;
    uint64_t my_lo, my_hi;
    double   slice_mb;
    int64_t  slice_primes;
    double   factor_ms, local_recon_ms, combine_ms, gather_ms;
    int      factors_found;
};

// One rank's factoring slice of an nranks-way job -- the split of alg2_k32.cpp.
static void rank0_slice(int nranks, uint64_t range_hi, uint64_t *lo, uint64_t *hi)
{
    const uint64_t RANGE_LO = 65537ULL;
    uint64_t span = range_hi - RANGE_LO + 1;
    *lo = RANGE_LO;
    *hi = RANGE_LO + span / (uint64_t)nranks - 1;
    if (nranks == 1) *hi = range_hi;
}

// combine(N): rank 0's N-1 dependent full-width modular multiplications.
// Measured, not modelled -- the operands are real |M|-bit residues.
static double measure_combine_ms(int nranks, const mpz_t modulo, gmp_randstate_t rs)
{
    if (nranks <= 1) return 0.0;
    size_t bits = mpz_sizeinbase(modulo, 2);
    mpz_t acc, partial;
    mpz_inits(acc, partial, NULL);
    mpz_urandomb(acc, rs, bits);
    mpz_mod(acc, acc, modulo);
    mpz_urandomb(partial, rs, bits);
    mpz_mod(partial, partial, modulo);

    double t0 = MPI_Wtime();
    for (int i = 0; i < nranks - 1; i++) {
        mpz_mul(acc, acc, partial);
        mpz_mod(acc, acc, modulo);
    }
    double dt = (MPI_Wtime() - t0) * 1e3;
    mpz_clears(acc, partial, NULL);
    return dt;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const char *data_dir = "data/challenges/k32";
    const char *json_out = NULL;
    const char *method   = "table";
    const char *splits_s = "1,2,4,8,16,32,64,128,192,256,384,512,768,1024";
    const char *primes_file = "data/tables/k32_primes.bin";
    const char *primes_upper = K35U_DEFAULT_PATH;   // only read when k > 32
    const char *cbrt_dir = "data/cbrt";
    int  size_bits = 24576, kbits = 32, reps = 3;
    unsigned long exponent = 3;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--data")   && i+1 < argc) data_dir    = argv[++i];
        else if (!strcmp(argv[i], "--json")   && i+1 < argc) json_out    = argv[++i];
        else if (!strcmp(argv[i], "--method") && i+1 < argc) method      = argv[++i];
        else if (!strcmp(argv[i], "--splits") && i+1 < argc) splits_s    = argv[++i];
        else if (!strcmp(argv[i], "--primes") && i+1 < argc) primes_file = argv[++i];
        else if (!strcmp(argv[i], "--primes-upper") && i+1 < argc) primes_upper = argv[++i];
        else if (!strcmp(argv[i], "--cbrt-dir") && i+1 < argc) cbrt_dir = argv[++i];
        else if (!strcmp(argv[i], "--size")   && i+1 < argc) size_bits   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k")      && i+1 < argc) kbits       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")   && i+1 < argc) reps        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exponent") && i+1 < argc) exponent  = atol(argv[++i]);
        else { if (!rank) fprintf(stderr, "unknown option '%s'\n", argv[i]); MPI_Finalize(); return 1; }
    }
    if (kbits < 17 || kbits > 35) {
        if (!rank) fprintf(stderr, "--k must be in [17,35]\n");
        MPI_Finalize(); return 1;
    }

    std::vector<int> splits;
    for (const char *p = splits_s; *p; ) {
        splits.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    std::sort(splits.begin(), splits.end());

    char cf[512], mf[512];
    snprintf(cf, sizeof(cf), "%s/challenge_0_%d_%d.txt", data_dir, kbits, size_bits);
    snprintf(mf, sizeof(mf), "%s/modulo_%d_%d.txt",      data_dir, kbits, size_bits);

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);
    if (!ksc_broadcast_challenge_modulo(challenge, modulo, cf, mf, rank)) {
        if (!rank) fprintf(stderr, "cannot read %s / %s\n", cf, mf);
        mpz_clears(challenge, modulo, NULL); MPI_Finalize(); return 1;
    }

    int sp_count = 0;
    unsigned *sp = kf_generate_small_primes(&sp_count);

    // The merged per-variant cbrt cache, loaded once and shared by every split,
    // exactly as the solvers do under Plan A. This is not an optimisation: with
    // a cold cache each cbrt(p) is a full-width modexp costing about as much as
    // the entire sequential baseline, so local_recon would measure cache-filling
    // rather than reconstruction, and the job would never finish.
    cbrt_cache_t cache; kf_cbrt_cache_init(&cache);
    {
        char p[KSC_CBRT_PATH_MAX];
        ksc_shared_cbrt_path(p, sizeof(p), cbrt_dir, mf);
        if (!kf_cbrt_cache_load(&cache, p)) {
            if (!rank) fprintf(stderr,
                "FATAL: no merged cbrt cache at %s.\n"
                "  local_recon would measure modexps, not reconstruction. Build it with\n"
                "  analysis/merge_cbrt_cache.py after a normal solver run at this size.\n", p);
            mpz_clears(challenge, modulo, NULL); MPI_Finalize(); return 1;
        }
        if (!rank) printf("cbrt: loaded %s (%d entries)\n", p, cache.count);
    }
    gmp_randstate_t rs; gmp_randinit_mt(rs); gmp_randseed_ui(rs, 20260812UL);

    // The baseline this is all divided into: the sequential inverse root.
    double powm_ms = 0.0;
    if (rank == 0) {
        mpz_t b, out; mpz_inits(b, out, NULL);
        ksc_compute_e_inv(b, modulo, exponent);
        double t0 = MPI_Wtime();
        mpz_powm(out, challenge, b, modulo);
        powm_ms = (MPI_Wtime() - t0) * 1e3;
        mpz_clears(b, out, NULL);
        printf("=== cpu_scale_sim ===\n");
        printf("size=%d bits  k=%d  method=%s  ranks/node=%d  threads=%d\n",
               size_bits, kbits, method, nprocs, omp_get_max_threads());
        printf("powm baseline: %.3f ms\n\n", powm_ms);
        printf("%8s %12s %10s %12s %12s %12s %12s %10s\n",
               "ranks", "slice_MB", "factors", "factor_ms", "l.recon_ms",
               "combine_ms", "total_ms", "speedup");
        fflush(stdout);
    }
    MPI_Bcast(&powm_ms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Exactly the bounds the solvers use: alg2_k32 stops at UINT32_MAX,
    // alg2_k35 overshoots 2^35 by 1024 to catch nextprime overflow.
    const uint64_t range_hi = (kbits <= 32) ? ((kbits == 32) ? (uint64_t)UINT32_MAX
                                                             : ((1ULL << kbits) - 1))
                                            : ((1ULL << kbits) + 1024ULL);
    std::vector<SimRow> rows;

    for (int N : splits) {
        SimRow r{}; r.nranks = N;
        rank0_slice(N, range_hi, &r.my_lo, &r.my_hi);

        // A rank's range can straddle 2^32, so k35 needs both tables --- the
        // same split alg2_k35.cpp performs. Omitting the upper one would
        // silently report a seventh of k35's work as all of it.
        uint32_t *slice   = NULL; uint64_t slice_n  = 0;
        uint64_t *uslice  = NULL; int64_t  uslice_n = 0;
        if (!strcmp(method, "table")) {
            if (r.my_lo <= (uint64_t)UINT32_MAX) {
                uint64_t lo32 = std::min(r.my_hi, (uint64_t)UINT32_MAX);
                slice = kf_load_prime_slice(&slice_n, primes_file,
                                            (uint32_t)r.my_lo, (uint32_t)lo32);
                if (!slice) { if (!rank) fprintf(stderr, "cannot load %s\n", primes_file); break; }
            }
            if (r.my_hi > (uint64_t)UINT32_MAX) {
                uint64_t ulo = std::max(r.my_lo, (uint64_t)UINT32_MAX + 1);
                uslice = kf_load_prime_slice_u64(&uslice_n, primes_upper, ulo, r.my_hi);
                if (!uslice) { if (!rank) fprintf(stderr, "cannot load %s\n", primes_upper); break; }
            }
        }
        r.slice_primes = (int64_t)slice_n + uslice_n;
        r.slice_mb     = (slice_n * 4.0 + uslice_n * 8.0) / 1e6;

        // ---- factor + local_recon, best of `reps` (warm) ----
        double best_f = 1e300, best_r = 1e300;
        for (int rep = 0; rep < reps; rep++) {
            kfactor_list_t found; kfl_init(&found);
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            if (!strcmp(method, "table")) {
                if (slice)  kf_trial_divide_table(challenge, slice, (int64_t)slice_n, &found);
                if (uslice) kf_trial_divide_table_u64(challenge, uslice, uslice_n, &found);
            } else {
                kf_trial_divide_sieve(challenge, r.my_lo, r.my_hi, sp_count, sp, &found);
            }
            double t1 = MPI_Wtime();

            bool dirty = false;
            mpz_t partial; mpz_init(partial);
            kf_compute_cbrt_from_factors_omp(partial, &found, modulo, &cache, &dirty, exponent);
            double t2 = MPI_Wtime();

            best_f = std::min(best_f, (t1 - t0) * 1e3);
            best_r = std::min(best_r, (t2 - t1) * 1e3);
            r.factors_found = found.count;
            mpz_clear(partial); kfl_free(&found);
        }
        if (slice) free(slice);
        if (uslice) free(uslice);

        r.factor_ms = best_f; r.local_recon_ms = best_r;
        r.combine_ms = measure_combine_ms(N, modulo, rs);
        r.gather_ms  = gather_model_ms(N);

        // A phase is what the slowest rank spent on it, as the solvers report.
        double mine[3] = { r.factor_ms, r.local_recon_ms, r.combine_ms }, mx[3];
        MPI_Reduce(mine, mx, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            r.factor_ms = mx[0]; r.local_recon_ms = mx[1]; r.combine_ms = mx[2];
            double total = r.factor_ms + r.local_recon_ms + r.combine_ms + r.gather_ms;
            printf("%8d %12.2f %10d %12.3f %12.3f %12.3f %12.3f %9.2fx\n",
                   N, r.slice_mb, r.factors_found, r.factor_ms, r.local_recon_ms,
                   r.combine_ms, total, total > 0 ? powm_ms / total : 0.0);
            fflush(stdout);
            rows.push_back(r);
        }
    }

    if (rank == 0 && json_out) {
        FILE *f = fopen(json_out, "w");
        if (f) {
            fprintf(f, "{\n  \"metadata\": {\n"
                       "    \"algorithm\": \"cpu_scale_sim\",\n"
                       "    \"method\": \"%s\",\n    \"smoothness_bound_bits\": %d,\n"
                       "    \"challenge_bits\": %d,\n    \"exponent\": %lu,\n"
                       "    \"ranks_per_node\": %d,\n    \"threads_per_rank\": %d,\n"
                       "    \"powm_baseline_ms\": %.4f,\n"
                       "    \"note\": \"factor/local_recon/combine MEASURED; gather modelled "
                       "(0.0113*N^0.357 ms, rank-count only, does not scale with |M|)\"\n  },\n",
                    method, kbits, size_bits, exponent, nprocs, omp_get_max_threads(), powm_ms);
            fprintf(f, "  \"rows\": [\n");
            for (size_t i = 0; i < rows.size(); i++) {
                const SimRow &r = rows[i];
                double tot  = r.factor_ms + r.local_recon_ms + r.combine_ms + r.gather_ms;
                double totg = r.factor_ms + r.local_recon_ms + r.combine_ms;
                fprintf(f, "    { \"nranks\": %d, \"nodes\": %d, \"slice_mb\": %.3f, "
                           "\"slice_primes\": %lld, \"factors_found\": %d, "
                           "\"factor_ms\": %.4f, \"local_recon_ms\": %.4f, "
                           "\"combine_ms\": %.4f, \"gather_ms\": %.4f, "
                           "\"total_ms\": %.4f, \"speedup\": %.4f, "
                           "\"total_no_gather_ms\": %.4f, \"speedup_no_gather\": %.4f }%s\n",
                        r.nranks, (r.nranks + 1) / 2, r.slice_mb,
                        (long long)r.slice_primes, r.factors_found,
                        r.factor_ms, r.local_recon_ms, r.combine_ms, r.gather_ms,
                        tot,  tot  > 0 ? powm_ms / tot  : 0.0,
                        totg, totg > 0 ? powm_ms / totg : 0.0,
                        i + 1 < rows.size() ? "," : "");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
            printf("\nwrote %s\n", json_out);
        }
    }

    kf_cbrt_cache_clear(&cache);
    gmp_randclear(rs);
    free(sp);
    mpz_clears(challenge, modulo, NULL);
    MPI_Finalize();
    return 0;
}

// Edit: 2026-08-12
