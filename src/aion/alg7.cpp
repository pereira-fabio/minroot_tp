// =============================================================================
// alg7.cpp — Algorithm 7 solver: parallel smoothness test
// (renamed from k_alg7_mpi.cpp 2026-06-11 to match the alg2_*/alg3/alg4 convention)
//
// Biryukov et al. CRYPTO 2024, §3.5 — "Parallel Smoothness Test"
// Paper params: T=13, #CPU=2^29, M=2^40, speedup=20×
//
// Core idea — inverted sieve over R candidates simultaneously:
//   OLD (alg3): one candidate  y   = challenge * r^e mod M  per blinding attempt
//   NEW (alg7): R candidates   y_r = challenge + r * M       for r in [0, R)
//
// For each prime q < B, all r where q | y_r satisfy:
//   r ≡ -(challenge mod q) * (M mod q)^{-1}  (mod q)
//
// The sieve marks log2(q) into log_sum[r] for those r values.
// MPI_Allreduce (SUM) combines log_sum across all ranks.
// Candidates where log_sum[r] >= log2(y_r) - SLACK are investigated.
// Reconstruction path is identical to Algorithm 3.
//
// Communication pattern:
//   Rank 0: broadcast challenge (same as alg3)
//   All ranks: inverted sieve into local log_sum[R]
//   MPI_Allreduce(SUM): combine log_sum → global log_sum
//   All ranks: find candidate r (deterministic scan)
//   All ranks: sieve y_r in their prime slice (Phase 2 — same as alg3)
//   ksc_alg3_broadcast_cofactor + ksc_gather_and_combine_cbrt (same as alg3)
//
// Note on pre-generated challenges:
//   Benchmark challenges are constructed to be B-smooth (challenge = product of
//   small primes). For such challenges, y_0 = challenge is always the smooth
//   candidate (found_r = 0). Phase 1 adds overhead vs alg3. For random challenges
//   (where blinding is needed), alg7 tests R candidates per pass and outperforms
//   alg3's sequential blinding strategy.
//
// Usage:
//   alg7_mpi [challenge] [modulo] [--method sieve|powm] [--R N]
//   alg7_mpi --all [--data DIR] [--R N]
// =============================================================================

#include "common.h"
#include "factorization.h"
#include "solver_common.h"
#include "sizes.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// run_method_t defined in k_common.h

#define ALG7_R_DEFAULT   (1u << 20) // 2^20 candidates — 4 MB log_sum per rank
#define ALG7_SLACK       65.0f      // allow cofactor up to 2^65 (alg3-style)
#define ALG7_MAX_ATTEMPTS 16        // outer blinding retries if no hit found
// kprint_stat() defined in k_common.h

// ---------------------------------------------------------------------------
// mod_inv_u64 — extended Euclidean: a^{-1} mod m
// Returns 0 if a == 0 or gcd(a, m) != 1.
// ---------------------------------------------------------------------------

static uint64_t mod_inv_u64(uint64_t a, uint64_t m)
{
    if (a == 0 || m <= 1) return 0;
    int64_t old_r = (int64_t)a, r = (int64_t)m;
    int64_t old_s = 1,          s = 0;
    while (r != 0) {
        int64_t q = old_r / r;
        int64_t tr = r;  r = old_r - q * r;  old_r = tr;
        int64_t ts = s;  s = old_s - q * s;  old_s = ts;
    }
    if (old_r != 1) return 0;
    return old_s < 0 ? (uint64_t)(old_s + (int64_t)m) : (uint64_t)old_s;
}

// ---------------------------------------------------------------------------
// solve_one
// ---------------------------------------------------------------------------

// run_one — benchmark one challenge, RETURN the results (no JSON write).
// Mirrors the alg2/alg3/alg4 run_one so the caller can merge multiple challenges.
static benchmark_results_t run_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method,
    unsigned long exponent,
    uint32_t R,
    const char *cbrt_dir = KSC_CBRT_DIR)
{
    if (rank == 0) {
        printf("\n=== %s ===\n", challenge_file);
        fflush(stdout);
    }

    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    if (!ksc_broadcast_challenge_modulo(challenge, modulo,
                                         challenge_file, modulo_file, rank)) {
        mpz_clears(challenge, modulo, result, NULL);
        return benchmark_results_t{};
    }

    ksc_print_challenge_header(challenge, modulo, rank, nranks);
    if (rank == 0) {
        printf("R (candidates/pass): %u = 2^%.1f  log_sum_size: %.1f MB\n",
               R, log2((double)R), (double)R * sizeof(float) / 1e6);
        fflush(stdout);
    }

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    mpz_t e_inv;
    mpz_init(e_inv);
    ksc_compute_e_inv(e_inv, modulo, exponent);

    // cbrt cache — alg7-specific suffix to avoid conflicts with other solvers
    char cbrt_cache_path[KSC_CBRT_PATH_MAX];
    ksc_make_cbrt_cache_path(cbrt_cache_path, sizeof(cbrt_cache_path),
                             cbrt_dir, modulo_file, ".cbrt_alg7", rank);

    cbrt_cache_t local_cbrt_cache;
    kf_cbrt_cache_init(&local_cbrt_cache);
    bool cbrt_cache_dirty = false;
    kf_cbrt_cache_load(&local_cbrt_cache, cbrt_cache_path);

    // ---- Load prime slice (always: Phase 1 inverted sieve needs actual prime values) ----
    const char *k32_path = ksc_ensure_k32_primes(sp_count, sp, rank);
    uint64_t slice_count = 0;
    uint32_t *prime_slice = kf_load_prime_slice(
        &slice_count, k32_path,
        (uint32_t)my_lo,
        (uint32_t)(my_hi > (uint64_t)UINT32_MAX ? UINT32_MAX : my_hi));

    if (!prime_slice) {
        fprintf(stderr, "[rank %d] Failed to load prime slice [%llu, %llu]\n",
                rank, (unsigned long long)my_lo, (unsigned long long)my_hi);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0) {
        printf("Prime slice loaded: %llu primes in [%llu, %llu]\n",
               (unsigned long long)slice_count,
               (unsigned long long)my_lo,
               (unsigned long long)my_hi);
        fflush(stdout);
    }

    // Precompute cbrt cache for smooth prime factors in this challenge (untimed).
    ksc_ensure_cbrt_cache(
        &local_cbrt_cache, &cbrt_cache_dirty, cbrt_cache_path,
        challenge, modulo, e_inv,
        my_lo, my_hi, sp_count, sp,
        (method == METHOD_TABLE) ? prime_slice : NULL,
        (method == METHOD_TABLE) ? (int64_t)slice_count : 0,
        false, rank);

    // ---- Precompute M_mod_q and M_inv_q for all primes in slice ----
    // M_mod_q[i] = modulo mod prime_slice[i]; M_inv_q[i] = (M mod q)^{-1} mod q.
    // Both depend only on M, not on the challenge, so they are computed ONCE
    // here instead of recomputing mod_inv_u64 in every inverted-sieve pass
    // (Fix C: hoist the per-prime modular inverse out of the hot loop).
    uint32_t *M_mod_q = (uint32_t *)malloc(slice_count * sizeof(uint32_t));
    uint64_t *M_inv_q = (uint64_t *)malloc(slice_count * sizeof(uint64_t));
    if ((!M_mod_q || !M_inv_q) && slice_count > 0) {
        fprintf(stderr, "[rank %d] OOM for M_mod_q/M_inv_q (%llu entries)\n",
                rank, (unsigned long long)slice_count);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0) {
        printf("Precomputing M mod q and (M mod q)^-1 for %llu primes...",
               (unsigned long long)slice_count);
        fflush(stdout);
    }
    // mpz_tdiv_ui only reads `modulo` — safe to call from many threads at once.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (uint64_t i = 0; i < slice_count; i++) {
        uint32_t mm = (uint32_t)mpz_tdiv_ui(modulo, (unsigned long)prime_slice[i]);
        M_mod_q[i] = mm;
        M_inv_q[i] = mod_inv_u64((uint64_t)mm, (uint64_t)prime_slice[i]);
    }
    if (rank == 0) { printf(" done.\n"); fflush(stdout); }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Blinding table (outer retries if no hit in first R candidates) ----
    const int BLIND_TABLE_SIZE = ALG7_MAX_ATTEMPTS - 1;
    blinding_pair_t *blind_table = NULL;
    if (rank == 0) {
        blind_table = kf_blinding_table_alloc(BLIND_TABLE_SIZE, modulo, exponent);
        if (!blind_table) {
            fprintf(stderr, "[rank 0] Failed to allocate blinding table\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // ---- log_sum array ----
    float *log_sum = (float *)calloc(R, sizeof(float));
    if (!log_sum) {
        fprintf(stderr, "[rank %d] OOM for log_sum[%u]\n", rank, R);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    mpz_t blinded_challenge, cofactor;
    mpz_init(blinded_challenge);
    mpz_init_set_ui(cofactor, 1);

    bench_accum_t fac_acc = {0, 0}; // inverted sieve + allreduce + candidate scan
    bench_accum_t loc_acc    = {0, 0}; // local cbrt (phase 2)
    bench_accum_t gat_acc    = {0, 0}; // MPI gather (phase 2)
    bench_accum_t com_acc    = {0, 0}; // combine + unblind (phase 2)
    bench_accum_t recon_acc  = {0, 0}; // phase 2 total (fac2 + loc + gat + com)

    int n_threads_json = 1;
#ifdef _OPENMP
    n_threads_json = omp_get_max_threads();
#endif

    benchmark_results_t json_results;
    const char *mname = (method == METHOD_TABLE)  ? "table"
                        : (method == METHOD_POWM) ? "powm"
                                                  : "sieve";
    ksc_init_json_results(&json_results, 32, mname,
                          nranks, n_threads_json, challenge, rank);
    json_results.config.algorithm = "alg7";
    json_results.config.exponent  = exponent;

    // Per-challenge context passed to ksc_reconstruct_y (Phase 2).
    ksc_recon_ctx_t recon_ctx;
    recon_ctx.modulo           = &modulo;
    recon_ctx.cbrt_bytes       = cbrt_bytes;
    recon_ctx.e_inv            = &e_inv;
    recon_ctx.cache            = &local_cbrt_cache;
    recon_ctx.cache_dirty      = &cbrt_cache_dirty;
    recon_ctx.my_lo            = my_lo;
    recon_ctx.my_hi            = my_hi;
    recon_ctx.sp_count         = sp_count;
    recon_ctx.sp               = sp;
    recon_ctx.blind_table      = blind_table;
    recon_ctx.blind_table_size = BLIND_TABLE_SIZE;
    recon_ctx.rank             = rank;
    recon_ctx.nranks           = nranks;
    recon_ctx.method           = method;
    recon_ctx.prime_slice      = prime_slice;
    recon_ctx.slice_count      = (int64_t)slice_count;

    size_t blinded_nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
    unsigned char *blinded_buf = (unsigned char *)malloc(blinded_nbytes);
    float log2_M = (float)mpz_sizeinbase(modulo, 2);

    // ---- Benchmark loop ----
    // Single timed run for this challenge (bench starts at 1 so the warm-run
    // recording gates `bench > 0` fire; the old per-challenge repeat / --bench is
    // removed and multiple challenges are merged by the caller).
    for (int bench = 1; bench < 2 && method != METHOD_POWM; bench++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_phase1_start = std::chrono::high_resolution_clock::now();

        // ================================================================
        // Phase 1 — inverted sieve over R candidates
        // ================================================================
        int found_r = -1;
        int attempt = 0;

        for (attempt = 0; attempt < ALG7_MAX_ATTEMPTS && found_r < 0; attempt++)
        {
            // -- Rank 0 computes blinded challenge, exports to buffer --
            if (rank == 0) {
                if (attempt == 0) {
                    mpz_set(blinded_challenge, challenge);
                } else {
                    int idx = attempt - 1;
                    if (idx < BLIND_TABLE_SIZE) {
                        mpz_mul(blinded_challenge, challenge,
                                blind_table[idx].r_blind);
                        mpz_mod(blinded_challenge, blinded_challenge, modulo);
                    } else {
                        mpz_set(blinded_challenge, challenge);
                    }
                }
                memset(blinded_buf, 0, blinded_nbytes);
                size_t written = 0;
                unsigned char *tmp = (unsigned char *)mpz_export(
                    NULL, &written, 1, 1, 0, 0, blinded_challenge);
                if (tmp && written > 0 && written <= blinded_nbytes) {
                    memcpy(blinded_buf + blinded_nbytes - written, tmp, written);
                    free(tmp);
                }
            }
            MPI_Bcast(blinded_buf, (int)blinded_nbytes, MPI_UNSIGNED_CHAR,
                      0, MPI_COMM_WORLD);
            if (rank != 0)
                mpz_import(blinded_challenge, blinded_nbytes, 1, 1, 0, 0,
                           blinded_buf);

            // -- Inverted sieve (Fix C: OpenMP-parallel over primes) --
            // For each prime q in slice: mark r values where q | (blinded + r*M)
            // r ≡ -(blinded mod q) * (M mod q)^{-1}  (mod q)
            //
            // The expensive per-prime arithmetic (mpz_tdiv_ui on the multi-kbit
            // blinded challenge + the 128-bit product) runs fully in parallel —
            // this is the dominant cost and was previously single-threaded.
            // mpz_tdiv_ui only READS blinded_challenge, so concurrent calls are
            // safe.  Only the scattered accumulation into the shared log_sum[r]
            // needs synchronisation, done with a lightweight #pragma omp atomic;
            // contention is low because marking is a small fraction of the
            // per-prime work and writes are spread across R cells.
            memset(log_sum, 0, R * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 2048)
#endif
            for (uint64_t i = 0; i < slice_count; i++) {
                uint64_t q = (uint64_t)prime_slice[i];
                uint64_t M_mod = (uint64_t)M_mod_q[i];
                if (M_mod == 0) continue; // q | M (can't happen: M is prime, q < M)

                uint64_t M_inv = M_inv_q[i]; // precomputed once (Fix C)
                if (M_inv == 0) continue;

                uint64_t x_mod_q = (uint64_t)mpz_tdiv_ui(blinded_challenge, q);

                // r_start = (-(x_mod_q * M_inv) mod q + q) mod q
                uint64_t r_start;
                if (x_mod_q == 0) {
                    r_start = 0;
                } else {
                    __uint128_t prod = (__uint128_t)x_mod_q * M_inv;
                    uint64_t    pm   = (uint64_t)(prod % q);
                    r_start = (pm == 0) ? 0 : (q - pm);
                }

                float lq = log2f((float)q);
                for (uint64_t r = r_start; r < (uint64_t)R; r += q) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                    log_sum[r] += lq;
                }
            }

            // -- MPI_Allreduce: sum log_sum across all ranks --
            MPI_Allreduce(MPI_IN_PLACE, log_sum, (int)R, MPI_FLOAT,
                          MPI_SUM, MPI_COMM_WORLD);

            // -- Scan for smooth candidate --
            // Threshold: log2(y_r) - SLACK where y_r = blinded + r*M
            // blinded < M, so log2(y_r) <= log2_M + log2f(r+1)
            for (uint32_t r = 0; r < R; r++) {
                float log2_y_r = (r == 0)
                    ? log2_M
                    : (log2_M + log2f((float)(r + 1)));
                if (log_sum[r] >= log2_y_r - ALG7_SLACK) {
                    found_r = (int)r;
                    break;
                }
            }

            if (rank == 0 && bench == 0) {
                printf("Phase 1 attempt %d: %s\n", attempt,
                       found_r >= 0 ? "candidate found" : "no candidate");
                fflush(stdout);
            }
        }

        auto t_phase1_end = std::chrono::high_resolution_clock::now();
        double phase1_dt = std::chrono::duration<double, std::milli>(
            t_phase1_end - t_phase1_start).count();

        if (rank == 0 && bench > 0) {
            accum_record(&fac_acc, phase1_dt);
            json_results.timing.factor_times.push_back(phase1_dt);
            json_results.timing.blinding_attempts.push_back(attempt);
        }

        if (found_r < 0) {
            if (rank == 0)
                fprintf(stderr, "[bench %d] No smooth candidate in %d attempts — skipped\n",
                        bench, ALG7_MAX_ATTEMPTS);
            if (bench > 0) {
                json_results.timing.recon_times.push_back(0.0);
                json_results.timing.total_times.push_back(phase1_dt);
                json_results.hits.hit_type.push_back(0);
                json_results.hits.cofactor_bits.push_back(0);
                json_results.hits.medium_factor_count.push_back(0);
            }
            continue;
        }

        // ================================================================
        // Phase 2 — direct sieve of y_{found_r} + reconstruction
        // ================================================================
        // y_r = blinded_challenge + found_r * modulo  (computed independently
        // by all ranks since blinded_challenge and found_r are identical everywhere)
        mpz_t y_r;
        mpz_init(y_r);
        mpz_set(y_r, blinded_challenge);
        if (found_r > 0)
            mpz_addmul_ui(y_r, modulo, (unsigned long)found_r);

        if (rank == 0 && bench == 0) {
            printf("Candidate r=%d  y_r=%zu bits\n",
                   found_r, mpz_sizeinbase(y_r, 2));
            fflush(stdout);
        }

        double fac2_dt = 0.0, recon_dt = 0.0;
        int complete = ksc_reconstruct_y(
            result, cofactor,
            y_r, attempt, bench,
            &loc_acc, &gat_acc, &com_acc,
            &fac2_dt, &recon_dt,
            &json_results, &recon_ctx);
        mpz_clear(y_r);

        if (rank == 0 && bench == 0) {
            int is_smooth = (complete && mpz_cmp_ui(cofactor, 1) == 0);
            printf("Cofactor: %s\n",
                   complete ? (is_smooth ? "B-smooth" : "almost-smooth")
                            : "incomplete — skipped");
            fflush(stdout);
        }

        if (!complete) {
            if (bench > 0) {
                json_results.timing.recon_times.push_back(fac2_dt);
                json_results.timing.total_times.push_back(phase1_dt + fac2_dt);
                json_results.hits.hit_type.push_back(0);
                json_results.hits.cofactor_bits.push_back(0);
                json_results.hits.medium_factor_count.push_back(0);
            }
            continue;
        }

        if (bench > 0 && rank == 0) {
            json_results.hits.hit_type.push_back(mpz_cmp_ui(cofactor, 1) == 0 ? 1 : 2);
            json_results.hits.cofactor_bits.push_back((int)mpz_sizeinbase(cofactor, 2));
            json_results.hits.medium_factor_count.push_back(
                mpz_cmp_ui(cofactor, 1) == 0 ? 0 : 1);
        }

        if (rank == 0) {
            accum_record(&recon_acc, recon_dt);
            if (bench > 0) {
                json_results.timing.recon_times.push_back(recon_dt);
                json_results.timing.total_times.push_back(phase1_dt + recon_dt);
            }
        }
    }

    free(blinded_buf);

    // ---- Timing report (one timed run per challenge) ----
    if (rank == 0) {
        printf("\n");
        printf("Phase 1 (inv sieve): %.3f ms\n", fac_acc.sum);
        printf("Phase 2 (recon):     %.3f ms\n", recon_acc.sum);
        printf("Total:               %.3f ms\n", fac_acc.sum + recon_acc.sum);

        ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                         &recon_acc, &fac_acc,
                                         rank, exponent);
        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_eth_root(result, challenge_mod, modulo, exponent);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    // ---- Cleanup ----
    free(log_sum);
    free(M_mod_q);
    free(M_inv_q);
    free(prime_slice);
    mpz_clear(blinded_challenge);
    mpz_clear(cofactor);
    if (rank == 0)
        kf_blinding_table_free(blind_table, BLIND_TABLE_SIZE);
    if (cbrt_cache_dirty)
        kf_cbrt_cache_save(&local_cbrt_cache, cbrt_cache_path);
    kf_cbrt_cache_clear(&local_cbrt_cache);
    mpz_clear(e_inv);
    mpz_clears(challenge, modulo, result, NULL);

    return json_results;
}

// solve_one — backward-compatible wrapper: run_one + write JSON (single challenge).
static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method,
    unsigned long exponent,
    uint32_t R,
    const char *cbrt_dir = KSC_CBRT_DIR,
    const char *json_output_file = nullptr)
{
    benchmark_results_t r = run_one(rank, nranks, challenge_file, modulo_file,
                                    sp_count, sp, my_lo, my_hi, method, exponent, R, cbrt_dir);
    if (rank == 0 && json_output_file && json_output_file[0] != '\0')
        write_benchmark_json(json_output_file, r);
    return 1;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Error: MPI_THREAD_FUNNELED not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char *challenge_file = NULL;
    const char *modulo_file    = NULL;
    const char *data_dir       = "data/challenges/alg3"; // reuse alg3 challenges
    const char *json_file      = NULL;
    const char *cbrt_dir       = KSC_CBRT_DIR; // per-node isolation via --cbrt-dir
    int run_all        = 0;
    int n_challenges   = 1;   // >1: challenge_0=cold warmup, 1..N-1 merged (with --all)
    run_method_t method = METHOD_SIEVE;
    unsigned long exponent = 3;
    uint32_t R = ALG7_R_DEFAULT;

    int positional = 0;
    for (int argi = 1; argi < argc; argi++) {
        if (strcmp(argv[argi], "--all") == 0) {
            run_all = 1;
        } else if (strcmp(argv[argi], "--data") == 0 && argi + 1 < argc) {
            data_dir = argv[++argi];
        } else if (strcmp(argv[argi], "--method") == 0 && argi + 1 < argc) {
            const char *m = argv[++argi];
            if (strcmp(m, "sieve") == 0)       method = METHOD_SIEVE;
            else if (strcmp(m, "table") == 0)  method = METHOD_TABLE;
            else if (strcmp(m, "powm") == 0)   method = METHOD_POWM;
            else {
                if (rank == 0)
                    fprintf(stderr, "Unknown --method '%s' (use sieve, table, or powm)\n", m);
                MPI_Finalize(); return 1;
            }
        } else if (strcmp(argv[argi], "--json") == 0 && argi + 1 < argc) {
            json_file = argv[++argi];
        } else if (strcmp(argv[argi], "--cbrt-dir") == 0 && argi + 1 < argc) {
            cbrt_dir = argv[++argi];
        } else if (strcmp(argv[argi], "--n-challenges") == 0 && argi + 1 < argc) {
            n_challenges = atoi(argv[++argi]);
            if (n_challenges < 1) n_challenges = 1;
        } else if (strcmp(argv[argi], "--exponent") == 0 && argi + 1 < argc) {
            exponent = (unsigned long)atol(argv[++argi]);
        } else if (strcmp(argv[argi], "--R") == 0 && argi + 1 < argc) {
            long rv = atol(argv[++argi]);
            if (rv < 1024 || rv > (1 << 24)) {
                if (rank == 0)
                    fprintf(stderr, "--R must be in [1024, 2^24]\n");
                MPI_Finalize(); return 1;
            }
            R = (uint32_t)rv;
        } else if (argv[argi][0] != '-') {
            if (positional == 0)      { challenge_file = argv[argi]; positional++; }
            else if (positional == 1) { modulo_file    = argv[argi]; positional++; }
            else {
                if (rank == 0) fprintf(stderr, "Unexpected argument '%s'\n", argv[argi]);
                MPI_Finalize(); return 1;
            }
        } else {
            if (rank == 0) fprintf(stderr, "Unknown option '%s'\n", argv[argi]);
            MPI_Finalize(); return 1;
        }
    }

    if (!run_all && (!challenge_file || !modulo_file)) {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: alg7_mpi <challenge> <modulo> [--method sieve|table|powm] "
                    "[--R N] [--cbrt-dir DIR]\n"
                    "       alg7_mpi --all [--data DIR] [--method sieve|table|powm] "
                    "[--R N] [--n-challenges N] [--cbrt-dir DIR] [--json PREFIX]\n"
                    "       (--all writes PREFIX_<size>.json; --n-challenges N: "
                    "challenge_0=cold warmup, 1..N-1 merged)\n");
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        printf("=== alg7_mpi (Algorithm 7 — parallel smoothness test) ===\n");
        printf("MPI ranks:      %d\n", nranks);
#ifdef _OPENMP
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
#endif
        printf("Method:         %s\n",
               method == METHOD_SIEVE   ? "sieve"
               : method == METHOD_TABLE ? "table"
                                        : "powm");
        printf("R (candidates): %u (2^%.1f)\n", R, log2((double)R));
        if (run_all)          printf("Mode:           --all\n");
        fflush(stdout);
    }

    int sp_count;
    unsigned *sp = kf_generate_small_primes(&sp_count);

    // Partition sieve range [65537, 2^32] across ranks — same partition as alg3
    const uint64_t RANGE_LO   = 65537ULL;
    const uint64_t RANGE_HI   = (uint64_t)UINT32_MAX;
    const uint64_t range_size = RANGE_HI - RANGE_LO + 1;

    uint64_t my_lo = RANGE_LO + (range_size * (uint64_t)rank)       / (uint64_t)nranks;
    uint64_t my_hi = RANGE_LO + (range_size * (uint64_t)(rank + 1)) / (uint64_t)nranks - 1;
    if (rank == nranks - 1) my_hi = RANGE_HI;

    if (rank == 0) {
        printf("Prime range/rank: ~%.1fM primes\n\n",
               (double)range_size / nranks / 1e6);
        fflush(stdout);
    }

    if (run_all) {
        const char *sizes[] = { KSC_CANONICAL_SIZES_LIST };
        int nsizes = KSC_N_CANONICAL_SIZES;
        for (int s = 0; s < nsizes; s++) {
            char cf[512], mf[512], jf[600];
            const char *jf_ptr = NULL;
            snprintf(mf, sizeof(mf), "%s/modulo_alg3_%s.txt", data_dir, sizes[s]);
            if (json_file) {
                snprintf(jf, sizeof(jf), "%s_%s.json", json_file, sizes[s]);
                jf_ptr = jf;
            }
            if (n_challenges == 1) {
                snprintf(cf, sizeof(cf), "%s/challenge_0_alg3_%s.txt", data_dir, sizes[s]);
                solve_one(rank, nranks, cf, mf,
                          sp_count, sp, my_lo, my_hi, method, exponent, R, cbrt_dir, jf_ptr);
            } else {
                // Cold run: challenge_0 warms OS page cache and cbrt cache; saved as cold_run.
                snprintf(cf, sizeof(cf), "%s/challenge_0_alg3_%s.txt", data_dir, sizes[s]);
                benchmark_results_t cold = run_one(rank, nranks, cf, mf,
                                                   sp_count, sp, my_lo, my_hi,
                                                   method, exponent, R, cbrt_dir);

                // Warm challenges 1..n_challenges-1: merge into one combined result.
                benchmark_results_t combined;
                bool has_data = false;
                for (int ci = 1; ci < n_challenges; ci++) {
                    char cfi[512];
                    snprintf(cfi, sizeof(cfi), "%s/challenge_%d_alg3_%s.txt",
                             data_dir, ci, sizes[s]);
                    int exists = 0;
                    if (rank == 0) {
                        FILE *fck = fopen(cfi, "r");
                        if (fck) { fclose(fck); exists = 1; }
                    }
                    MPI_Bcast(&exists, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (!exists) {
                        if (rank == 0)
                            printf("  [SKIP] challenge_%d not found for size %s\n",
                                   ci, sizes[s]);
                        continue;
                    }
                    benchmark_results_t r = run_one(rank, nranks, cfi, mf,
                                                    sp_count, sp, my_lo, my_hi,
                                                    method, exponent, R, cbrt_dir);
                    if (!has_data) { combined = std::move(r); has_data = true; }
                    else           { merge_benchmark_results(combined, r); }
                }
                if (has_data && rank == 0 && jf_ptr) {
                    set_cold_run(combined, cold);
                    write_benchmark_json(jf_ptr, combined);
                }
            }
        }
    } else {
        solve_one(rank, nranks, challenge_file, modulo_file,
                  sp_count, sp, my_lo, my_hi, method, exponent, R, cbrt_dir, json_file);
    }

    free(sp);
    MPI_Finalize();
    return 0;
}


// Edit: 2026-06-12
