// =============================================================================
// alg_crt.cpp — CRT-based inverse-root solver (advisor todo.md "CRT method")
//
// Computes x = c^b mod M with b = e^-1 mod (M-1) (the inverse e-th root, the same
// value run_powm_baseline computes), WITHOUT one big mpz_powm mod M. The exponent
// b is applied by alpha-ary exponentiation whose per-step "raise to alpha" is done
// in parallel over small primes via the CRT (crt_common). No factorization, no
// smoothness, no blinding — works for ANY challenge.
//
// Structure mirrors alg7.cpp (MPI, --all sweep, all-warm-run merge) minus the sieve.
//
// Usage:
//   alg_crt <challenge> <modulo> [--alpha A] [--exponent e]
//   alg_crt --all [--data DIR] [--alpha A] [--exponent e] [--n-challenges N] [--json PREFIX]
// =============================================================================

#include "common.h"
#include "solver_common.h"
#include "crt_common.h"
#include "sizes.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>

#define ALG_CRT_ALPHA_DEFAULT 16

// run_one — benchmark one challenge, RETURN the results (no JSON write).
static benchmark_results_t run_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    unsigned long exponent, unsigned long alpha)
{
    if (rank == 0) { printf("\n=== %s ===\n", challenge_file); fflush(stdout); }

    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    if (!ksc_broadcast_challenge_modulo(challenge, modulo,
                                        challenge_file, modulo_file, rank)) {
        mpz_clears(challenge, modulo, result, NULL);
        return benchmark_results_t{};
    }
    ksc_print_challenge_header(challenge, modulo, rank, nranks);

    // b = e^-1 mod (M-1) — the inverse-root exponent.
    mpz_t b; mpz_init(b);
    ksc_compute_e_inv(b, modulo, exponent);

    // CRT context (deterministic; every rank builds it independently). Not part
    // of the timed solve, but measured and reported: it builds k ~ alpha*|M|/30
    // CRT basis elements K_i of alpha*|M| bits each, so it is not free.
    crt_ctx_t ctx;
    double setup_ms = MPI_Wtime();
    crt_ctx_init(&ctx, modulo, alpha);
    setup_ms = (MPI_Wtime() - setup_ms) * 1e3;
    int lo = (int)((long)ctx.k * rank       / nranks);
    int hi = (int)((long)ctx.k * (rank + 1) / nranks);
    if (rank == 0) {
        printf("CRT: alpha=%lu  primes=%d  prod=%zu bits  ~%d primes/rank  b=%zu bits\n",
               alpha, ctx.k, mpz_sizeinbase(ctx.prod, 2),
               nranks ? ctx.k / nranks : ctx.k, mpz_sizeinbase(b, 2));
        fflush(stdout);
    }

    // base-alpha digits of b (least-significant first).
    std::vector<unsigned long> digits;
    {
        mpz_t bb; mpz_init_set(bb, b);
        while (mpz_sgn(bb) > 0) {
            digits.push_back(mpz_tdiv_ui(bb, alpha));
            mpz_tdiv_q_ui(bb, bb, alpha);
        }
        mpz_clear(bb);
        if (digits.empty()) digits.push_back(0);
    }

    // digit table g[d] = c^d mod M, d in [0, alpha).
    mpz_t cmod; mpz_init(cmod); mpz_mod(cmod, challenge, modulo);
    mpz_t *g = (mpz_t *)malloc((size_t)alpha * sizeof(mpz_t));
    for (unsigned long d = 0; d < alpha; d++) {
        mpz_init(g[d]);
        if (d == 0) mpz_set_ui(g[d], 1);
        else { mpz_mul(g[d], g[d - 1], cmod); mpz_mod(g[d], g[d], modulo); }
    }

    bench_accum_t fac_acc   = {0, 0};
    bench_accum_t recon_acc = {0, 0};

    benchmark_results_t json_results;
    ksc_init_json_results(&json_results, 0, "crt", nranks, 1, challenge, rank);
    json_results.config.algorithm = "alg_crt";
    json_results.config.exponent  = exponent;

    // ---- timed alpha-ary exponentiation: result = c^b mod M ----
    crt_profile_t prof;
    crt_profile_reset(&prof);
    double digitmul_ms = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    mpz_set(result, g[digits.back()]);              // c^(most-significant digit)
    for (int j = (int)digits.size() - 2; j >= 0; j--) {
        crt_step(result, &ctx, modulo, lo, hi, rank, nranks, &prof); // result <- result^alpha
        double dm = MPI_Wtime();
        mpz_mul(result, result, g[digits[j]]);                // * c^(digit_j)
        mpz_mod(result, result, modulo);
        digitmul_ms += (MPI_Wtime() - dm) * 1e3;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- phase profile: reduce across ranks (max = critical path, mean = share) ----
    // Reported for BOTH because the two differ diagnostically: `combine` is
    // rank-0-only work, so max >> mean; `gather` on non-root ranks absorbs load
    // imbalance as wait time, so a large gather max with a small local spread
    // means ranks are arriving at the collective at different times.
    {
        const char *names[8] = { "local", "localmod", "pack", "gather",
                                 "combine", "bcast", "digit_mul", "setup" };
        double mine[8] = { prof.local_ms, prof.localmod_ms, prof.pack_ms,
                           prof.gather_ms, prof.combine_ms, prof.bcast_ms,
                           digitmul_ms, setup_ms };
        double mx[8], sm[8];
        MPI_Reduce(mine, mx, 8, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(mine, sm, 8, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            size_t mbits = mpz_sizeinbase(modulo, 2);
            double acc = 0.0;
            for (int i = 0; i < 7; i++) acc += mx[i];   // setup excluded from solve
            printf("\n--- CRT phase profile  (|M|=%zu bits, alpha=%lu, %d ranks, "
                   "%ld steps, k=%d primes, |prod|=%zu bits) ---\n",
                   mbits, alpha, nranks, prof.steps, ctx.k,
                   mpz_sizeinbase(ctx.prod, 2));
            printf("  %-10s %12s %12s %9s %14s\n",
                   "phase", "max_ms", "mean_ms", "%crit", "us/step");
            for (int i = 0; i < 7; i++)
                printf("  %-10s %12.1f %12.1f %8.1f%% %14.2f\n",
                       names[i], mx[i], sm[i] / nranks,
                       acc > 0 ? mx[i] / acc * 100.0 : 0.0,
                       prof.steps ? mx[i] * 1e3 / (double)prof.steps : 0.0);
            printf("  %-10s %12.1f %12.1f %8s %14s   (untimed, excluded above)\n",
                   names[7], mx[7], sm[7] / nranks, "-", "-");
            printf("  %-10s %12.1f  (wall %.1f ms; %.1f%% accounted)\n",
                   "SUM", acc, dt, dt > 0 ? acc / dt * 100.0 : 0.0);

            // Machine-parseable, one line per phase, for the analysis pipeline.
            for (int i = 0; i < 8; i++)
                printf("CRTPROF|%zu|%lu|%d|%s|%.4f|%.4f\n",
                       mbits, alpha, nranks, names[i], mx[i], sm[i] / nranks);
            fflush(stdout);

            // Feed the three phases the shared JSON schema already carries.
            json_results.timing.local_cbrt_times.push_back(mx[0]);
            json_results.timing.gather_times.push_back(mx[3]);
            json_results.timing.combine_times.push_back(mx[4]);
        }
    }

    if (rank == 0) {
        accum_record(&fac_acc, dt);
        json_results.timing.factor_times.push_back(dt);
        json_results.timing.recon_times.push_back(0.0);
        json_results.timing.total_times.push_back(dt);
        json_results.timing.blinding_attempts.push_back(0);
        json_results.hits.hit_type.push_back(1);
        json_results.hits.cofactor_bits.push_back(0);
        json_results.hits.medium_factor_count.push_back(0);

        printf("\nCRT compute: %.3f ms (%zu alpha-steps of ^%lu)\n",
               fac_acc.sum, digits.size() - 1, alpha);
        ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                         &recon_acc, &fac_acc, rank, exponent);
        mpz_t challenge_mod; mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_eth_root(result, challenge_mod, modulo, exponent);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    for (unsigned long d = 0; d < alpha; d++) mpz_clear(g[d]);
    free(g);
    mpz_clear(cmod);
    crt_ctx_free(&ctx);
    mpz_clear(b);
    mpz_clears(challenge, modulo, result, NULL);
    return json_results;
}

static void solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    unsigned long exponent, unsigned long alpha,
    const char *json_output_file)
{
    benchmark_results_t r = run_one(rank, nranks, challenge_file, modulo_file,
                                    exponent, alpha);
    if (rank == 0 && json_output_file && json_output_file[0] != '\0')
        write_benchmark_json(json_output_file, r);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char *challenge_file = NULL;
    const char *modulo_file    = NULL;
    const char *data_dir       = "data/challenges/alg3"; // reuse alg3 challenges (any c works)
    const char *json_file      = NULL;
    int run_all        = 0;
    int n_challenges   = 1;
    unsigned long exponent = 3;
    unsigned long alpha    = ALG_CRT_ALPHA_DEFAULT;

    int positional = 0;
    for (int argi = 1; argi < argc; argi++) {
        if (strcmp(argv[argi], "--all") == 0) {
            run_all = 1;
        } else if (strcmp(argv[argi], "--data") == 0 && argi + 1 < argc) {
            data_dir = argv[++argi];
        } else if (strcmp(argv[argi], "--json") == 0 && argi + 1 < argc) {
            json_file = argv[++argi];
        } else if (strcmp(argv[argi], "--alpha") == 0 && argi + 1 < argc) {
            long a = atol(argv[++argi]);
            if (a < 2) { if (rank == 0) fprintf(stderr, "--alpha must be >= 2\n"); MPI_Finalize(); return 1; }
            alpha = (unsigned long)a;
        } else if (strcmp(argv[argi], "--exponent") == 0 && argi + 1 < argc) {
            exponent = (unsigned long)atol(argv[++argi]);
        } else if (strcmp(argv[argi], "--n-challenges") == 0 && argi + 1 < argc) {
            n_challenges = atoi(argv[++argi]);
            if (n_challenges < 1) n_challenges = 1;
        } else if (argv[argi][0] != '-') {
            if (positional == 0)      { challenge_file = argv[argi]; positional++; }
            else if (positional == 1) { modulo_file    = argv[argi]; positional++; }
            else { if (rank == 0) fprintf(stderr, "Unexpected argument '%s'\n", argv[argi]); MPI_Finalize(); return 1; }
        } else {
            if (rank == 0) fprintf(stderr, "Unknown option '%s'\n", argv[argi]);
            MPI_Finalize(); return 1;
        }
    }

    if (!run_all && (!challenge_file || !modulo_file)) {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: alg_crt <challenge> <modulo> [--alpha A] [--exponent e]\n"
                    "       alg_crt --all [--data DIR] [--alpha A] [--exponent e] "
                    "[--n-challenges N] [--json PREFIX]\n");
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        printf("=== alg_crt (CRT-based inverse-root exponentiation) ===\n");
        printf("MPI ranks:  %d\n", nranks);
        printf("alpha:      %lu\n", alpha);
        printf("exponent:   %lu\n", exponent);
        if (run_all) printf("Mode:       --all\n");
        fflush(stdout);
    }

    if (run_all) {
        const char *sizes[] = { KSC_CANONICAL_SIZES_LIST };
        int nsizes = KSC_N_CANONICAL_SIZES;
        for (int s = 0; s < nsizes; s++) {
            char mf[512], jf[600];
            const char *jf_ptr = NULL;
            snprintf(mf, sizeof(mf), "%s/modulo_alg3_%s.txt", data_dir, sizes[s]);
            if (json_file) { snprintf(jf, sizeof(jf), "%s_%s.json", json_file, sizes[s]); jf_ptr = jf; }

            if (n_challenges == 1) {
                char cf[512];
                snprintf(cf, sizeof(cf), "%s/challenge_0_alg3_%s.txt", data_dir, sizes[s]);
                solve_one(rank, nranks, cf, mf, exponent, alpha, jf_ptr);
            } else {
                // Plan A (2026-07-17): all N challenges are timed warm runs — no
                // separate cold run.
                benchmark_results_t combined;
                bool has_data = false;
                for (int ci = 0; ci < n_challenges; ci++) {
                    char cfi[512];
                    snprintf(cfi, sizeof(cfi), "%s/challenge_%d_alg3_%s.txt", data_dir, ci, sizes[s]);
                    int exists = 0;
                    if (rank == 0) { FILE *fck = fopen(cfi, "r"); if (fck) { fclose(fck); exists = 1; } }
                    MPI_Bcast(&exists, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (!exists) continue;
                    benchmark_results_t r = run_one(rank, nranks, cfi, mf, exponent, alpha);
                    if (!has_data) { combined = std::move(r); has_data = true; }
                    else           { merge_benchmark_results(combined, r); }
                }
                if (has_data && rank == 0 && jf_ptr) {
                    write_benchmark_json(jf_ptr, combined);
                }
            }
        }
    } else {
        solve_one(rank, nranks, challenge_file, modulo_file, exponent, alpha, json_file);
    }

    MPI_Finalize();
    return 0;
}


// Edit: 2026-07-28
