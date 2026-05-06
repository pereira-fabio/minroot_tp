// =============================================================================
// k64_mpi.cpp — Paper-faithful cube root solver for 64-bit prime factors
//
// Implements Algorithm 2 (Biryukov et al., CRYPTO 2024) with B = 2^64.
//
// Unlike k16/k32 which PARTITION a prime range across ranks, k64 uses the
// paper's intended MPI model: each rank is an independent "group" that holds
// one unique blinding factor r and runs Pollard's rho on its own blinded value
// y = challenge * r^3 mod M.  The first rank to fully factor its y wins.
//
// Why no trial division:
//   Trial division up to B = 2^64 requires testing ~3e17 primes — infeasible.
//   Pollard's rho finds factors in O(p^{1/4}) per factor p, which is fast for
//   smooth numbers: O((2^64)^{1/4}) = O(2^16) = 65536 iterations per factor.
//
// Parallelism:
//   MPI level: R = nranks independent groups, each with distinct r.
//              More ranks = more blinding factors tried per unit time.
//   OMP level: Within each rank, kf_factor_recursive runs threads that race
//              with different Pollard c-values.  More threads = faster rho.
//
// Communication per bench iteration:
//   MPI_Bcast        challenge (rank 0 -> all, startup only)
//   [factoring: all ranks independent, no communication]
//   MPI_Allreduce    MAX(success_token) to find the winner
//   MPI_Bcast        winner broadcasts cbrt(challenge) bytes
//
// Build:
//   mpicxx -O3 -std=c++17 -fopenmp -Wall
//       src/k64_mpi.cpp src/k_solver_common.cpp
//       src/k_common.cpp src/k_factorization.cpp
//       -lgmp -o k64_mpi
// =============================================================================

#include "k_common.h"
#include "k_factorization.h"
#include "k_solver_common.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// How many blinding rounds before declaring failure.
// Round r: rank i tries pair (r * nranks + i), so nranks distinct r-values
// are probed per round.  32 rounds * 512 ranks = 16384 distinct r-values max.
#define K64_ROUNDS 32
#define K64_MAX_PAIRS (K64_ROUNDS * 512) // generous upper bound

typedef enum
{
    METHOD_RHO,
    METHOD_POWM
} run_method_t;

// ---------------------------------------------------------------------------
// cbrt reconstruction
//
// result = prod_i  cbrt(p_i)^{e_i}  mod M
// Looks up in cache; computes via mpz_powm on miss and stores it.
// ---------------------------------------------------------------------------

static bool cbrt_from_factors(
    mpz_t result,
    const kfactor_list_t *factors,
    const mpz_t modulo,
    cbrt_cache_t *cache,
    bool *dirty)
{
    mpz_set_ui(result, 1);

    mpz_t e_inv, m1;
    mpz_inits(e_inv, m1, NULL);
    mpz_sub_ui(m1, modulo, 1);
    mpz_set_ui(e_inv, 3);
    if (!mpz_invert(e_inv, e_inv, m1))
    {
        mpz_clears(e_inv, m1, NULL);
        return false;
    }

    mpz_t p_mpz, cbrt_p, term;
    mpz_inits(p_mpz, cbrt_p, term, NULL);

    for (int i = 0; i < factors->count; i++)
    {
        unsigned long p = mpz_get_ui(factors->items[i].prime);
        int e = factors->items[i].exponent;

        int idx = kf_cbrt_cache_find(cache, p);
        if (idx >= 0)
        {
            if (e == 1)
                mpz_set(term, cache->items[idx].cbrt);
            else
                mpz_powm_ui(term, cache->items[idx].cbrt,
                            (unsigned long)e, modulo);
        }
        else
        {
            mpz_set(p_mpz, factors->items[i].prime);
            mpz_powm(cbrt_p, p_mpz, e_inv, modulo);
            kf_cbrt_cache_append(cache, p, cbrt_p);
            *dirty = true;
            if (e == 1)
                mpz_set(term, cbrt_p);
            else
                mpz_powm_ui(term, cbrt_p, (unsigned long)e, modulo);
        }
        mpz_mul(result, result, term);
        mpz_mod(result, result, modulo);
    }
    mpz_clears(e_inv, m1, p_mpz, cbrt_p, term, NULL);
    return true;
}

// ---------------------------------------------------------------------------
// factor_and_check
//
// Wrapper around kf_factor_challenge that also verifies completeness:
// checks that product(factors) == y.  Returns true iff y was fully factored.
// ---------------------------------------------------------------------------

static bool factor_and_check(const mpz_t y, kfactor_list_t *factors,
                             unsigned long c_start)
{
    // kf_factor_challenge calls kf_factor_recursive internally but does not
    // expose c_start.  To let each rank explore a different c-value region,
    // call kf_factor_recursive directly and consolidate manually.
    kfactor_list_t raw;
    kfl_init(&raw);
    kf_factor_recursive(y, &raw, c_start);

    // Consolidate duplicates -> (prime, exponent) pairs
    for (int i = 0; i < raw.count; i++)
    {
        if (mpz_cmp_ui(raw.items[i].prime, 0) == 0)
            continue;
        int exp = 1;
        for (int j = i + 1; j < raw.count; j++)
            if (mpz_cmp(raw.items[i].prime, raw.items[j].prime) == 0)
            {
                exp++;
                mpz_set_ui(raw.items[j].prime, 0);
            }
        kfl_append_mpz(factors, raw.items[i].prime, exp);
    }
    kfl_free(&raw);

    // Verify: product of (prime^exponent) must equal y
    mpz_t prod;
    mpz_init_set_ui(prod, 1);
    for (int i = 0; i < factors->count; i++)
        for (int e = 0; e < factors->items[i].exponent; e++)
            mpz_mul(prod, prod, factors->items[i].prime);
    bool ok = (mpz_cmp(prod, y) == 0);
    mpz_clear(prod);
    return ok;
}

// ---------------------------------------------------------------------------
// solve_one
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count, run_method_t method,
    const char *json_file)
{
    if (rank == 0)
    {
        printf("\n=== %s ===\n", challenge_file);
        fflush(stdout);
    }

    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    if (!ksc_broadcast_challenge_modulo(challenge, modulo,
                                        challenge_file, modulo_file, rank))
    {
        mpz_clears(challenge, modulo, result, NULL);
        return 0;
    }

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    if (rank == 0)
    {
        printf("Challenge: %zu bits  Modulo: %zu bits\n",
               mpz_sizeinbase(challenge, 2), mpz_sizeinbase(modulo, 2));
        printf("Ranks: %d  Threads/rank: %d  "
               "Total smoothness probes/round: %d\n",
               nranks, n_threads, nranks);
        printf("Each rank probes y = challenge * r^3 mod M with its own r.\n");
        printf("OMP threads within each rank race Pollard c-values.\n\n");
        fflush(stdout);
    }

    // ── Shared cbrt cache ─────────────────────────────────────────────────────
    cbrt_cache_t cache;
    kf_cbrt_cache_init(&cache);
    bool dirty = false;

    char cache_path[280];
    {
        const char *bn = modulo_file ? modulo_file : "modulo";
        const char *sl = strrchr(bn, '/');
        const char *fn = sl ? sl + 1 : bn;
        char stem[256];
        strncpy(stem, fn, sizeof(stem) - 1);
        stem[sizeof(stem) - 1] = '\0';
        size_t bl = strlen(stem);
        if (bl >= 4 && !strcmp(stem + bl - 4, ".txt"))
            stem[bl - 4] = '\0';
        snprintf(cache_path, sizeof(cache_path), "%s/%s.cbrt",
                 KSC_CBRT_DIR, stem);
    }
    if (kf_cbrt_cache_load(&cache, cache_path) && rank == 0)
        printf("[rank 0] cbrt cache: %d entries from %s\n\n",
               cache.count, cache_path);
    else if (rank == 0)
        printf("[rank 0] cbrt cache: empty — will build during runs\n\n");

    // ── Blinding table ────────────────────────────────────────────────────────
    // All ranks allocate the same table from the same GMP RNG seed so that
    // pair[i] is identical on every rank.  Rank r uses pair (round*nranks + r).
    int total_pairs = K64_ROUNDS * nranks;
    blinding_pair_t *pairs = kf_blinding_table_alloc(total_pairs, modulo);
    if (!pairs)
    {
        fprintf(stderr, "[rank %d] blinding alloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t res_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    // ── JSON init ─────────────────────────────────────────────────────────────
    benchmark_results_t json_res;
    ksc_init_json_results(&json_res, 64, bench_count,
                          method == METHOD_POWM ? "powm" : "rho",
                          nranks, n_threads, challenge, rank);

    bench_accum_t rho_acc = {0, 0}, total_acc = {0, 0};

    // ── Warmup (2 silent Pollard runs) ────────────────────────────────────────
    if (method == METHOD_RHO)
    {
        for (int wu = 0; wu < 2; wu++)
        {
            mpz_t y;
            mpz_init(y);
            if (rank == 0)
                mpz_set(y, challenge);
            else
            {
                mpz_mul(y, challenge, pairs[rank].r3);
                mpz_mod(y, y, modulo);
            }
            auto wt0 = std::chrono::high_resolution_clock::now();
            kfactor_list_t wf;
            kfl_init(&wf);
            kf_factor_recursive(y, &wf, (unsigned long)rank * 1000003UL + 1);
            kfl_free(&wf);
            auto wt1 = std::chrono::high_resolution_clock::now();
            if (rank == 0)
                json_res.warmup_total_ms.push_back(
                    std::chrono::duration<double, std::milli>(wt1 - wt0).count());
            mpz_clear(y);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0)
            printf("Warmup complete (2 runs)\n\n");
    }

    // ── Benchmark loop ────────────────────────────────────────────────────────
    for (int bench = 0; bench < bench_count && method == METHOD_RHO; bench++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        int winning_rank = -1;
        int winning_pair = -1;
        int winning_factors = 0;
        double win_rho_ms = 0.0;
        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double local_rho_ms = 0.0;

        for (int round = 0; round < K64_ROUNDS && winning_rank < 0; round++)
        {
            int pidx = round * nranks + rank; // this rank's pair for this round

            // ---- Blind ----
            mpz_t y;
            mpz_init(y);
            if (round == 0 && rank == 0)
                mpz_set(y, challenge); // r = 1: direct, no blinding
            else
            {
                mpz_mul(y, challenge, pairs[pidx].r3);
                mpz_mod(y, y, modulo);
            }

            // ---- Pollard's rho ----
            // Each rank uses c_start offset by rank*prime to avoid correlation.
            // kf_factor_recursive spawns OMP threads internally.
            auto t_rho0 = std::chrono::high_resolution_clock::now();
            kfactor_list_t factors;
            kfl_init(&factors);
            bool ok = factor_and_check(y, &factors,
                                       (unsigned long)rank * 1000003UL +
                                           (unsigned long)round * 999983UL + 1);
            auto t_rho1 = std::chrono::high_resolution_clock::now();
            local_rho_ms +=
                std::chrono::duration<double, std::milli>(t_rho1 - t_rho0).count();

            // ---- Compute local cbrt if successful ----
            if (ok)
            {
                winning_factors = factors.count;
                mpz_t cbrt_y;
                mpz_init(cbrt_y);
                cbrt_from_factors(cbrt_y, &factors, modulo, &cache, &dirty);
                // Unblind: cbrt(challenge) = cbrt(y) * r^{-1} mod M
                if (round == 0 && rank == 0)
                    mpz_set(local_cbrt, cbrt_y);
                else
                {
                    mpz_mul(local_cbrt, cbrt_y, pairs[pidx].r_inv);
                    mpz_mod(local_cbrt, local_cbrt, modulo);
                }
                mpz_clear(cbrt_y);
            }
            kfl_free(&factors);
            mpz_clear(y);

            // ---- MPI race: find lowest-index successful rank ----
            // Token = rank+1 if succeeded, 0 otherwise.
            // MAX gives the highest rank that succeeded; MIN(nonzero) gives lowest.
            // Use MIN with sentinel: token = 0 means failure, else rank+1.
            int my_tok = ok ? (rank + 1) : (nranks + 1); // nranks+1 = "failed"
            int min_tok = 0;
            MPI_Allreduce(&my_tok, &min_tok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            if (min_tok <= nranks) // at least one rank succeeded
            {
                winning_rank = min_tok - 1;
                winning_pair = round * nranks + winning_rank;
                win_rho_ms = local_rho_ms;
                break;
            }
        }

        // ---- Broadcast winner's cbrt ----
        if (winning_rank < 0)
        {
            if (rank == 0)
                fprintf(stderr, "[bench %d] All %d rounds exhausted\n",
                        bench, K64_ROUNDS);
            mpz_clear(local_cbrt);
            continue;
        }

        unsigned char *rbuf = (unsigned char *)calloc(res_bytes, 1);
        if (rank == winning_rank)
        {
            size_t wr = 0;
            unsigned char *tmp =
                (unsigned char *)mpz_export(NULL, &wr, 1, 1, 0, 0, local_cbrt);
            if (tmp && wr && wr <= res_bytes)
            {
                memcpy(rbuf + res_bytes - wr, tmp, wr);
                free(tmp);
            }
        }
        MPI_Bcast(rbuf, (int)res_bytes, MPI_UNSIGNED_CHAR,
                  winning_rank, MPI_COMM_WORLD);
        mpz_import(result, res_bytes, 1, 1, 0, 0, rbuf);
        free(rbuf);
        mpz_clear(local_cbrt);

        // Broadcast timing / factor count from winner to rank 0
        MPI_Bcast(&win_rho_ms, 1, MPI_DOUBLE, winning_rank, MPI_COMM_WORLD);
        MPI_Bcast(&winning_factors, 1, MPI_INT, winning_rank, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_dt =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rank == 0)
        {
            accum_record(&rho_acc, win_rho_ms);
            accum_record(&total_acc, total_dt);

            if (bench == 0)
            {
                printf("  cold: rank %d won (pair %d), %d factors, "
                       "rho=%.2f ms, total=%.2f ms\n\n",
                       winning_rank, winning_pair,
                       winning_factors, win_rho_ms, total_dt);
                fflush(stdout);
            }
            if (bench > 0)
            {
                json_res.factor_times.push_back(win_rho_ms);
                json_res.recon_times.push_back(total_dt - win_rho_ms);
                json_res.total_times.push_back(total_dt);
                json_res.pollard_ms.push_back(win_rho_ms);
                json_res.blinding_attempts.push_back(winning_pair + 1);
                json_res.factors_large.push_back(winning_factors);
            }
        }
    }

    // ── Save cache ────────────────────────────────────────────────────────────
    if (dirty && rank == 0)
    {
        kf_cbrt_cache_save(&cache, cache_path);
        printf("[rank 0] cbrt cache saved: %d entries -> %s\n",
               cache.count, cache_path);
    }

    // ── Report ────────────────────────────────────────────────────────────────
    if (rank == 0)
    {
        int n_warm = bench_count > 1 ? bench_count - 1 : 0;
        if (n_warm > 0 && method == METHOD_RHO)
        {
            printf("\nResults over %d warm runs:\n", n_warm);
            print_timing_stat_moments("Pollard rho (winner):", rho_acc.sum,
                                      rho_acc.sumsq, n_warm, 30, true);
            print_timing_stat_moments("Total (rho+broadcast):", total_acc.sum,
                                      total_acc.sumsq, n_warm, 30, true);
        }
        mpz_t tmp;
        mpz_init(tmp);
        double powm_avg = run_powm_baseline(tmp, challenge, modulo,
                                            bench_count, 30);
        json_res.powm_baseline_ms = powm_avg;
        mpz_clear(tmp);
        if (n_warm > 0 && rho_acc.sum > 0)
        {
            double rho_avg = rho_acc.sum / n_warm;
            double total_avg = total_acc.sum / n_warm;
            printf("\n--- Speedup vs sequential powm ---\n");
            printf("  powm vs Pollard (winner alone): %.2fx\n",
                   powm_avg / rho_avg);
            printf("  powm vs total:                  %.2fx\n",
                   powm_avg / total_avg);
        }
        mpz_t ch_mod;
        mpz_init(ch_mod);
        mpz_mod(ch_mod, challenge, modulo);
        verify_cube_root(result, ch_mod, modulo);
        mpz_clear(ch_mod);
        fflush(stdout);
    }

    kf_cbrt_cache_clear(&cache);
    kf_blinding_table_free(pairs, total_pairs);
    mpz_clears(challenge, modulo, result, NULL);

    if (rank == 0 && json_file && json_file[0])
        write_benchmark_json(json_file, json_res);

    return 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
    {
        fprintf(stderr, "MPI_THREAD_FUNNELED not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char *challenge_file = "data/challenges/k64/challenge_0_64_7168.txt";
    const char *modulo_file = "data/challenges/k64/modulo_64_7168.txt";
    const char *json_file = nullptr;
    int bench_count = 1;
    run_method_t method = METHOD_RHO;
    bool challenge_set = false;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--bench") && i + 1 < argc)
            bench_count = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_file = argv[++i];
        else if (!strcmp(argv[i], "--method") && i + 1 < argc)
        {
            const char *m = argv[++i];
            if (!strcmp(m, "rho"))
                method = METHOD_RHO;
            else if (!strcmp(m, "powm"))
                method = METHOD_POWM;
            else
            {
                if (rank == 0)
                    fprintf(stderr, "Unknown method '%s'\n", m);
                MPI_Finalize();
                return 1;
            }
        }
        else if (argv[i][0] != '-')
        {
            if (!challenge_set)
            {
                challenge_file = argv[i];
                challenge_set = true;
            }
            else
                modulo_file = argv[i];
        }
        else
        {
            if (rank == 0)
                fprintf(stderr, "Unknown option '%s'\n", argv[i]);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0)
    {
        printf("=== k64_mpi ===\n");
        printf("Algorithm: smoothness-based with Pollard's rho (paper Alg. 2)\n");
        printf("Ranks: %d  Threads/rank: %d  "
               "Distinct r-values/round: %d  Max rounds: %d\n\n",
               nranks,
#ifdef _OPENMP
               omp_get_max_threads(),
#else
               1,
#endif
               nranks, K64_ROUNDS);
        fflush(stdout);
    }

    solve_one(rank, nranks, challenge_file, modulo_file,
              bench_count, method, json_file);

    MPI_Finalize();
    return 0;
}