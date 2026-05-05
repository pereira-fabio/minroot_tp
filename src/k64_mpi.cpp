// =============================================================================
// k64_mpi.cpp — MPI+OpenMP hybrid cube root solver for 64-bit prime factors
//
// Factoring strategy:
//   Phase A  Trial division   — each rank sieves its slice of [65537, 2^32]
//                               to find all small factors.  Identical to the
//                               k32_mpi sieve path.
//   Phase B  Factor gather    — MPI_Gather small factors to rank 0.
//   Phase C  Pollard's rho    — rank 0 runs kf_factor_recursive (OMP-parallel)
//                               on the cofactor = blinded / small_product.
//   Phase D  Factor broadcast — rank 0 serialises large factors; MPI_Bcast.
//   Phase E  cbrt partition   — factor[i] → rank (i % nranks); each rank
//                               computes its partial cbrt product.
//   Phase F  Gather + combine — same ksc_gather_and_combine_cbrt as k32.
//
// The cbrt cache is shared (data/cbrt/modulo_64_<bits>.cbrt, no rank suffix)
// because every rank may need any factor's cbrt for reconstruction.
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
#include <inttypes.h>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define BLIND_TABLE_SIZE 255  // precomputed blinding pairs
#define K64_MAX_FAC_RANK 128  // max distinct factors per rank from trial div
#define K64_MAX_FAC_LARGE 128 // max large factors from Pollard's rho

typedef enum
{
    METHOD_SIEVE,
    METHOD_POWM
} run_method_t;

// ---------------------------------------------------------------------------
// Small factor gather: MPI wire format
//
// Each rank sends up to K64_MAX_FAC_RANK factors from trial division.
// All factors from trial division are ≤ 2^32, so uint32_t is sufficient.
// Wire format per slot (fixed size, one per rank):
//   int32_t  count                             (number of valid entries)
//   uint32_t primes[K64_MAX_FAC_RANK]          (prime values)
//   int32_t  exponents[K64_MAX_FAC_RANK]       (corresponding exponents)
// ---------------------------------------------------------------------------

#define K64_SLOT_BYTES \
    (sizeof(int32_t) + K64_MAX_FAC_RANK * (sizeof(uint32_t) + sizeof(int32_t)))

static void pack_small_factors(const kfactor_list_t *f, unsigned char *slot)
{
    int32_t n = (int32_t)(f->count < K64_MAX_FAC_RANK ? f->count : K64_MAX_FAC_RANK);
    memcpy(slot, &n, sizeof(int32_t));
    slot += sizeof(int32_t);
    uint32_t *primes = (uint32_t *)slot;
    int32_t *exps = (int32_t *)(slot + K64_MAX_FAC_RANK * sizeof(uint32_t));
    for (int i = 0; i < n; i++)
    {
        primes[i] = (uint32_t)mpz_get_ui(f->items[i].prime);
        exps[i] = (int32_t)f->items[i].exponent;
    }
}

static void unpack_small_factors(const unsigned char *slot, kfactor_list_t *out)
{
    int32_t n;
    memcpy(&n, slot, sizeof(int32_t));
    slot += sizeof(int32_t);
    const uint32_t *primes = (const uint32_t *)slot;
    const int32_t *exps = (const int32_t *)(slot + K64_MAX_FAC_RANK * sizeof(uint32_t));
    for (int i = 0; i < n; i++)
        kfl_append_ul(out, (unsigned long)primes[i], (int)exps[i]);
}

// ---------------------------------------------------------------------------
// Compute partial cbrt product for factors assigned to this rank
//
// Factor at index i in the full list is assigned to rank (i % nranks).
// result = ∏ cbrt(p_i)^e_i  mod M   for all i where (i % nranks == rank)
// ---------------------------------------------------------------------------

static void compute_partial_cbrt_k64(
    mpz_t result,
    const kfactor_list_t *all_factors,
    const mpz_t modulo,
    cbrt_cache_t *cache,
    bool *cache_dirty,
    int rank, int nranks,
    double *hit_ms_out, double *miss_ms_out,
    int *hits_out, int *misses_out)
{
    mpz_set_ui(result, 1);
    double hit_ms = 0.0, miss_ms = 0.0;
    int hits = 0, misses = 0;

    for (int i = 0; i < all_factors->count; i++)
    {
        if (i % nranks != rank)
            continue; // not this rank's factor

        unsigned long p = mpz_get_ui(all_factors->items[i].prime);
        int exp = all_factors->items[i].exponent;

        auto t0 = std::chrono::high_resolution_clock::now();
        int idx = kf_cbrt_cache_find(cache, p);
        if (idx >= 0)
        {
            // Cache hit: look up precomputed cbrt
            mpz_t term;
            mpz_init(term);
            if (exp == 1)
                mpz_set(term, cache->items[idx].cbrt);
            else
                mpz_powm_ui(term, cache->items[idx].cbrt, (unsigned long)exp, modulo);
            mpz_mul(result, result, term);
            mpz_mod(result, result, modulo);
            mpz_clear(term);
            auto t1 = std::chrono::high_resolution_clock::now();
            hit_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            hits++;
        }
        else
        {
            // Cache miss: compute via mpz_powm (should be rare after precompute pass)
            mpz_t e_inv, m1, p_mpz, cbrt_p, term;
            mpz_inits(e_inv, m1, p_mpz, cbrt_p, term, NULL);
            mpz_sub_ui(m1, modulo, 1);
            mpz_set_ui(e_inv, 3);
            mpz_invert(e_inv, e_inv, m1);
            mpz_set_ui(p_mpz, p);
            mpz_powm(cbrt_p, p_mpz, e_inv, modulo);
            kf_cbrt_cache_append(cache, p, cbrt_p);
            *cache_dirty = true;
            if (exp == 1)
                mpz_set(term, cbrt_p);
            else
                mpz_powm_ui(term, cbrt_p, (unsigned long)exp, modulo);
            mpz_mul(result, result, term);
            mpz_mod(result, result, modulo);
            mpz_clears(e_inv, m1, p_mpz, cbrt_p, term, NULL);
            auto t1 = std::chrono::high_resolution_clock::now();
            miss_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            misses++;
        }
    }
    if (hit_ms_out)
        *hit_ms_out = hit_ms;
    if (miss_ms_out)
        *miss_ms_out = miss_ms;
    if (hits_out)
        *hits_out = hits;
    if (misses_out)
        *misses_out = misses;
}

// ---------------------------------------------------------------------------
// solve_one
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count, run_method_t method,
    const char *json_output_file)
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
        printf("Ranks: %d  Threads/rank: %d  Total workers: %d\n",
               nranks, n_threads, nranks * n_threads);
        fflush(stdout);
    }

    // ── Small prime sieve (for trial division up to 2^32) ────────────────────
    int sp_count = 0;
    unsigned *sp = kf_generate_small_primes(&sp_count);

    // ── Sieve range partition across MPI ranks ────────────────────────────────
    // Each rank covers [65537, 2^32] / nranks.  Identical to k32_mpi sieve.
    const uint64_t RANGE_LO = 65537ULL;
    const uint64_t RANGE_HI = (uint64_t)UINT32_MAX;
    const uint64_t RANGE_SZ = RANGE_HI - RANGE_LO + 1;
    uint64_t my_lo = RANGE_LO + (RANGE_SZ * (uint64_t)rank) / (uint64_t)nranks;
    uint64_t my_hi = RANGE_LO + (RANGE_SZ * (uint64_t)(rank + 1)) / (uint64_t)nranks - 1;
    if (rank == nranks - 1)
        my_hi = RANGE_HI;

    // ── cbrt cache (shared across all ranks) ─────────────────────────────────
    cbrt_cache_t cbrt_cache;
    kf_cbrt_cache_init(&cbrt_cache);
    bool cbrt_cache_dirty = false;

    ksc_ensure_k64_cbrt(&cbrt_cache, challenge, modulo, modulo_file, rank);

    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    // ── Blinding table ────────────────────────────────────────────────────────
    blinding_pair_t *blind_table = nullptr;
    if (rank == 0)
    {
        blind_table = kf_blinding_table_alloc(BLIND_TABLE_SIZE, modulo);
        if (!blind_table)
        {
            fprintf(stderr, "[rank 0] Failed to allocate blinding table\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // ── JSON results ──────────────────────────────────────────────────────────
    benchmark_results_t json_results;
    ksc_init_json_results(&json_results, 64, bench_count,
                          method == METHOD_POWM ? "powm" : "sieve",
                          nranks, n_threads, challenge, rank);

    bench_accum_t fac_acc = {0, 0}, loc_acc = {0, 0}, gat_acc = {0, 0},
                  com_acc = {0, 0}, recon_acc = {0, 0};

    // ── 2 warmup runs ─────────────────────────────────────────────────────────
    if (method != METHOD_POWM)
    {
        for (int wu = 0; wu < 2; wu++)
        {
            auto wt0 = std::chrono::high_resolution_clock::now();
            kfactor_list_t wf;
            kfl_init(&wf);
            kf_trial_divide_sieve(challenge, my_lo, my_hi, sp_count, sp, &wf);
            kfl_free(&wf);
            auto wt1 = std::chrono::high_resolution_clock::now();
            if (rank == 0)
                json_results.warmup_total_ms.push_back(
                    std::chrono::duration<double, std::milli>(wt1 - wt0).count());
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if (rank == 0)
            printf("Warmup complete (2 runs)\n\n");
    }

    // ── Benchmark loop ────────────────────────────────────────────────────────
    for (int bench = 0; bench < bench_count && method != METHOD_POWM; bench++)
    {

        // ---- BLINDING + FACTORING ----
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        int attempt = 0;
        const int MAX_ATTEMPTS = 256;
        size_t blinded_nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
        unsigned char *blinded_buf = (unsigned char *)malloc(blinded_nbytes);

        // Complete factor list (small + large), built fresh each iteration
        kfactor_list_t all_factors;
        kfl_init(&all_factors);

        double pollard_elapsed = 0.0;
        int n_small = 0, n_large = 0;
        int local_small_count = 0; // updated each successful attempt

        mpz_t blinded_challenge;
        mpz_init(blinded_challenge);

        for (attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
        {
            // -- Rank 0 applies blinding and broadcasts --
            if (rank == 0)
            {
                if (attempt == 0)
                    mpz_set(blinded_challenge, challenge);
                else
                {
                    int idx = attempt - 1;
                    if (idx < BLIND_TABLE_SIZE)
                    {
                        mpz_mul(blinded_challenge, challenge, blind_table[idx].r3);
                        mpz_mod(blinded_challenge, blinded_challenge, modulo);
                    }
                    else
                        mpz_set(blinded_challenge, challenge);
                }
                memset(blinded_buf, 0, blinded_nbytes);
                size_t wr = 0;
                unsigned char *tmp = (unsigned char *)mpz_export(
                    NULL, &wr, 1, 1, 0, 0, blinded_challenge);
                if (tmp && wr > 0 && wr <= blinded_nbytes)
                {
                    memcpy(blinded_buf + blinded_nbytes - wr, tmp, wr);
                    free(tmp);
                }
            }
            MPI_Bcast(blinded_buf, (int)blinded_nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            if (rank != 0)
                mpz_import(blinded_challenge, blinded_nbytes, 1, 1, 0, 0, blinded_buf);

            // -- Phase A: Trial division — each rank sieves its slice --
            kfactor_list_t local_small;
            kfl_init(&local_small);
            kf_trial_divide_sieve(blinded_challenge, my_lo, my_hi,
                                  sp_count, sp, &local_small);
            int local_small_count = local_small.count; // save before free

            // -- Phase B: Gather small factors to rank 0 --
            // Pack this rank's factors into a fixed-size slot
            unsigned char *my_slot = (unsigned char *)calloc(K64_SLOT_BYTES, 1);
            pack_small_factors(&local_small, my_slot);
            kfl_free(&local_small);

            unsigned char *all_slots = nullptr;
            if (rank == 0)
                all_slots = (unsigned char *)malloc((size_t)nranks * K64_SLOT_BYTES);
            MPI_Gather(my_slot, (int)K64_SLOT_BYTES, MPI_UNSIGNED_CHAR,
                       all_slots, (int)K64_SLOT_BYTES, MPI_UNSIGNED_CHAR,
                       0, MPI_COMM_WORLD);
            free(my_slot);

            // -- Phase C: Rank 0 computes cofactor and runs Pollard's rho --
            // Wire format for large factors broadcast:
            //   [int32_t count]  [uint64_t primes[count]]  [int32_t exps[count]]
            static uint64_t large_primes[K64_MAX_FAC_LARGE];
            static int32_t large_exps[K64_MAX_FAC_LARGE];
            int32_t large_count = 0;
            int complete = 0;

            if (rank == 0)
            {
                // Aggregate all small factors
                kfactor_list_t all_small_gathered;
                kfl_init(&all_small_gathered);
                for (int rk = 0; rk < nranks; rk++)
                    unpack_small_factors(all_slots + (size_t)rk * K64_SLOT_BYTES,
                                         &all_small_gathered);
                free(all_slots);

                // Compute small product = ∏ p_i^e_i
                mpz_t small_prod;
                mpz_init_set_ui(small_prod, 1);
                for (int i = 0; i < all_small_gathered.count; i++)
                {
                    unsigned long p = mpz_get_ui(all_small_gathered.items[i].prime);
                    for (int e = 0; e < all_small_gathered.items[i].exponent; e++)
                        mpz_mul_ui(small_prod, small_prod, p);
                }

                // cofactor = blinded_challenge / small_product
                mpz_t cofactor;
                mpz_init(cofactor);
                if (mpz_divisible_p(blinded_challenge, small_prod))
                    mpz_divexact(cofactor, blinded_challenge, small_prod);
                else
                    mpz_set(cofactor, blinded_challenge); // shouldn't happen

                // -- Phase C: Pollard's rho on cofactor (rank 0, OMP threads) --
                // Cofactor has already had all 32-bit factors removed, so its
                // remaining factors are in [2^32, 2^64] — Pollard's rho is fast.
                // DO NOT call kf_factor_challenge here (it only trial-divides
                // up to 1000 — use kf_factor_recursive which uses OMP directly).
                double t_poll0 = MPI_Wtime();
                kfactor_list_t large_factors;
                kfl_init(&large_factors);
                if (mpz_cmp_ui(cofactor, 1) > 0)
                {
                    if (bench == 0)
                    {
                        printf("  Cofactor: %zu bits — Pollard's rho (OMP)\n",
                               mpz_sizeinbase(cofactor, 2));
                        fflush(stdout);
                    }
                    kf_factor_recursive(cofactor, &large_factors, 1);
                }
                double t_poll1 = MPI_Wtime();
                pollard_elapsed = (t_poll1 - t_poll0) * 1000.0; // ms

                mpz_clear(cofactor);
                mpz_clear(small_prod);

                // Check completeness: small_prod * large_prod == blinded_challenge?
                mpz_t check;
                mpz_init_set_ui(check, 1);
                for (int i = 0; i < all_small_gathered.count; i++)
                {
                    unsigned long p = mpz_get_ui(all_small_gathered.items[i].prime);
                    for (int e = 0; e < all_small_gathered.items[i].exponent; e++)
                        mpz_mul_ui(check, check, p);
                }
                for (int i = 0; i < large_factors.count; i++)
                {
                    for (int e = 0; e < large_factors.items[i].exponent; e++)
                        mpz_mul(check, check, large_factors.items[i].prime);
                }
                complete = (mpz_cmp(check, blinded_challenge) == 0) ? 1 : 0;
                mpz_clear(check);

                if (bench == 0)
                {
                    printf("Attempt %d: small=%d  large=%d  complete=%s\n",
                           attempt + 1, all_small_gathered.count,
                           large_factors.count, complete ? "yes" : "no");
                    fflush(stdout);
                }

                // Pack large factors for broadcast
                large_count = (int32_t)(large_factors.count < K64_MAX_FAC_LARGE
                                            ? large_factors.count
                                            : K64_MAX_FAC_LARGE);
                for (int i = 0; i < large_count; i++)
                {
                    large_primes[i] = (uint64_t)mpz_get_ui(large_factors.items[i].prime);
                    large_exps[i] = (int32_t)large_factors.items[i].exponent;
                }

                // Build all_factors on rank 0: small + large
                kfl_free(&all_factors);
                kfl_init(&all_factors);
                for (int i = 0; i < all_small_gathered.count; i++)
                    kfl_append_ul(&all_factors,
                                  mpz_get_ui(all_small_gathered.items[i].prime),
                                  all_small_gathered.items[i].exponent);
                for (int i = 0; i < large_count; i++)
                    kfl_append_ul(&all_factors, (unsigned long)large_primes[i],
                                  (int)large_exps[i]);

                n_small = all_small_gathered.count;
                n_large = large_count;
                kfl_free(&all_small_gathered);
                kfl_free(&large_factors);
            } // end rank 0

            // -- Phase D: Broadcast completeness flag + large factors --
            MPI_Bcast(&complete, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&large_count, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
            if (large_count > 0)
            {
                MPI_Bcast(large_primes, large_count, MPI_UINT64_T, 0, MPI_COMM_WORLD);
                MPI_Bcast(large_exps, large_count, MPI_INT32_T, 0, MPI_COMM_WORLD);
            }
            MPI_Bcast(&pollard_elapsed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (complete)
            {
                // Update local_small_count to reflect last successful attempt
                local_small_count = local_small_count; // already set above
                break;
            }
        } // end retry loop

        // Rank 0 already has all_factors built. Non-root ranks need it too.
        // Broadcast: small count, primes[], exps[], then large factors already sent.
        // Simplest correct approach: broadcast the full merged factor list.
        int32_t total_count = (rank == 0) ? (int32_t)all_factors.count : 0;
        MPI_Bcast(&total_count, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

        if (rank != 0)
        {
            kfl_free(&all_factors);
            kfl_init(&all_factors);
        }

        // Broadcast primes as uint64_t and exponents as int32_t
        std::vector<uint64_t> all_prime_buf(total_count);
        std::vector<int32_t> all_exp_buf(total_count);
        if (rank == 0)
            for (int i = 0; i < total_count; i++)
            {
                all_prime_buf[i] = (uint64_t)mpz_get_ui(all_factors.items[i].prime);
                all_exp_buf[i] = (int32_t)all_factors.items[i].exponent;
            }
        if (total_count > 0)
        {
            MPI_Bcast(all_prime_buf.data(), total_count, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            MPI_Bcast(all_exp_buf.data(), total_count, MPI_INT32_T, 0, MPI_COMM_WORLD);
        }
        if (rank != 0)
            for (int i = 0; i < total_count; i++)
                kfl_append_ul(&all_factors,
                              (unsigned long)all_prime_buf[i],
                              (int)all_exp_buf[i]);

        free(blinded_buf);
        mpz_clear(blinded_challenge);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();
        double fac_dt = std::chrono::duration<double, std::milli>(t_fac1 - t_fac0).count();

        if (rank == 0)
        {
            accum_record(&fac_acc, fac_dt);
            if (bench > 0)
            {
                json_results.factor_times.push_back(fac_dt);
                json_results.blinding_attempts.push_back(attempt + 1);
                json_results.pollard_ms.push_back(pollard_elapsed);
                json_results.factors_small.push_back(n_small);
                json_results.factors_large.push_back(n_large);
            }
            if (bench == 0)
            {
                printf("Total factors: %d (%d small ≤ 2^32, %d large > 2^32)\n",
                       all_factors.count, n_small, n_large);
                printf("Pollard's rho time: %.4f ms\n\n", pollard_elapsed);
                fflush(stdout);
            }
        }
        // Record per-rank prime stats (first warm run only)
        if (bench == 1)
            ksc_gather_prime_stats(&json_results,
                                   local_small_count,
                                   (int64_t)(my_hi - my_lo + 1),
                                   rank, nranks);

        // -- Phase F: cbrt partition — factor[i] → rank (i % nranks) --
        auto t_loc0 = std::chrono::high_resolution_clock::now();
        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        double hit_ms = 0.0, miss_ms = 0.0;
        int hits = 0, misses = 0;
        compute_partial_cbrt_k64(local_cbrt, &all_factors, modulo,
                                 &cbrt_cache, &cbrt_cache_dirty,
                                 rank, nranks,
                                 &hit_ms, &miss_ms, &hits, &misses);
        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(t_loc1 - t_loc0).count();

        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            accum_record(&loc_acc, max_local_dt);
            if (bench > 0)
            {
                json_results.local_cbrt_times.push_back(max_local_dt);
                json_results.cbrt_hit_ms.push_back(hit_ms);
                json_results.cbrt_miss_ms.push_back(miss_ms);
                json_results.cbrt_hits.push_back(hits);
                json_results.cbrt_misses.push_back(misses);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // -- Phase G: gather + combine --
        double t_gat = 0.0, t_com = 0.0;
        ksc_gather_and_combine_cbrt(result, local_cbrt, modulo, cbrt_bytes,
                                    rank, nranks, attempt,
                                    blind_table, BLIND_TABLE_SIZE,
                                    &t_gat, &t_com);
        mpz_clear(local_cbrt);
        kfl_free(&all_factors);

        if (rank == 0)
        {
            accum_record(&gat_acc, t_gat);
            accum_record(&com_acc, t_com);
            double recon_dt = max_local_dt + t_gat + t_com;
            accum_record(&recon_acc, recon_dt);
            if (bench > 0)
            {
                json_results.gather_times.push_back(t_gat);
                json_results.combine_times.push_back(t_com);
                json_results.recon_times.push_back(recon_dt);
                if (!json_results.factor_times.empty())
                    json_results.total_times.push_back(
                        json_results.factor_times.back() + recon_dt);
            }
        }
    } // end bench loop

    // ── Save updated cbrt cache ───────────────────────────────────────────────
    if (cbrt_cache_dirty && rank == 0)
    {
        char stem[256], cache_path[280];
        const char *bname = modulo_file ? modulo_file : "modulo";
        const char *sl = strrchr(bname, '/');
        const char *fn = sl ? sl + 1 : bname;
        strncpy(stem, fn, sizeof(stem) - 1);
        stem[sizeof(stem) - 1] = '\0';
        size_t bl = strlen(stem);
        if (bl >= 4 && strcmp(stem + bl - 4, ".txt") == 0)
            stem[bl - 4] = '\0';
        snprintf(cache_path, sizeof(cache_path), "%s/%s.cbrt", KSC_CBRT_DIR, stem);
        kf_cbrt_cache_save(&cbrt_cache, cache_path);
    }

    // ── Report + write JSON ───────────────────────────────────────────────────
    if (rank == 0)
    {
        ksc_print_timing_report(bench_count, &fac_acc, &loc_acc,
                                &gat_acc, &com_acc, &recon_acc);
        ksc_run_and_report_powm_baseline(&json_results, challenge, modulo,
                                         bench_count, &recon_acc, &fac_acc, rank);
        mpz_t ch_mod;
        mpz_init(ch_mod);
        mpz_mod(ch_mod, challenge, modulo);
        verify_cube_root(result, ch_mod, modulo);
        mpz_clear(ch_mod);
        fflush(stdout);
    }

    // Cleanup
    kf_cbrt_cache_clear(&cbrt_cache);
    if (rank == 0 && blind_table)
        kf_blinding_table_free(blind_table, BLIND_TABLE_SIZE);
    free(sp);
    mpz_clears(challenge, modulo, result, NULL);

    if (rank == 0 && json_output_file && json_output_file[0])
        write_benchmark_json(json_output_file, json_results);

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

    const char *challenge_file = "data/challenges/k64/challenge_0_64_7000.txt";
    const char *modulo_file = "data/challenges/k64/modulo_64_7000.txt";
    const char *json_file = nullptr;
    int bench_count = 1;
    run_method_t method = METHOD_SIEVE;
    bool challenge_set = false; // tracks whether first positional arg was consumed

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--bench") && i + 1 < argc)
            bench_count = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            json_file = argv[++i];
        else if (!strcmp(argv[i], "--method") && i + 1 < argc)
        {
            const char *m = argv[++i];
            if (!strcmp(m, "sieve"))
                method = METHOD_SIEVE;
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
        printf("Ranks: %d  Threads/rank: %d  Method: %s  Bench: %d\n\n",
               nranks,
#ifdef _OPENMP
               omp_get_max_threads(),
#else
               1,
#endif
               method == METHOD_POWM ? "powm" : "sieve",
               bench_count);
        fflush(stdout);
    }

    solve_one(rank, nranks, challenge_file, modulo_file,
              bench_count, method, json_file);

    MPI_Finalize();
    return 0;
}