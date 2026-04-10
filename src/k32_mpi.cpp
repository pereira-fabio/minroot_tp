// =============================================================================
// k32_mpi.cpp — MPI+OpenMP hybrid cube root solver for 32-bit prime factors
//
// Architecture:
//   MPI layer:    splits prime range [65537, 2^32] across nodes
//   OpenMP layer: parallelises sieve / trial division within each node
//
// Key improvement over previous version — distributed reconstruction:
//   OLD: rank 0 gathers all (prime, exponent) pairs and does every powm itself.
//        For 7000-bit challenges this cost ~650 ms regardless of node count.
//   NEW: every rank computes cbrt for its own factors locally (parallel powm
//        across all ranks), then each rank sends only ONE fixed-size partial
//        product (~875 bytes) to rank 0. Rank 0 multiplies nranks numbers.
//        Reconstruction now scales with the rest of the computation.
//
// Communication pattern:
//   MPI_Bcast:   challenge + modulo bytes → all ranks (once per challenge)
//   MPI_Barrier: sync for accurate timing
//   MPI_Reduce:  max local-cbrt time across ranks → rank 0 (timing only)
//   MPI_Barrier: ensure all ranks reached gather before measuring comm time
//   MPI_Gather:  partial cbrt bytes from each rank → rank 0
//                payload: nranks × ceil(modulo_bits/8)+1 ≈ nranks × 875 bytes
//
// Methods:
//   sieve  — segmented sieve, primes generated on-the-fly   [default]
//   table  — precomputed prime file; each rank loads its slice via binary
//             search so only ~range/nranks × 4 bytes are read per rank
//
// Usage:
//   k32_mpi [challenge] [modulo] [--method sieve|table] [--primes FILE]
//           [--precompute] [--bench N]
//   k32_mpi --all [--data DIR] [--method sieve|table] [--primes FILE]
//           [--bench N]
//
// Prime table generation (rank 0 only, one-time, ~10 s):
//   srun ... k32_mpi --precompute [--primes k32_primes.bin]
// =============================================================================

#include "k_common.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define SEG_SIZE (1 << 19)                      // 512 KiB sieve segment
#define PRIME_TABLE_MAGIC 0x4B33325052494D53ULL // "K32PRIMS"
#define WRITE_BATCH 65536

typedef enum
{
    METHOD_SIEVE,
    METHOD_TABLE
} run_method_t;

// ---------------------------------------------------------------------------
// Factor list — (prime, exponent) packed as uint64_t pairs for easy MPI use
// (kept for internal bookkeeping; no longer gathered to rank 0 directly)
// ---------------------------------------------------------------------------

typedef struct
{
    uint64_t prime;
    uint64_t exponent;
} factor_msg_t;
typedef struct
{
    factor_msg_t *items;
    int count;
    int capacity;
} factor_list_t;

static void fl_init(factor_list_t *f)
{
    f->items = NULL;
    f->count = 0;
    f->capacity = 0;
}

static void fl_free(factor_list_t *f)
{
    free(f->items);
    f->items = NULL;
    f->count = 0;
    f->capacity = 0;
}

static void fl_append(factor_list_t *f, uint64_t prime, uint64_t exp)
{
    if (f->count == f->capacity)
    {
        int nc = (f->capacity == 0) ? 64 : f->capacity * 2;
        f->items = (factor_msg_t *)realloc(f->items,
                                           (size_t)nc * sizeof(factor_msg_t));
        f->capacity = nc;
    }
    f->items[f->count].prime = prime;
    f->items[f->count].exponent = exp;
    f->count++;
}

// ---------------------------------------------------------------------------
// Small prime generator (same as k32.cpp)
// ---------------------------------------------------------------------------

static unsigned *generate_small_primes(int *count)
{
    const int limit = 65536;
    char *sieve = (char *)calloc((size_t)(limit + 1), 1);
    memset(sieve, 1, (size_t)(limit + 1));
    sieve[0] = sieve[1] = 0;
    for (int i = 2; (long long)i * i <= limit; i++)
        if (sieve[i])
            for (int j = i * i; j <= limit; j += i)
                sieve[j] = 0;
    *count = 0;
    for (int i = 2; i <= limit; i++)
        if (sieve[i])
            (*count)++;
    unsigned *primes = (unsigned *)malloc((size_t)(*count) * sizeof(unsigned));
    int idx = 0;
    for (int i = 2; i <= limit; i++)
        if (sieve[i])
            primes[idx++] = (unsigned)i;
    free(sieve);
    return primes;
}

// ---------------------------------------------------------------------------
// Prime table I/O
//
// File format: [uint64_t magic][uint64_t count][uint32_t prime[count]]
// Sorted ascending, range [65537, 2^32], ~813 MB total.
//
// generate_and_save_prime_table: called by rank 0 only.
// load_prime_slice: called by every rank; uses binary search to seek
//   directly to [range_lo, range_hi] — only that slice is read into RAM.
// ---------------------------------------------------------------------------

static bool generate_and_save_prime_table(const char *path,
                                          int sp_count, const unsigned *sp)
{
    printf("[rank 0] Generating prime table [65537, 2^32] → %s ...\n", path);
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "Cannot create %s\n", path);
        return false;
    }

    uint64_t magic = PRIME_TABLE_MAGIC, count = 0;
    fwrite(&magic, 8, 1, f);
    long count_pos = ftell(f);
    fwrite(&count, 8, 1, f); // placeholder

    char *seg = (char *)malloc(SEG_SIZE);
    uint32_t *batch = (uint32_t *)malloc(WRITE_BATCH * sizeof(uint32_t));
    int bi = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t lo = 65537ULL; lo <= (uint64_t)UINT32_MAX; lo += SEG_SIZE)
    {
        uint64_t hi = lo + SEG_SIZE - 1;
        if (hi > (uint64_t)UINT32_MAX)
            hi = (uint64_t)UINT32_MAX;
        size_t seg_len = (size_t)(hi - lo + 1);

        memset(seg, 1, seg_len);
        for (int i = 0; i < sp_count; i++)
        {
            uint64_t p = sp[i];
            uint64_t start = ((lo + p - 1) / p) * p;
            if (start == p)
                start += p;
            for (uint64_t j = start; j <= hi; j += p)
                seg[j - lo] = 0;
        }
        for (uint64_t c = lo; c <= hi; c++)
        {
            if (!seg[c - lo])
                continue;
            batch[bi++] = (uint32_t)c;
            count++;
            if (bi == WRITE_BATCH)
            {
                fwrite(batch, 4, (size_t)WRITE_BATCH, f);
                bi = 0;
            }
        }
    }
    if (bi > 0)
        fwrite(batch, 4, (size_t)bi, f);
    free(seg);
    free(batch);

    fseek(f, count_pos, SEEK_SET);
    fwrite(&count, 8, 1, f);
    fclose(f);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[rank 0] Generated %" PRIu64 " primes in %.1f ms  (~%.0f MB)\n",
           count, ms, (double)(16 + count * 4) / 1e6);
    return true;
}

// Binary search: index of first prime >= target in the sorted uint32_t file.
// Primes start at byte offset 16 (after magic + count header).
static uint64_t prime_file_lower_bound(FILE *f, uint64_t total_count,
                                       uint32_t target)
{
    uint64_t lo = 0, hi = total_count;
    while (lo < hi)
    {
        uint64_t mid = lo + (hi - lo) / 2;
        uint32_t p;
        fseek(f, (long)(16LL + (long long)mid * 4LL), SEEK_SET);
        if (fread(&p, 4, 1, f) != 1)
            break;
        if (p < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Load only the slice [range_lo, range_hi] of the prime table.
// Returns a malloc'd uint32_t array; sets *slice_count.
// Returns NULL on failure (file not found or I/O error).
static uint32_t *load_prime_slice(uint64_t *slice_count, const char *path,
                                  uint32_t range_lo, uint32_t range_hi)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    uint64_t magic, total_count;
    if (fread(&magic, 8, 1, f) != 1 || magic != PRIME_TABLE_MAGIC ||
        fread(&total_count, 8, 1, f) != 1)
    {
        fclose(f);
        return NULL;
    }

    uint64_t start_idx = prime_file_lower_bound(f, total_count, range_lo);
    uint64_t end_idx;
    if (range_hi >= (uint32_t)UINT32_MAX)
        end_idx = total_count;
    else
        end_idx = prime_file_lower_bound(f, total_count, range_hi + 1U);

    *slice_count = end_idx - start_idx;
    if (*slice_count == 0)
    {
        fclose(f);
        return (uint32_t *)malloc(4);
    } // empty, not NULL

    uint32_t *slice = (uint32_t *)malloc(*slice_count * 4);
    if (!slice)
    {
        fclose(f);
        return NULL;
    }

    fseek(f, (long)(16LL + (long long)start_idx * 4LL), SEEK_SET);
    if (fread(slice, 4, *slice_count, f) != *slice_count)
    {
        free(slice);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return slice;
}

// ---------------------------------------------------------------------------
// Trial division — SIEVE method (segmented sieve + OpenMP, unchanged)
// ---------------------------------------------------------------------------

static void trial_divide_sieve(const mpz_t challenge,
                               uint64_t range_lo, uint64_t range_hi,
                               int sp_count, const unsigned *sp,
                               factor_list_t *found)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    factor_list_t *tf = (factor_list_t *)malloc(
        (size_t)nthreads * sizeof(factor_list_t));
    for (int t = 0; t < nthreads; t++)
        fl_init(&tf[t]);

    uint64_t range_size = range_hi - range_lo + 1;

#ifdef _OPENMP
#pragma omp parallel
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int t = 0, nt = 1;
#endif
        uint64_t t_lo = range_lo + (range_size * (uint64_t)t) / (uint64_t)nt;
        uint64_t t_hi = range_lo + (range_size * (uint64_t)(t + 1)) / (uint64_t)nt - 1;
        if (t == nt - 1)
            t_hi = range_hi;

        char *seg = (char *)malloc(SEG_SIZE);

        for (uint64_t lo = t_lo; lo <= t_hi; lo += SEG_SIZE)
        {
            uint64_t hi = lo + SEG_SIZE - 1;
            if (hi > t_hi)
                hi = t_hi;
            size_t seg_len = (size_t)(hi - lo + 1);

            memset(seg, 1, seg_len);
            for (int i = 0; i < sp_count; i++)
            {
                uint64_t p = sp[i];
                uint64_t start = ((lo + p - 1) / p) * p;
                if (start == p)
                    start += p;
                for (uint64_t j = start; j <= hi; j += p)
                    seg[j - lo] = 0;
            }
            for (uint64_t c = lo; c <= hi; c++)
            {
                if (!seg[c - lo])
                    continue;
                if (mpz_tdiv_ui(challenge, (unsigned long)c) != 0)
                    continue;
                int e = count_exponent(challenge, (unsigned long)c);
                if (e > 0)
                    fl_append(&tf[t], c, (uint64_t)e);
            }
        }
        free(seg);
    }

    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            fl_append(found, tf[t].items[i].prime, tf[t].items[i].exponent);
        fl_free(&tf[t]);
    }
    free(tf);
}

// ---------------------------------------------------------------------------
// Trial division — TABLE method
// Each rank iterates its own pre-loaded prime slice (loaded before bench loop).
// ---------------------------------------------------------------------------

static void trial_divide_table(const mpz_t challenge,
                               const uint32_t *primes, int64_t prime_count,
                               factor_list_t *found)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    factor_list_t *tf = (factor_list_t *)malloc(
        (size_t)nthreads * sizeof(factor_list_t));
    for (int t = 0; t < nthreads; t++)
        fl_init(&tf[t]);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < prime_count; i++)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        unsigned long p = (unsigned long)primes[i];
        if (mpz_tdiv_ui(challenge, p) != 0)
            continue;
        int e = count_exponent(challenge, p);
        if (e > 0)
            fl_append(&tf[t], (uint64_t)p, (uint64_t)e);
    }

    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            fl_append(found, tf[t].items[i].prime, tf[t].items[i].exponent);
        fl_free(&tf[t]);
    }
    free(tf);
}

// ---------------------------------------------------------------------------
// Local cube root reconstruction — called by EVERY rank
//
// Computes  partial = ∏ cbrt(p_i)^e_i  mod M  for this rank's local factors.
// e = 3^-1 mod (M-1) is precomputed once on each rank (same M, same e).
//
// For a 1024-bit challenge with 8 ranks each rank has ~4 factors on average,
// so no OpenMP is needed here — the parallelism comes from all ranks running
// simultaneously.
//
// The results from all ranks are gathered and multiplied at rank 0, giving
//   cbrt(challenge) = ∏_r partial_r  mod M
// by the cbrt homomorphism.
// ---------------------------------------------------------------------------

static void compute_local_cbrt(mpz_t result,
                               const factor_list_t *factors,
                               const mpz_t e, // 3^-1 mod (M-1)
                               const mpz_t modulo)
{
    mpz_set_ui(result, 1);
    for (int i = 0; i < factors->count; i++)
    {
        mpz_t p, cbrt_p, powered;
        mpz_inits(p, cbrt_p, powered, NULL);

        mpz_set_ui(p, (unsigned long)factors->items[i].prime);
        mpz_powm(cbrt_p, p, e, modulo); // cbrt(p) = p^e mod M

        if (factors->items[i].exponent == 1)
        {
            mpz_mul(result, result, cbrt_p);
        }
        else
        {
            // cbrt(p^k) = cbrt(p)^k
            mpz_powm_ui(powered, cbrt_p,
                        (unsigned long)factors->items[i].exponent, modulo);
            mpz_mul(result, result, powered);
        }
        mpz_mod(result, result, modulo);
        mpz_clears(p, cbrt_p, powered, NULL);
    }
}

// ---------------------------------------------------------------------------
// solve_one — solve a single (challenge, modulo) pair across all MPI ranks
//
// Reconstruction flow (NEW vs OLD):
//   OLD: gather all (prime,exp) pairs to rank 0 → rank 0 does all powm
//   NEW: every rank computes its partial cbrt locally (powm parallelised)
//        → MPI_Gather of nranks × ~875 bytes → rank 0 multiplies nranks terms
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count, int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi,
    run_method_t method, const char *primes_file)
{
    if (rank == 0)
    {
        printf("\n=== %s ===\n", challenge_file);
        fflush(stdout);
    }

    // ---- Rank 0 reads files; all ranks get the values via Bcast ----
    // If rank 0 cannot open the files it broadcasts read_ok=0 and all ranks
    // call MPI_Abort together — avoiding the hang caused by exit(1) on rank 0
    // while other ranks block on MPI_Bcast.

    int read_ok = 1;
    size_t challenge_nbytes = 0, modulo_nbytes = 0;
    unsigned char *challenge_bytes = NULL, *modulo_bytes = NULL;

    mpz_t challenge, modulo, result;
    mpz_inits(challenge, modulo, result, NULL);

    if (rank == 0)
    {
        FILE *fc = fopen(challenge_file, "rb");
        FILE *fm = fopen(modulo_file, "rb");
        if (!fc || !fm)
        {
            fprintf(stderr, "Error: cannot open '%s' or '%s'\n",
                    challenge_file, modulo_file);
            if (fc)
                fclose(fc);
            if (fm)
                fclose(fm);
            read_ok = 0;
        }
        else
        {
            fclose(fc);
            fclose(fm);
            read_mpz_from_file(challenge, challenge_file);
            read_mpz_from_file(modulo, modulo_file);
            challenge_bytes = (unsigned char *)mpz_export(
                NULL, &challenge_nbytes, 1, 1, 0, 0, challenge);
            modulo_bytes = (unsigned char *)mpz_export(
                NULL, &modulo_nbytes, 1, 1, 0, 0, modulo);
        }
    }

    // Broadcast success flag before any data — all ranks abort cleanly on failure
    MPI_Bcast(&read_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!read_ok)
    {
        mpz_clears(challenge, modulo, result, NULL);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 0;
    }

    // Broadcast sizes, allocate on non-root, then broadcast data
    uint64_t cn64 = (uint64_t)challenge_nbytes;
    uint64_t mn64 = (uint64_t)modulo_nbytes;
    MPI_Bcast(&cn64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mn64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    challenge_nbytes = (size_t)cn64;
    modulo_nbytes = (size_t)mn64;

    if (rank != 0)
    {
        challenge_bytes = (unsigned char *)malloc(challenge_nbytes);
        modulo_bytes = (unsigned char *)malloc(modulo_nbytes);
    }
    MPI_Bcast(challenge_bytes, (int)challenge_nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(modulo_bytes, (int)modulo_nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        mpz_import(challenge, challenge_nbytes, 1, 1, 0, 0, challenge_bytes);
        mpz_import(modulo, modulo_nbytes, 1, 1, 0, 0, modulo_bytes);
    }
    free(challenge_bytes);
    free(modulo_bytes);

    if (rank == 0)
    {
        printf("Challenge: %zu bits\n", mpz_sizeinbase(challenge, 2));
        printf("Modulo:    %zu bits\n", mpz_sizeinbase(modulo, 2));
        printf("Ranks: %d  Threads/rank: %d  Total workers: %d\n",
               nranks,
#ifdef _OPENMP
               omp_get_max_threads(), nranks * omp_get_max_threads());
#else
               1, nranks);
#endif
        fflush(stdout);
    }

    // ---- Load prime slice for table method (once, outside bench loop) ----
    // Each rank loads only its [my_lo, my_hi] slice via binary search on the
    // sorted prime file — no full 800+ MB read per rank.

    uint32_t *prime_slice = NULL;
    int64_t slice_count = 0;

    if (method == METHOD_TABLE)
    {
        uint64_t sc = 0;
        prime_slice = load_prime_slice(&sc, primes_file,
                                       (uint32_t)my_lo, (uint32_t)my_hi);
        slice_count = (int64_t)sc;
        if (!prime_slice)
        {
            fprintf(stderr, "[rank %d] Failed to load prime slice from %s\n",
                    rank, primes_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (rank == 0)
            printf("Rank 0 slice: %" PRId64 " primes\n", slice_count);
        fflush(stdout);
    }

    // Fixed-size byte buffer for partial cbrt export (big-endian, zero-padded).
    // Size is the same on all ranks since modulo is broadcast.
    size_t cbrt_bytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1; // +1 safety

    // Precompute e = 3^-1 mod (M-1) once — same value on every rank.
    mpz_t phi, e_inv, three_mpz;
    mpz_inits(phi, e_inv, three_mpz, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three_mpz, 3);
    mpz_invert(e_inv, three_mpz, phi);
    mpz_clears(phi, three_mpz, NULL);

    // Timing accumulators (rank 0 records all; max_local via MPI_Reduce)
    double fac_sum = 0, fac_sq = 0;
    double loc_sum = 0, loc_sq = 0; // max local cbrt across ranks
    double gat_sum = 0, gat_sq = 0;
    double com_sum = 0, com_sq = 0;

    // ---- Benchmark loop ----

    for (int bench = 0; bench < bench_count; bench++)
    {

        factor_list_t local_factors;
        fl_init(&local_factors);

        // ---- FACTORING ----
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac0 = std::chrono::high_resolution_clock::now();

        if (method == METHOD_SIEVE)
            trial_divide_sieve(challenge, my_lo, my_hi,
                               sp_count, sp, &local_factors);
        else
            trial_divide_table(challenge, prime_slice, slice_count,
                               &local_factors);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_fac1 = std::chrono::high_resolution_clock::now();

        {
            double dt = std::chrono::duration<double, std::milli>(t_fac1 - t_fac0).count();
            if (rank == 0)
            {
                fac_sum += dt;
                fac_sq += dt * dt;
            }
        }

        if (bench == 0 && rank == 0)
        {
            printf("Factors found by rank 0: %d distinct primes\n",
                   local_factors.count);
            fflush(stdout);
        }

        // ---- DISTRIBUTED RECONSTRUCTION ----
        //
        // Step 1 — every rank computes partial cbrt for its own factors.
        //   Runs in parallel across all ranks (like the factoring phase).
        //   Each rank does ~(total_factors / nranks) powm calls.
        auto t_loc0 = std::chrono::high_resolution_clock::now();

        mpz_t local_cbrt;
        mpz_init(local_cbrt);
        compute_local_cbrt(local_cbrt, &local_factors, e_inv, modulo);

        auto t_loc1 = std::chrono::high_resolution_clock::now();
        double local_dt = std::chrono::duration<double, std::milli>(t_loc1 - t_loc0).count();

        // Reduce to get max local cbrt time (the bottleneck rank) at rank 0
        double max_local_dt = 0.0;
        MPI_Reduce(&local_dt, &max_local_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            loc_sum += max_local_dt;
            loc_sq += max_local_dt * max_local_dt;
        }

        // Barrier so gather time measures only MPI communication, not waiting
        // for slower ranks to finish their local cbrt.
        MPI_Barrier(MPI_COMM_WORLD);

        // Step 2 — export local_cbrt to a fixed-size big-endian byte buffer.
        //   Zero-pad at the front (MSB side) so all buffers are cbrt_bytes long.
        unsigned char *local_buf = (unsigned char *)calloc(cbrt_bytes, 1);
        {
            size_t written = 0;
            unsigned char *tmp = (unsigned char *)mpz_export(
                NULL, &written, 1, 1, 0, 0, local_cbrt);
            if (tmp && written > 0 && written <= cbrt_bytes)
            {
                memcpy(local_buf + cbrt_bytes - written, tmp, written);
                free(tmp);
            }
        }
        mpz_clear(local_cbrt);

        // Step 3 — gather all partial cbrt buffers at rank 0.
        //   Total payload: nranks × cbrt_bytes ≈ nranks × 875 bytes (< 10 KB
        //   for 8 ranks with 7000-bit modulo).
        unsigned char *all_bufs = NULL;
        if (rank == 0)
            all_bufs = (unsigned char *)malloc((size_t)nranks * cbrt_bytes);

        auto t_gat0 = std::chrono::high_resolution_clock::now();
        MPI_Gather(local_buf, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                   all_bufs, (int)cbrt_bytes, MPI_UNSIGNED_CHAR,
                   0, MPI_COMM_WORLD);
        auto t_gat1 = std::chrono::high_resolution_clock::now();
        free(local_buf);

        if (rank == 0)
        {
            double dt = std::chrono::duration<double, std::milli>(t_gat1 - t_gat0).count();
            gat_sum += dt;
            gat_sq += dt * dt;
        }

        // Step 4 — rank 0 multiplies the nranks partial results together.
        //   result = ∏_r partial_r  mod M
        if (rank == 0)
        {
            auto t_com0 = std::chrono::high_resolution_clock::now();

            mpz_set_ui(result, 1);
            mpz_t partial;
            mpz_init(partial);
            for (int r = 0; r < nranks; r++)
            {
                mpz_import(partial, cbrt_bytes, 1, 1, 0, 0,
                           all_bufs + (size_t)r * cbrt_bytes);
                mpz_mul(result, result, partial);
                mpz_mod(result, result, modulo);
            }
            mpz_clear(partial);
            free(all_bufs);

            auto t_com1 = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double, std::milli>(t_com1 - t_com0).count();
            com_sum += dt;
            com_sq += dt * dt;
        }

        fl_free(&local_factors);
    }

    // ---- Timing report (rank 0 only) ----
    if (rank == 0)
    {
        printf("\n");
        if (bench_count == 1)
        {
            printf("Factoring:         %.3f ms\n", fac_sum);
            printf("Local cbrt (max):  %.3f ms\n", loc_sum);
            printf("Gather:            %.3f ms\n", gat_sum);
            printf("Combine:           %.3f ms\n", com_sum);
            printf("Reconstruction:    %.3f ms\n", loc_sum + gat_sum + com_sum);
            printf("Total:             %.3f ms\n",
                   fac_sum + loc_sum + gat_sum + com_sum);
        }
        else
        {
            printf("Results over %d runs:\n", bench_count);
            print_timing_stat_moments("Factoring:",
                                      fac_sum, fac_sq, bench_count,
                                      30, true);
            print_timing_stat_moments("Local cbrt (max):",
                                      loc_sum, loc_sq, bench_count,
                                      30, true);
            print_timing_stat_moments("Gather:",
                                      gat_sum, gat_sq, bench_count,
                                      30, true);
            print_timing_stat_moments("Combine (rank 0):",
                                      com_sum, com_sq, bench_count,
                                      30, true);
            double recon_sum = loc_sum + gat_sum + com_sum;
            double recon_sq = loc_sq + gat_sq + com_sq; // approx (correlation ignored)
            print_timing_stat_moments("Reconstruction total:",
                                      recon_sum, recon_sq, bench_count,
                                      30, true);
            double total_sum = fac_sum + recon_sum;
            print_timing_total("Total:", total_sum / bench_count,
                               30, NULL);
        }

        // Verify
        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_cube_root(result, challenge_mod, modulo);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    free(prime_slice);
    mpz_clear(e_inv);
    mpz_clears(challenge, modulo, result, NULL);
    return 1;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // MPI_THREAD_FUNNELED: only the main thread calls MPI functions.
    // OpenMP threads run inside trial_divide_* and compute_local_cbrt
    // but never touch MPI directly — safe with FUNNELED.
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
    {
        fprintf(stderr, "Error: MPI_THREAD_FUNNELED not supported\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char *challenge_file = NULL;
    const char *modulo_file = NULL;
    const char *data_dir = "data/challenges/k32";
    const char *primes_file = "k32_primes.bin";
    int bench_count = 1;
    int run_all = 0;
    int do_precompute = 0;
    run_method_t method = METHOD_SIEVE;

    int positional = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--bench") == 0 && argi + 1 < argc)
        {
            bench_count = atoi(argv[++argi]);
        }
        else if (strcmp(argv[argi], "--all") == 0)
        {
            run_all = 1;
        }
        else if (strcmp(argv[argi], "--data") == 0 && argi + 1 < argc)
        {
            data_dir = argv[++argi];
        }
        else if (strcmp(argv[argi], "--method") == 0 && argi + 1 < argc)
        {
            const char *m = argv[++argi];
            if (strcmp(m, "sieve") == 0)
                method = METHOD_SIEVE;
            else if (strcmp(m, "table") == 0)
                method = METHOD_TABLE;
            else
            {
                if (rank == 0)
                    fprintf(stderr, "Unknown --method '%s'\n", m);
                MPI_Finalize();
                return 1;
            }
        }
        else if (strcmp(argv[argi], "--primes") == 0 && argi + 1 < argc)
        {
            primes_file = argv[++argi];
        }
        else if (strcmp(argv[argi], "--precompute") == 0)
        {
            do_precompute = 1;
        }
        else if (argv[argi][0] != '-')
        {
            if (positional == 0)
            {
                challenge_file = argv[argi];
                positional++;
            }
            else if (positional == 1)
            {
                modulo_file = argv[argi];
                positional++;
            }
            else
            {
                if (rank == 0)
                    fprintf(stderr, "Unexpected argument '%s'\n", argv[argi]);
                MPI_Finalize();
                return 1;
            }
        }
        else
        {
            if (rank == 0)
                fprintf(stderr, "Unknown option '%s'\n", argv[argi]);
            MPI_Finalize();
            return 1;
        }
    }

    if (!run_all && !do_precompute && (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: k32_mpi <challenge> <modulo> [--method sieve|table] [--bench N]\n"
                    "       k32_mpi --all [--data DIR] [--method sieve|table] [--bench N]\n"
                    "       k32_mpi --precompute [--primes FILE]\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("=== k32_mpi ===\n");
        printf("MPI ranks:      %d\n", nranks);
#ifdef _OPENMP
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
#endif
        printf("Method:         %s\n",
               method == METHOD_SIEVE ? "sieve" : "table");
        if (bench_count > 1)
            printf("Benchmark runs: %d\n", bench_count);
        if (run_all)
            printf("Mode:           --all\n");
        fflush(stdout);
    }

    // Small primes — every rank generates these independently (~0.1 ms, no Bcast)
    int sp_count;
    unsigned *sp = generate_small_primes(&sp_count);

    // Partition [65537, 2^32] evenly across MPI ranks
    const uint64_t RANGE_LO = 65537ULL;
    const uint64_t RANGE_HI = (uint64_t)UINT32_MAX;
    const uint64_t range_size = RANGE_HI - RANGE_LO + 1;

    uint64_t my_lo = RANGE_LO + (range_size * (uint64_t)rank) / (uint64_t)nranks;
    uint64_t my_hi = RANGE_LO + (range_size * (uint64_t)(rank + 1)) / (uint64_t)nranks - 1;
    if (rank == nranks - 1)
        my_hi = RANGE_HI;

    if (rank == 0)
    {
        printf("Prime range per rank: ~%.1fM primes\n\n",
               (double)range_size / nranks / 1e6);
        fflush(stdout);
    }

    // Warm up OpenMP thread pool before any timed region
#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    // --precompute: rank 0 generates the prime table file; all ranks then exit
    if (do_precompute)
    {
        if (rank == 0)
        {
            int sp2;
            unsigned *sp2_arr = generate_small_primes(&sp2);
            generate_and_save_prime_table(primes_file, sp2, sp2_arr);
            free(sp2_arr);
        }
        free(sp);
        MPI_Finalize();
        return 0;
    }

    // --method table: rank 0 generates the prime file if it doesn't yet exist,
    // then all ranks wait before each loading their own slice inside solve_one.
    if (method == METHOD_TABLE)
    {
        int table_ok = 1;
        if (rank == 0)
        {
            FILE *test = fopen(primes_file, "rb");
            if (!test)
            {
                printf("[rank 0] Prime table not found — generating...\n");
                fflush(stdout);
                table_ok = generate_and_save_prime_table(primes_file, sp_count, sp) ? 1 : 0;
            }
            else
            {
                fclose(test);
            }
        }
        MPI_Bcast(&table_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!table_ok)
        {
            if (rank == 0)
                fprintf(stderr, "Failed to create prime table\n");
            free(sp);
            MPI_Finalize();
            return 1;
        }
        MPI_Barrier(MPI_COMM_WORLD); // ensure file is fully written before any rank reads
    }

    if (run_all)
    {
        // Run all four standard challenge sizes in one MPI execution.
        // Avoids multiple srun calls in the SLURM script (causes reinit hangs).
        const char *sizes[] = {"1024", "2048", "5000", "7000"};
        for (int s = 0; s < 4; s++)
        {
            char cf[512], mf[512];
            snprintf(cf, sizeof(cf), "%s/challenge_32_%s.txt", data_dir, sizes[s]);
            snprintf(mf, sizeof(mf), "%s/modulo_32_%s.txt", data_dir, sizes[s]);
            solve_one(rank, nranks, cf, mf, bench_count,
                      sp_count, sp, my_lo, my_hi, method, primes_file);
        }
    }
    else
    {
        solve_one(rank, nranks, challenge_file, modulo_file, bench_count,
                  sp_count, sp, my_lo, my_hi, method, primes_file);
    }

    free(sp);
    MPI_Finalize();
    return 0;
}