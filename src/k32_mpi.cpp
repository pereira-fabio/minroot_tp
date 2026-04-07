// =============================================================================
// k32_mpi.cpp — MPI+OpenMP hybrid cube root solver for 32-bit prime factors
//
// Architecture:
//   MPI layer:    splits the prime range [65537, 2^32] across nodes
//   OpenMP layer: parallelizes sieve + trial division within each node
//
// Mapping to hardware (AMD EPYC 7452, 2 NUMA nodes × 32 cores):
//   --ntasks-per-node=2  →  one MPI rank per NUMA node
//   --cpus-per-task=32   →  32 OpenMP threads per rank
//   Total: 8 ranks × 32 threads = 256 cores for 4-node jobs
//
// Communication pattern — designed to be minimal:
//   MPI_Bcast:   challenge + modulo bytes → all ranks (once per challenge)
//   MPI_Barrier: synchronize before and after factoring (for timing)
//   MPI_Gather:  factor counts from each rank → rank 0
//   MPI_Gatherv: factor (prime,exponent) pairs → rank 0  (~34 pairs total)
//   No communication during the hot trial division loop.
//
// Key design decision — broadcast bytes, not GMP integers:
//   mpz_inp_raw / mpz_out_raw are not thread-safe and only work with files.
//   We export the challenge and modulo as raw byte arrays on rank 0,
//   broadcast the bytes, then reconstruct with mpz_import on other ranks.
//   This is the standard pattern for distributing GMP integers via MPI.
//
// Modes:
//   Single challenge:  k32_mpi <challenge> <modulo> [--bench N]
//   All sizes:         k32_mpi --all [--data DIR] [--bench N]
//   --all runs all four sizes (1024/2048/5000/7000) in one MPI execution,
//   avoiding the multiple-srun-per-job issue on SLURM.
// =============================================================================

#include "k_common.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#define SEG_SIZE (1 << 19) // 512K sieve segment — fits in L2 cache

// ---------------------------------------------------------------------------
// Small prime generation — same as k32.cpp
// Primes up to 65536 are needed to correctly sieve any 32-bit range.
// Computed once on every rank (cheap, ~0.1ms).
// ---------------------------------------------------------------------------

static unsigned *generate_small_primes(int *count)
{
    const int limit = 65536;
    char *sieve = (char *)calloc(limit + 1, 1);
    memset(sieve, 1, limit + 1);
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
            primes[idx++] = i;
    free(sieve);
    return primes;
}

// ---------------------------------------------------------------------------
// Factor list — packed as uint64_t pairs for MPI transport
//
// factor_msg_t { prime, exponent } is exactly 2 × uint64_t = 16 bytes.
// This lets us send the whole list as a flat MPI_UINT64_T array using
// MPI_Gatherv, with count = list.count * 2.  No custom MPI derived types needed.
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

static void factor_list_init(factor_list_t *f)
{
    f->items = NULL;
    f->count = 0;
    f->capacity = 0;
}

static void factor_list_append(factor_list_t *f, uint64_t prime, uint64_t exp)
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

static void factor_list_free(factor_list_t *f)
{
    free(f->items);
    f->items = NULL;
    f->count = 0;
    f->capacity = 0;
}

// ---------------------------------------------------------------------------
// trial_divide_range — segmented sieve + OpenMP (same as k32.cpp)
//
// Each MPI rank calls this on its own sub-range of [65537, 2^32].
// Within the rank, OpenMP further divides the sub-range across threads.
// No MPI communication happens inside this function — all parallelism is
// via shared memory (OpenMP), not message passing.
// ---------------------------------------------------------------------------

static void trial_divide_range(
    const mpz_t challenge,
    uint64_t range_lo, uint64_t range_hi,
    int sp_count, const unsigned *sp,
    factor_list_t *found)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    // per-thread factor lists avoid any locking during the hot loop
    factor_list_t *tf = (factor_list_t *)malloc(
        (size_t)nthreads * sizeof(factor_list_t));
    for (int t = 0; t < nthreads; t++)
        factor_list_init(&tf[t]);

    uint64_t range_size = range_hi - range_lo + 1;

#ifdef _OPENMP
#pragma omp parallel
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
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
                int exp = count_exponent(challenge, (unsigned long)c);
                if (exp > 0)
                    factor_list_append(&tf[t], c, (uint64_t)exp);
            }
        }
        free(seg);
    }

    // merge thread-local results
    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            factor_list_append(found, tf[t].items[i].prime,
                               tf[t].items[i].exponent);
        factor_list_free(&tf[t]);
    }
    free(tf);
}

// ---------------------------------------------------------------------------
// reconstruct_cube_root — on-the-fly powm, OpenMP-parallelized (rank 0 only)
//
// Called only on rank 0 after gathering all factors.
// e = 3^-1 mod (M-1) is computed once; each powm call is independent.
// For 7000-bit modulo with 229 factors, reconstruction takes ~650ms —
// the remaining bottleneck after factoring is distributed via MPI.
// ---------------------------------------------------------------------------

static void reconstruct_cube_root(
    mpz_t result,
    const factor_list_t *all_factors,
    const mpz_t modulo)
{
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi); // e = 3^-1 mod (M-1), computed once

    int n = all_factors->count;
    mpz_t *parts = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    for (int i = 0; i < n; i++)
        mpz_init(parts[i]);

    // independent powm calls — parallelize with OpenMP on rank 0's cores
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++)
    {
        mpz_t p_mpz, cbrt_p;
        mpz_inits(p_mpz, cbrt_p, NULL);
        mpz_set_ui(p_mpz, (unsigned long)all_factors->items[i].prime);
        mpz_powm(cbrt_p, p_mpz, e, modulo); // cbrt(p) = p^e mod M

        if (all_factors->items[i].exponent == 1)
            mpz_set(parts[i], cbrt_p);
        else
            mpz_powm_ui(parts[i], cbrt_p,
                        (unsigned long)all_factors->items[i].exponent, modulo);

        mpz_clears(p_mpz, cbrt_p, NULL);
    }

    mpz_set_ui(result, 1);
    for (int i = 0; i < n; i++)
    {
        mpz_mul(result, result, parts[i]);
        mpz_mod(result, result, modulo);
        mpz_clear(parts[i]);
    }
    free(parts);
    mpz_clears(phi, e, three, NULL);
}

// ---------------------------------------------------------------------------
// solve_one — solve a single (challenge, modulo) pair across all MPI ranks
//
// Can be called multiple times in one MPI execution (used by --all mode).
// MPI_Init/Finalize are NOT called here — they bracket the entire program.
//
// Flow:
//   rank 0: read files → export bytes
//   all:    MPI_Bcast bytes → mpz_import
//   all:    trial_divide_range on own slice of [65537, 2^32]
//   all:    MPI_Barrier (synchronize timing)
//   all:    MPI_Gather factor counts → rank 0
//   all:    MPI_Gatherv factor data  → rank 0
//   rank 0: reconstruct + verify
// ---------------------------------------------------------------------------

static int solve_one(
    int rank, int nranks,
    const char *challenge_file, const char *modulo_file,
    int bench_count, int sp_count, const unsigned *sp,
    uint64_t my_lo, uint64_t my_hi)
{
    if (rank == 0)
    {
        printf("\n=== %s ===\n", challenge_file);
        fflush(stdout);
    }

    // --- rank 0 reads and serializes challenge/modulo to byte arrays ---
    size_t challenge_nbytes = 0, modulo_nbytes = 0;
    unsigned char *challenge_bytes = NULL, *modulo_bytes = NULL;

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);

    if (rank == 0)
    {
        read_mpz_from_file(challenge, challenge_file);
        read_mpz_from_file(modulo, modulo_file);
        // mpz_export: converts GMP integer to a plain byte array
        // arguments: (output buf, &size, order=1=MSB first, word_size=1, endian=0=native, nails=0, op)
        challenge_bytes = (unsigned char *)mpz_export(
            NULL, &challenge_nbytes, 1, 1, 0, 0, challenge);
        modulo_bytes = (unsigned char *)mpz_export(
            NULL, &modulo_nbytes, 1, 1, 0, 0, modulo);
    }

    // broadcast sizes first, then allocate on non-zero ranks, then data
    MPI_Bcast(&challenge_nbytes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&modulo_nbytes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        challenge_bytes = (unsigned char *)malloc(challenge_nbytes);
        modulo_bytes = (unsigned char *)malloc(modulo_nbytes);
    }

    MPI_Bcast(challenge_bytes, (int)challenge_nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(modulo_bytes, (int)modulo_nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // reconstruct GMP integers from bytes on non-zero ranks
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

    double total_factor_ms = 0.0, total_gather_ms = 0.0, total_root_ms = 0.0;
    int success = 0;

    mpz_t result;
    mpz_init(result);

    for (int bench = 0; bench < bench_count; bench++)
    {
        factor_list_t local_factors;
        factor_list_init(&local_factors);

        // --- factoring phase: each rank independently sieves its slice ---
        MPI_Barrier(MPI_COMM_WORLD); // sync before timing
        auto t0 = std::chrono::high_resolution_clock::now();

        trial_divide_range(challenge, my_lo, my_hi,
                           sp_count, sp, &local_factors);

        MPI_Barrier(MPI_COMM_WORLD); // sync after — slowest rank sets the time
        auto t1 = std::chrono::high_resolution_clock::now();
        total_factor_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- gather phase: collect all factors at rank 0 ---
        // Two-step: first gather counts (so rank 0 knows buffer size),
        //           then gather the actual factor data with MPI_Gatherv.
        auto t2 = std::chrono::high_resolution_clock::now();

        int local_count = local_factors.count;
        int *all_counts = (rank == 0)
                              ? (int *)malloc((size_t)nranks * sizeof(int))
                              : NULL;

        // step 1: each rank sends its local_count to rank 0
        MPI_Gather(&local_count, 1, MPI_INT,
                   all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        factor_list_t all_factors;
        factor_list_init(&all_factors);
        int *displs = NULL, *recvcounts = NULL;
        factor_msg_t *recvbuf = NULL;
        int total_count = 0;

        if (rank == 0)
        {
            // build displacement array for MPI_Gatherv
            // each factor_msg_t = 2 × uint64_t, so recvcounts[r] = all_counts[r] * 2
            displs = (int *)malloc((size_t)nranks * sizeof(int));
            recvcounts = (int *)malloc((size_t)nranks * sizeof(int));
            displs[0] = 0;
            for (int r = 0; r < nranks; r++)
            {
                recvcounts[r] = all_counts[r] * 2;
                if (r > 0)
                    displs[r] = displs[r - 1] + recvcounts[r - 1];
                total_count += all_counts[r];
            }
            recvbuf = (factor_msg_t *)malloc(
                (size_t)total_count * sizeof(factor_msg_t));
        }

        // step 2: each rank sends its factor list; rank 0 receives all
        MPI_Gatherv(
            local_factors.items, local_count * 2, MPI_UINT64_T,
            recvbuf, recvcounts, displs, MPI_UINT64_T,
            0, MPI_COMM_WORLD);

        auto t3 = std::chrono::high_resolution_clock::now();
        total_gather_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // --- reconstruction phase (rank 0 only) ---
        if (rank == 0)
        {
            // unpack gathered flat array back into a factor_list_t
            for (int i = 0; i < total_count; i++)
                factor_list_append(&all_factors,
                                   recvbuf[i].prime, recvbuf[i].exponent);
            free(recvbuf);
            free(all_counts);
            free(displs);
            free(recvcounts);

            if (bench == 0)
            {
                printf("Factors found: %d (distinct primes)\n", all_factors.count);
                // sanity check: divide out all factors and verify remainder = 1
                mpz_t remaining;
                mpz_init_set(remaining, challenge);
                for (int i = 0; i < all_factors.count; i++)
                    for (uint64_t e = 0; e < all_factors.items[i].exponent; e++)
                        mpz_divexact_ui(remaining, remaining,
                                        (unsigned long)all_factors.items[i].prime);
                if (mpz_cmp_ui(remaining, 1) != 0)
                {
                    printf("WARNING: Not fully factored! Remaining: ");
                    gmp_printf("%Zd\n", remaining);
                }
                mpz_clear(remaining);
                fflush(stdout);
            }

            auto t4 = std::chrono::high_resolution_clock::now();
            reconstruct_cube_root(result, &all_factors, modulo);
            auto t5 = std::chrono::high_resolution_clock::now();
            total_root_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

            factor_list_free(&all_factors);
            success = 1;
        }

        factor_list_free(&local_factors);
    }

    // --- report and verify (rank 0 only) ---
    if (rank == 0)
    {
        printf("\n");
        if (bench_count == 1)
        {
            printf("Factoring time:      %.3f ms\n", total_factor_ms);
            printf("Gather time:         %.3f ms\n", total_gather_ms);
            printf("Reconstruction time: %.3f ms\n", total_root_ms);
            printf("Total time:          %.3f ms\n",
                   total_factor_ms + total_gather_ms + total_root_ms);
        }
        else
        {
            printf("Results over %d runs:\n", bench_count);
            printf("  Average factoring time:      %.3f ms\n", total_factor_ms / bench_count);
            printf("  Average gather time:         %.3f ms\n", total_gather_ms / bench_count);
            printf("  Average reconstruction time: %.3f ms\n", total_root_ms / bench_count);
            printf("  Average total:               %.3f ms\n",
                   (total_factor_ms + total_gather_ms + total_root_ms) / bench_count);
        }

        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_cube_root(result, challenge_mod, modulo);
        mpz_clear(challenge_mod);
        fflush(stdout);
    }

    mpz_clears(challenge, modulo, result, NULL);
    return success;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // MPI_THREAD_FUNNELED: only the main thread calls MPI functions.
    // OpenMP threads run inside trial_divide_range and reconstruct_cube_root
    // but never touch MPI directly — this is safe with FUNNELED.
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

    // all ranks parse arguments identically — no need to broadcast
    const char *challenge_file = NULL;
    const char *modulo_file = NULL;
    const char *data_dir = "data/challenges/k32";
    int bench_count = 1, run_all = 0;

    int positional_index = 0;
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--bench") == 0)
        {
            if (argi + 1 >= argc)
            {
                if (rank == 0)
                    fprintf(stderr, "Missing --bench value\n");
                MPI_Finalize();
                return 1;
            }
            bench_count = atoi(argv[++argi]);
            continue;
        }
        if (strcmp(argv[argi], "--all") == 0)
        {
            run_all = 1;
            continue;
        }
        if (strcmp(argv[argi], "--data") == 0)
        {
            if (argi + 1 >= argc)
            {
                if (rank == 0)
                    fprintf(stderr, "Missing --data value\n");
                MPI_Finalize();
                return 1;
            }
            data_dir = argv[++argi];
            continue;
        }
        if (positional_index == 0)
            challenge_file = argv[argi];
        else if (positional_index == 1)
            modulo_file = argv[argi];
        else
        {
            if (rank == 0)
                fprintf(stderr, "Unexpected argument '%s'\n", argv[argi]);
            MPI_Finalize();
            return 1;
        }
        positional_index++;
    }

    if (!run_all && (!challenge_file || !modulo_file))
    {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: k32_mpi <challenge> <modulo> [--bench N]\n"
                    "       k32_mpi --all [--data DIR] [--bench N]\n");
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
        if (bench_count > 1)
            printf("Benchmark runs: %d\n", bench_count);
        if (run_all)
            printf("Mode:           --all\n");
        fflush(stdout);
    }

    // small primes computed independently on every rank (identical result,
    // avoids an MPI broadcast for a negligible 0.1ms computation)
    int sp_count;
    unsigned *sp = generate_small_primes(&sp_count);

    // partition [65537, 2^32] across ranks — fixed for the entire program lifetime
    // so the partition is computed once here and passed into every solve_one call
    const uint64_t RANGE_LO = 65537;
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

    // warm up OpenMP thread pool once before any timed region
#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++)
        (void)i;
#endif

    if (run_all)
    {
        // run all four standard sizes sequentially within one MPI execution.
        // This avoids multiple srun calls in the SLURM script, which causes
        // MPI reinitialisation issues on some cluster configurations.
        const char *sizes[] = {"1024", "2048", "5000", "7000"};
        for (int s = 0; s < 4; s++)
        {
            char cf[512], mf[512];
            snprintf(cf, sizeof(cf), "%s/challenge_32_%s.txt", data_dir, sizes[s]);
            snprintf(mf, sizeof(mf), "%s/modulo_32_%s.txt", data_dir, sizes[s]);
            solve_one(rank, nranks, cf, mf, bench_count,
                      sp_count, sp, my_lo, my_hi);
        }
    }
    else
    {
        solve_one(rank, nranks, challenge_file, modulo_file, bench_count,
                  sp_count, sp, my_lo, my_hi);
    }

    free(sp);
    MPI_Finalize();
    return 0;
}