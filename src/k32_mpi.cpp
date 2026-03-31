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

// ---------------------------------------------------------------------------
// Segmented sieve
// ---------------------------------------------------------------------------

#define SEG_SIZE (1 << 19)   // 512K — fits in L2 cache

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
        if (sieve[i]) (*count)++;

    unsigned *primes = (unsigned *)malloc((size_t)(*count) * sizeof(unsigned));
    int idx = 0;
    for (int i = 2; i <= limit; i++)
        if (sieve[i]) primes[idx++] = i;

    free(sieve);
    return primes;
}

// ---------------------------------------------------------------------------
// Factor list for MPI communication
// We pack factors as pairs of uint64_t: [prime, exponent] for easy MPI send
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t prime;
    uint64_t exponent;
} factor_msg_t;

typedef struct {
    factor_msg_t *items;
    int           count;
    int           capacity;
} factor_list_t;

static void factor_list_init(factor_list_t *f)
{
    f->items    = NULL;
    f->count    = 0;
    f->capacity = 0;
}

static void factor_list_append(factor_list_t *f, uint64_t prime, uint64_t exponent)
{
    if (f->count == f->capacity)
    {
        int new_cap = (f->capacity == 0) ? 64 : f->capacity * 2;
        f->items = (factor_msg_t *)realloc(f->items,
                       (size_t)new_cap * sizeof(factor_msg_t));
        f->capacity = new_cap;
    }
    f->items[f->count].prime    = prime;
    f->items[f->count].exponent = exponent;
    f->count++;
}

static void factor_list_free(factor_list_t *f)
{
    free(f->items);
    f->items    = NULL;
    f->count    = 0;
    f->capacity = 0;
}

// ---------------------------------------------------------------------------
// Trial division over a range using segmented sieve + OpenMP
// Each OpenMP thread gets its own segment buffer and local factor list.
// Results are merged into the output list at the end.
// ---------------------------------------------------------------------------

static void trial_divide_range(
    const mpz_t    challenge,
    uint64_t       range_lo,
    uint64_t       range_hi,
    int            sp_count,
    const unsigned *sp,
    factor_list_t  *found)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    factor_list_t *thread_found = (factor_list_t *)malloc(
        (size_t)nthreads * sizeof(factor_list_t));
    for (int t = 0; t < nthreads; t++)
        factor_list_init(&thread_found[t]);

    uint64_t range_size = range_hi - range_lo + 1;

#ifdef _OPENMP
#pragma omp parallel
    {
        int t  = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int t = 0, nt = 1;
#endif
        uint64_t t_lo = range_lo + (range_size * (uint64_t)t)     / (uint64_t)nt;
        uint64_t t_hi = range_lo + (range_size * (uint64_t)(t+1)) / (uint64_t)nt - 1;
        if (t == nt - 1) t_hi = range_hi;

        char *seg = (char *)malloc(SEG_SIZE);

        for (uint64_t lo = t_lo; lo <= t_hi; lo += SEG_SIZE)
        {
            uint64_t hi      = lo + SEG_SIZE - 1;
            if (hi > t_hi)   hi = t_hi;
            size_t seg_len   = (size_t)(hi - lo + 1);

            memset(seg, 1, seg_len);

            // sieve composites
            for (int i = 0; i < sp_count; i++)
            {
                uint64_t p     = sp[i];
                uint64_t start = ((lo + p - 1) / p) * p;
                if (start == p) start += p;
                for (uint64_t j = start; j <= hi; j += p)
                    seg[j - lo] = 0;
            }

            // trial divide by surviving primes
            for (uint64_t c = lo; c <= hi; c++)
            {
                if (!seg[c - lo]) continue;
                if (mpz_tdiv_ui(challenge, (unsigned long)c) != 0) continue;

                int exp = count_exponent(challenge, (unsigned long)c);
                if (exp > 0)
                    factor_list_append(&thread_found[t], c, (uint64_t)exp);
            }
        }
        free(seg);
    }

    // merge thread results
    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < thread_found[t].count; i++)
            factor_list_append(found,
                thread_found[t].items[i].prime,
                thread_found[t].items[i].exponent);
        factor_list_free(&thread_found[t]);
    }
    free(thread_found);
}

// ---------------------------------------------------------------------------
// Cube root reconstruction on rank 0
// Receives all (prime, exponent) pairs, computes powm on-the-fly per factor,
// multiplies results together mod modulo.
// ---------------------------------------------------------------------------

static void reconstruct_cube_root(
    mpz_t                  result,
    const factor_list_t   *all_factors,
    const mpz_t            modulo)
{
    // compute e = 3^-1 mod (modulo-1) once
    mpz_t phi, e, three;
    mpz_inits(phi, e, three, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(three, 3);
    mpz_invert(e, three, phi);

    int n = all_factors->count;
    mpz_t *cbrt_parts = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    for (int i = 0; i < n; i++) mpz_init(cbrt_parts[i]);

    // all powm calls are independent — parallelize with OpenMP
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++)
    {
        mpz_t p_mpz, cbrt_p;
        mpz_inits(p_mpz, cbrt_p, NULL);

        mpz_set_ui(p_mpz, (unsigned long)all_factors->items[i].prime);
        mpz_powm(cbrt_p, p_mpz, e, modulo);

        if (all_factors->items[i].exponent == 1)
            mpz_set(cbrt_parts[i], cbrt_p);
        else
            mpz_powm_ui(cbrt_parts[i], cbrt_p,
                        (unsigned long)all_factors->items[i].exponent, modulo);

        mpz_clears(p_mpz, cbrt_p, NULL);
    }

    // combine
    mpz_set_ui(result, 1);
    for (int i = 0; i < n; i++)
    {
        mpz_mul(result, result, cbrt_parts[i]);
        mpz_mod(result, result, modulo);
        mpz_clear(cbrt_parts[i]);
    }
    free(cbrt_parts);

    mpz_clears(phi, e, three, NULL);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    // MPI_THREAD_FUNNELED: only the main thread calls MPI functions
    // OpenMP threads do pure computation, main thread handles communication
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
    {
        fprintf(stderr, "Error: MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // --- parse arguments (rank 0 only, then broadcast) ---
    const char *challenge_file = "data/challenges/k32/challenge_32_1024.txt";
    const char *modulo_file    = "data/challenges/k32/modulo_32_1024.txt";
    int bench_count            = 1;

    if (rank == 0)
    {
        int positional_index = 0;
        for (int argi = 1; argi < argc; argi++)
        {
            if (strcmp(argv[argi], "--bench") == 0)
            {
                bench_count = atoi(argv[++argi]);
                continue;
            }
            if (positional_index == 0)      challenge_file = argv[argi];
            else if (positional_index == 1) modulo_file    = argv[argi];
            positional_index++;
        }

        printf("=== k32_mpi ===\n");
        printf("MPI ranks:      %d\n", nranks);
#ifdef _OPENMP
        printf("OMP threads:    %d per rank\n", omp_get_max_threads());
        printf("Total workers:  %d\n", nranks * omp_get_max_threads());
#endif
        printf("Challenge file: %s\n", challenge_file);
        printf("Modulo file:    %s\n", modulo_file);
        if (bench_count > 1) printf("Benchmark mode: %d runs\n", bench_count);
    }

    // --- rank 0 reads challenge and modulo, exports as byte arrays ---
    // We broadcast raw bytes to all ranks — avoids GMP file I/O on every rank
    size_t challenge_nbytes = 0, modulo_nbytes = 0;
    unsigned char *challenge_bytes = NULL;
    unsigned char *modulo_bytes    = NULL;

    mpz_t challenge, modulo;
    mpz_inits(challenge, modulo, NULL);

    if (rank == 0)
    {
        read_mpz_from_file(challenge, challenge_file);
        read_mpz_from_file(modulo,    modulo_file);

        challenge_bytes = (unsigned char *)mpz_export(
            NULL, &challenge_nbytes, 1, 1, 0, 0, challenge);
        modulo_bytes = (unsigned char *)mpz_export(
            NULL, &modulo_nbytes, 1, 1, 0, 0, modulo);
    }

    // broadcast sizes then data
    MPI_Bcast(&challenge_nbytes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&modulo_nbytes,    1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        challenge_bytes = (unsigned char *)malloc(challenge_nbytes);
        modulo_bytes    = (unsigned char *)malloc(modulo_nbytes);
    }

    MPI_Bcast(challenge_bytes, (int)challenge_nbytes, MPI_UNSIGNED_CHAR,
              0, MPI_COMM_WORLD);
    MPI_Bcast(modulo_bytes,    (int)modulo_nbytes,    MPI_UNSIGNED_CHAR,
              0, MPI_COMM_WORLD);

    // reconstruct mpz_t from bytes on non-zero ranks
    if (rank != 0)
    {
        mpz_import(challenge, challenge_nbytes, 1, 1, 0, 0, challenge_bytes);
        mpz_import(modulo,    modulo_nbytes,    1, 1, 0, 0, modulo_bytes);
    }

    free(challenge_bytes);
    free(modulo_bytes);

    // --- setup: small primes for sieve (same on all ranks) ---
    int sp_count;
    unsigned *sp = generate_small_primes(&sp_count);

    // --- partition [65537, 2^32] across ranks ---
    // Each rank gets a contiguous slice of the prime range
    const uint64_t RANGE_LO   = 65537;
    const uint64_t RANGE_HI   = (uint64_t)UINT32_MAX;
    const uint64_t range_size = RANGE_HI - RANGE_LO + 1;

    uint64_t my_lo = RANGE_LO + (range_size * (uint64_t)rank)       / (uint64_t)nranks;
    uint64_t my_hi = RANGE_LO + (range_size * (uint64_t)(rank + 1)) / (uint64_t)nranks - 1;
    if (rank == nranks - 1) my_hi = RANGE_HI;

    if (rank == 0)
    {
        printf("\nPrime range per rank: ~%.1fM primes\n",
               (double)range_size / nranks / 1e6);
        printf("Challenge: %zu bits\n", mpz_sizeinbase(challenge, 2));
        printf("Starting factorization...\n\n");
    }

    // warm up OpenMP threads before timing
#ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i++) (void)i;
#endif

    // --- benchmark loop ---
    double total_factor_ms = 0.0;
    double total_gather_ms = 0.0;
    double total_root_ms   = 0.0;

    mpz_t result;
    mpz_init(result);

    for (int bench = 0; bench < bench_count; bench++)
    {
        // synchronize all ranks before timing
        MPI_Barrier(MPI_COMM_WORLD);
        auto t_factor_start = std::chrono::high_resolution_clock::now();

        // --- each rank trial-divides its slice of the prime range ---
        factor_list_t local_factors;
        factor_list_init(&local_factors);

        trial_divide_range(challenge, my_lo, my_hi, sp_count, sp, &local_factors);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t_factor_end = std::chrono::high_resolution_clock::now();
        total_factor_ms += std::chrono::duration<double, std::milli>(
            t_factor_end - t_factor_start).count();

        // --- gather all factors to rank 0 ---
        auto t_gather_start = std::chrono::high_resolution_clock::now();

        // step 1: tell rank 0 how many factors each rank found
        int local_count = local_factors.count;
        int *all_counts = NULL;
        if (rank == 0)
            all_counts = (int *)malloc((size_t)nranks * sizeof(int));

        MPI_Gather(&local_count, 1, MPI_INT,
                   all_counts,   1, MPI_INT, 0, MPI_COMM_WORLD);

        // step 2: rank 0 prepares receive buffer
        factor_list_t all_factors;
        factor_list_init(&all_factors);

        int *displs    = NULL;
        int *recvcounts = NULL;
        factor_msg_t *recvbuf = NULL;
        int total_count = 0;

        if (rank == 0)
        {
            displs     = (int *)malloc((size_t)nranks * sizeof(int));
            recvcounts = (int *)malloc((size_t)nranks * sizeof(int));
            displs[0]  = 0;
            for (int r = 0; r < nranks; r++)
            {
                // each factor_msg_t is 2 × uint64_t = 2 MPI_UINT64_T
                recvcounts[r] = all_counts[r] * 2;
                if (r > 0) displs[r] = displs[r-1] + recvcounts[r-1];
                total_count += all_counts[r];
            }
            recvbuf = (factor_msg_t *)malloc(
                (size_t)total_count * sizeof(factor_msg_t));
        }

        // step 3: gather factor data
        MPI_Gatherv(
            local_factors.items,    local_count * 2, MPI_UINT64_T,
            recvbuf,  recvcounts, displs, MPI_UINT64_T,
            0, MPI_COMM_WORLD);

        auto t_gather_end = std::chrono::high_resolution_clock::now();
        total_gather_ms += std::chrono::duration<double, std::milli>(
            t_gather_end - t_gather_start).count();

        // --- rank 0: reconstruct cube root ---
        if (rank == 0)
        {
            // build all_factors from gathered data
            for (int i = 0; i < total_count; i++)
                factor_list_append(&all_factors,
                                   recvbuf[i].prime,
                                   recvbuf[i].exponent);
            free(recvbuf);
            free(all_counts);
            free(displs);
            free(recvcounts);

            if (bench == 0)
            {
                printf("Total factors found: %d (distinct primes)\n",
                       all_factors.count);

                // verify full factorization
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
            }

            auto t_root_start = std::chrono::high_resolution_clock::now();
            reconstruct_cube_root(result, &all_factors, modulo);
            auto t_root_end = std::chrono::high_resolution_clock::now();
            total_root_ms += std::chrono::duration<double, std::milli>(
                t_root_end - t_root_start).count();

            factor_list_free(&all_factors);
        }

        factor_list_free(&local_factors);
    }

    // --- report (rank 0 only) ---
    if (rank == 0)
    {
        if (bench_count == 1)
        {
            printf("\nFactoring time:      %.3f ms\n", total_factor_ms);
            printf("Gather time:         %.3f ms\n", total_gather_ms);
            printf("Reconstruction time: %.3f ms\n", total_root_ms);
            printf("Total time:          %.3f ms\n",
                   total_factor_ms + total_gather_ms + total_root_ms);
        }
        else
        {
            printf("\nResults over %d runs:\n", bench_count);
            printf("  Average factoring time:      %.3f ms\n",
                   total_factor_ms / bench_count);
            printf("  Average gather time:         %.3f ms\n",
                   total_gather_ms / bench_count);
            printf("  Average reconstruction time: %.3f ms\n",
                   total_root_ms   / bench_count);
            printf("  Average total:               %.3f ms\n",
                   (total_factor_ms + total_gather_ms + total_root_ms) / bench_count);
        }

        // verification
        mpz_t challenge_mod;
        mpz_init(challenge_mod);
        mpz_mod(challenge_mod, challenge, modulo);
        verify_cube_root(result, challenge_mod, modulo);
        mpz_clear(challenge_mod);
    }

    // --- cleanup ---
    mpz_clears(challenge, modulo, result, NULL);
    free(sp);

    MPI_Finalize();
    return 0;
}