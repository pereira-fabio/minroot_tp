/* =====================================================================
 * attack_cuda.cu — GPU batched-search variant of the time-lock attack
 *
 * WHAT THIS IS
 *   tools/timelock/attack.c walks the T-step chain one blinding attempt at
 *   a time: pick r, form y = x*r^e mod p, trial-divide, retry if not
 *   smooth. That inner retry loop is PHASE 1 of the attack (the smoothness
 *   search, thesis sub4:twophases) — the one phase the rest of this project
 *   never measures, because the challenge generator hands the solvers a
 *   smooth challenge by construction.
 *
 *   This file inverts that loop: generate a whole BATCH of candidates,
 *   test them concurrently on the GPU, take the first hit. It is therefore
 *   the only place in the project where search throughput is a measured
 *   quantity rather than a Dickman-density estimate.
 *
 * WHY 64-BIT ONLY
 *   The demo puzzle is generated with -b 64 (scripts/aion/timelock_demo.slurm),
 *   so p, x and every candidate y fit in uint64_t, and every product fits in
 *   __uint128_t. That keeps the device code to plain machine words: no
 *   multi-limb mulmod, no CGBN, no GMP on the device. A wider modulus needs a
 *   192-bit device mulmod for the blinding step; that is deliberately left as
 *   future work and the tool refuses such a puzzle rather than silently
 *   truncating it.
 *
 * DIVISION OF LABOUR
 *   device : blinding (y = x*r^e mod p) + trial division + "is it smooth?"
 *   host   : re-derive the factorisation of the winning candidate ONCE per
 *            step, reconstruct, unblind, verify, derive the key, decrypt.
 *
 *   The kernel reports only WHICH r worked, not the exponent vector. Redoing
 *   the factorisation on the host costs one sweep of the prime table per step
 *   (against millions of sweeps on the device) and buys two things: the device
 *   code stays minimal, and the host re-test becomes an independent check of
 *   the kernel's smoothness verdict.
 *
 * CORRECTNESS GATES (all three must pass, or the run aborts)
 *   1. cbrt table   : cbrt[j]^e == primes[j] (mod p) for every j
 *   2. kernel       : every r the device reports smooth must factor completely
 *                     on the host — a device-only "hit" is a kernel bug
 *   3. per step     : result^e == x (mod p), same gate as attack.c
 *   final           : the decrypted plaintext must come back readable
 *
 * BUILD / RUN: see scripts/iris/timelock_gpu.slurm (no Makefile).
 * ===================================================================== */

#include <cuda_runtime.h>
#include <gmp.h>

extern "C" {
#include "puzzle_io.h"
#include "crypto.h"
#include "primes.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/* Primes staged through shared memory 1024 at a time (4 KB), matching the
 * KF_DELTA_BLOCK granularity used by the solver kernels. Tiling (rather than
 * parking the whole table in shared) keeps the kernel valid for any B. */
#define TILE      1024
#define BLOCKDIM   256

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* ---------------------------------------------------------------------
 * 64-bit modular arithmetic, shared verbatim between host and device.
 * p < 2^64 so every product fits in __uint128_t.
 * ------------------------------------------------------------------- */

__host__ __device__ static inline uint64_t mulmod64(uint64_t a, uint64_t b,
                                                    uint64_t m) {
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) % m);
}

__host__ __device__ static inline uint64_t powmod64(uint64_t b, uint64_t ex,
                                                    uint64_t m) {
    uint64_t r = 1;
    b %= m;
    while (ex) {
        if (ex & 1ULL) r = mulmod64(r, b, m);
        b = mulmod64(b, b, m);
        ex >>= 1;
    }
    return r;
}

/* Counter-based RNG: the batch index is the counter, so a given --seed
 * reproduces the run exactly regardless of how the batches are scheduled.
 * (prefilter_test.cpp needed the same property for its cross-rank invariant.) */
__host__ __device__ static inline uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ---------------------------------------------------------------------
 * Smoothness test, shared logic.
 *
 * Strip every prime factor <= B from t. Two ways out:
 *   t == 1                  -> fully smooth
 *   t < p_j^2 with t > 1    -> t has no factor below p_j, so t is PRIME;
 *                              it is still B-smooth iff t <= B
 * Running the table to exhaustion without either leaves t with a smallest
 * factor > B, hence t > B, so the same "t <= B" test rejects it correctly.
 *
 * attack.c reaches the same verdict via a binary search for the residual in
 * the table; since the table holds every prime <= B, "t <= B" is equivalent.
 * ------------------------------------------------------------------- */

__global__ void search_kernel(uint64_t p, uint64_t x, uint64_t e,
                              const uint32_t * __restrict__ primes,
                              int nprimes, uint64_t bound,
                              uint64_t seed, uint64_t counter_base,
                              int n_cand,
                              unsigned long long *hit_r, int *hit_flag,
                              unsigned long long *hit_count) {
    __shared__ uint32_t s_primes[TILE];

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Threads past the end of the batch still have to reach every
     * __syncthreads() below, so they are flagged done rather than returning. */
    bool done = (gid >= n_cand);

    uint64_t r = 0, t = 0;
    if (!done) {
        r = splitmix64(seed + counter_base + (uint64_t)gid) % p;
        if (r <= 1) r = 2;                       /* r in [2, p-1] as in attack.c */
        t = mulmod64(x, powmod64(r, e, p), p);   /* y = x * r^e mod p */
        if (t == 0) done = true;                 /* cannot happen; guards t/=pj */
    }

    for (int off = 0; off < nprimes; off += TILE) {
        const int n = (nprimes - off < TILE) ? (nprimes - off) : TILE;

        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            s_primes[i] = primes[off + i];
        __syncthreads();

        if (done) continue;

        for (int i = 0; i < n; i++) {
            const uint32_t pj  = s_primes[i];
            const uint64_t pj2 = (uint64_t)pj * (uint64_t)pj;
            if (t < pj2) { done = true; break; }
            while (t % pj == 0) t /= pj;
            if (t == 1)  { done = true; break; }
        }
    }

    if (gid >= n_cand) return;

    const bool smooth = (t == 1) || (t <= bound);
    if (smooth) {
        /* Counting EVERY hit, not just the winner, turns the run into a direct
         * measurement of the Dickman density rho(u) — the quantity the rest of
         * the thesis can only estimate. ~rho(u)*batch adds per launch, so the
         * contention is negligible. */
        atomicAdd(hit_count, 1ULL);
        /* First writer wins; any hit in the batch is as good as any other. */
        if (atomicCAS(hit_flag, 0, 1) == 0)
            *hit_r = (unsigned long long)r;
    }
}

/* Host mirror of the kernel's smoothness test, and the source of the exponent
 * vector used for reconstruction. Returns 1 if fully B-smooth. */
static int host_factor(uint64_t y, const prime_table_t *pt,
                       unsigned int *exponents) {
    for (size_t j = 0; j < pt->count; j++) exponents[j] = 0;

    uint64_t t = y;
    if (t == 0) return 0;

    for (size_t j = 0; j < pt->count; j++) {
        const uint64_t pj = (uint64_t)pt->data[j];
        if (t < pj * pj) break;
        while (t % pj == 0) { t /= pj; exponents[j]++; }
        if (t == 1) break;
    }
    if (t == 1) return 1;

    /* Residual is prime by construction; smooth iff it is in the table. */
    if (t <= (uint64_t)pt->bound) {
        size_t lo = 0, hi = pt->count;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if ((uint64_t)pt->data[mid] < t) lo = mid + 1; else hi = mid;
        }
        if (lo < pt->count && (uint64_t)pt->data[lo] == t) {
            exponents[lo]++;
            return 1;
        }
    }
    return 0;
}

static void usage(void) {
    fprintf(stderr,
"MinRoot time-lock — GPU batched smoothness search\n"
"\n"
"USAGE\n"
"  timelock_gpu <puzzle.mrt> [-B bound] [--batch N] [--seed S]\n"
"\n"
"OPTIONS\n"
"  -B <bound>   smoothness bound B          (default 65536)\n"
"  --batch <N>  candidates per kernel launch (default 4096)\n"
"  --seed <S>   RNG seed; fixed S reproduces a run exactly (default: time)\n"
"\n"
"The puzzle modulus must be <= 64 bits. Wider moduli need a 192-bit device\n"
"mulmod for the blinding step and are not supported here.\n"
"\n"
"BATCH SIZE is the central trade-off. Each step needs exactly ONE smooth\n"
"candidate, but a batch tests N of them at once, so the search overshoots by\n"
"roughly N*rho(u). Small N wastes less work; large N keeps the GPU busy.\n"
"The reported 'overshoot' column is that waste, measured.\n");
}

int main(int argc, char **argv) {
    const char   *puzzle_path = NULL;
    unsigned long B      = 65536UL;
    int           batch  = 4096;
    uint64_t      seed   = 0;
    int           seed_given = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-B")      && i + 1 < argc) B     = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")  && i + 1 < argc) { seed = strtoull(argv[++i], NULL, 10); seed_given = 1; }
        else if (argv[i][0] != '-') puzzle_path = argv[i];
        else { usage(); return 1; }
    }
    if (!puzzle_path) { usage(); return 1; }
    if (batch < BLOCKDIM) batch = BLOCKDIM;
    if (!seed_given) seed = (uint64_t)time(NULL) ^ ((uint64_t)clock() << 21);

    /* ---------------- puzzle ---------------- */
    puzzle_t pz;
    puzzle_init(&pz);
    if (puzzle_read(&pz, puzzle_path) != 0) {
        fprintf(stderr, "Error: could not read puzzle from %s\n", puzzle_path);
        return 1;
    }

    const size_t prime_bits = mpz_sizeinbase(pz.p, 2);
    if (prime_bits > 64 || !mpz_fits_ulong_p(pz.p)) {
        fprintf(stderr,
            "Error: modulus is %zu bits. This GPU search is 64-bit only.\n"
            "       Blinding a wider modulus needs a 192-bit device mulmod\n"
            "       (see the file header). Generate the puzzle with -b 64.\n",
            prime_bits);
        return 1;
    }

    const uint64_t p  = (uint64_t)mpz_get_ui(pz.p);
    const uint64_t e  = (uint64_t)pz.e;
    uint64_t       x  = (uint64_t)mpz_get_ui(pz.x0);
    const unsigned long T = pz.T;

    /* ---------------- device ---------------- */
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("Attack solver — GPU batched smoothness search\n");
    printf("  device : %s (%d SMs, %.1f GB)\n", prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
    printf("  T = %lu iterations, e = %lu, |p| = %zu bits, B = %lu\n",
           T, pz.e, prime_bits, B);
    printf("  batch = %d candidates/launch, seed = %llu\n\n",
           batch, (unsigned long long)seed);

    {
        double log2B = 0.0; unsigned long tmp = B;
        while (tmp >>= 1) log2B++;
        printf("  u = |p|/log2(B) = %.2f\n", (double)prime_bits / log2B);
    }

    /* ---------------- prime table ---------------- */
    printf("  Building prime sieve up to B=%lu ...\n", B);
    prime_table_t primes;
    prime_table_init(&primes, B);
    printf("  %zu primes in table.\n", primes.count);

    uint32_t *h_primes = (uint32_t *)malloc(primes.count * sizeof(uint32_t));
    for (size_t j = 0; j < primes.count; j++) {
        if (primes.data[j] > 0xFFFFFFFFUL) {
            fprintf(stderr, "Error: B > 2^32 unsupported (prime table is u32).\n");
            return 1;
        }
        h_primes[j] = (uint32_t)primes.data[j];
    }

    /* ---------------- cbrt table + gate 1 ---------------- */
    mpz_t phi, e_mpz, e_inv;
    mpz_init(phi);
    mpz_init_set_ui(e_mpz, pz.e);
    mpz_init(e_inv);
    mpz_sub_ui(phi, pz.p, 1);
    if (mpz_invert(e_inv, e_mpz, phi) == 0) {
        fprintf(stderr, "Error: e not invertible mod p-1\n");
        return 1;
    }
    if (!mpz_fits_ulong_p(e_inv)) {
        fprintf(stderr, "Error: e_inv does not fit in 64 bits\n");
        return 1;
    }
    const uint64_t e_inv64 = (uint64_t)mpz_get_ui(e_inv);

    printf("  Precomputing cbrt table (%zu entries) ...\n", primes.count);
    uint64_t *cbrt = (uint64_t *)malloc(primes.count * sizeof(uint64_t));
    for (size_t j = 0; j < primes.count; j++)
        cbrt[j] = powmod64((uint64_t)primes.data[j], e_inv64, p);

    /* GATE 1 — the table is the whole reconstruction, so verify all of it. */
    for (size_t j = 0; j < primes.count; j++) {
        if (powmod64(cbrt[j], e, p) != (uint64_t)primes.data[j] % p) {
            fprintf(stderr, "  [GATE 1 FAIL] cbrt[%zu]^e != %lu (mod p)\n",
                    j, primes.data[j]);
            return 1;
        }
    }
    printf("  cbrt table ready — gate 1 passed (%zu/%zu entries verified).\n\n",
           primes.count, primes.count);

    /* ---------------- upload ---------------- */
    uint32_t           *d_primes = NULL;
    unsigned long long *d_hit_r  = NULL;
    int                *d_flag   = NULL;
    unsigned long long *d_hits   = NULL;
    CUDA_CHECK(cudaMalloc(&d_primes, primes.count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_primes, h_primes, primes.count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_hit_r, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_flag,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hits,  sizeof(unsigned long long)));
    {   /* accumulates across the whole run; only read at the end */
        const unsigned long long z = 0;
        CUDA_CHECK(cudaMemcpy(d_hits, &z, sizeof(z), cudaMemcpyHostToDevice));
    }

    const int grid = (batch + BLOCKDIM - 1) / BLOCKDIM;

    unsigned int *exponents =
        (unsigned int *)malloc(primes.count * sizeof(unsigned int));

    /* ---------------- the chain ---------------- */
    struct timespec t0, tnow;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    unsigned long long total_candidates = 0;
    unsigned long long total_launches   = 0;
    unsigned long long counter          = 0;
    unsigned long report = T / 10; if (report < 1) report = 1;

    for (unsigned long step = 0; step < T; step++) {
        unsigned long long cand_this_step = 0;

        while (1) {
            const int zero = 0;
            CUDA_CHECK(cudaMemcpy(d_flag, &zero, sizeof(int),
                                  cudaMemcpyHostToDevice));

            search_kernel<<<grid, BLOCKDIM>>>(p, x, e, d_primes,
                                              (int)primes.count, (uint64_t)B,
                                              seed, counter, batch,
                                              d_hit_r, d_flag, d_hits);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            counter          += (uint64_t)batch;
            cand_this_step   += (unsigned long long)batch;
            total_candidates += (unsigned long long)batch;
            total_launches++;

            int flag = 0;
            CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (!flag) continue;

            unsigned long long hit = 0;
            CUDA_CHECK(cudaMemcpy(&hit, d_hit_r, sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
            const uint64_t r = (uint64_t)hit;

            /* GATE 2 — independently re-derive the device's verdict. */
            const uint64_t y = mulmod64(x, powmod64(r, e, p), p);
            if (!host_factor(y, &primes, exponents)) {
                fprintf(stderr,
                    "\n  [GATE 2 FAIL] step %lu: device reported r=%llu smooth,\n"
                    "  host says y = x*r^e mod p = %llu is NOT B-smooth.\n"
                    "  This is a kernel bug, not bad luck. Aborting.\n",
                    step, (unsigned long long)r, (unsigned long long)y);
                return 1;
            }

            /* Reconstruct y^(1/e) = prod cbrt[j]^exp[j], unblind, verify. */
            uint64_t result = 1;
            for (size_t j = 0; j < primes.count; j++) {
                if (exponents[j] == 0) continue;
                result = mulmod64(result,
                                  powmod64(cbrt[j], (uint64_t)exponents[j], p), p);
            }
            result = mulmod64(result, powmod64(r, p - 2, p), p);  /* r^-1, p prime */

            /* GATE 3 — same per-step gate as attack.c. */
            if (powmod64(result, e, p) != x) {
                fprintf(stderr,
                    "\n  [GATE 3 FAIL] step %lu: result^e != x (mod p). Aborting.\n",
                    step);
                return 1;
            }

            x = result;
            break;
        }

        if ((step + 1) % report == 0 || step + 1 == T) {
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            const double elapsed = (tnow.tv_sec - t0.tv_sec)
                                 + (tnow.tv_nsec - t0.tv_nsec) * 1e-9;
            printf("  step %5lu/%lu  candidates=%llu  total=%llu  elapsed=%.2fs"
                   "  (%.2f Mcand/s)\n",
                   step + 1, T, cand_this_step, total_candidates, elapsed,
                   total_candidates / elapsed * 1e-6);
            fflush(stdout);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tnow);
    const double total = (tnow.tv_sec - t0.tv_sec)
                       + (tnow.tv_nsec - t0.tv_nsec) * 1e-9;

    /* ---------------- final gate: decrypt ---------------- */
    mpz_t x_final;
    mpz_init_set_ui(x_final, (unsigned long)x);

    unsigned char key[32];
    sha256_mpz(x_final, key);

    size_t         pt_len;
    unsigned char *plaintext = aes256_decrypt(pz.ciphertext, pz.ciphertext_len,
                                              key, pz.iv, &pt_len);
    if (!plaintext) {
        fprintf(stderr, "\nDecryption failed — wrong final x_T. "
                        "Attack produced incorrect result.\n");
        return 1;
    }

    unsigned long long hits_total = 0;
    CUDA_CHECK(cudaMemcpy(&hits_total, d_hits, sizeof(hits_total),
                          cudaMemcpyDeviceToHost));
    const double hit_rate = (double)hits_total / (double)total_candidates;

    printf("\n=== Attack solved in %.3f s ===\n", total);
    printf("Kernel launches          : %llu  (%.2f per step)\n",
           total_launches, (double)total_launches / (double)T);
    printf("Candidates tested        : %llu  (avg %.1f per step)\n",
           total_candidates, (double)total_candidates / (double)T);
    printf("Search throughput        : %.3f Mcandidates/s\n",
           total_candidates / total * 1e-6);
    printf("Sieve work               : %.3f Gdiv/s (candidates x %zu primes)\n",
           (double)total_candidates * (double)primes.count / total * 1e-9,
           primes.count);

    /* PHASE 1 measured. Everywhere else in this project the smoothness density
     * is a Dickman estimate; here it is counted. */
    printf("\nSmooth candidates found  : %llu\n", hits_total);
    printf("Measured hit rate rho(u) : %.6f  (1 in %.1f)\n",
           hit_rate, hit_rate > 0.0 ? 1.0 / hit_rate : 0.0);
    printf("Overshoot                : %.2fx  (%.2f hits found per step,\n"
           "                            only 1 is consumed — a batch cannot\n"
           "                            stop at its first hit)\n",
           (double)hits_total / (double)T, (double)hits_total / (double)T);
    printf("\nMessage: %s\n", (char *)plaintext);

    free(plaintext);
    free(exponents);
    free(cbrt);
    free(h_primes);
    cudaFree(d_primes); cudaFree(d_hit_r); cudaFree(d_flag); cudaFree(d_hits);
    mpz_clear(x_final);
    mpz_clear(phi); mpz_clear(e_mpz); mpz_clear(e_inv);
    prime_table_free(&primes);
    puzzle_free(&pz);
    return 0;
}

// Edit: 2026-07-28
