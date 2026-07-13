// =============================================================================
// crt_test.cpp — correctness gate for the CRT step primitive (crt_common).
//
// Gate: for a random challenge c and modulus M, the parallel crt_step must equal
// the exact serial mpz_powm(c, alpha, M), for every rank count. This validates
// the small-prime CRT reconstruction (the piece the alg_crt driver relies on).
//
// Build:  mpicxx -O2 -std=c++17 -I src/common tests/crt_test.cpp \
//                src/common/crt_common.cpp src/common/common.cpp \
//                src/common/factorization.cpp src/common/solver_common.cpp -lgmp \
//                -o crt_test
// Run:    for np in 1 2 4 8; do mpirun -np $np ./crt_test; done   # all must PASS
// =============================================================================

#include "crt_common.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void bcast_mpz(mpz_t x, size_t nbytes, int rank)
{
    unsigned char *buf = (unsigned char *)calloc(nbytes, 1);
    if (rank == 0) {
        size_t written = 0;
        unsigned char *tmp = (unsigned char *)mpz_export(NULL, &written, 1, 1, 0, 0, x);
        if (tmp && written > 0 && written <= nbytes) memcpy(buf + nbytes - written, tmp, written);
        if (tmp) free(tmp);
    }
    MPI_Bcast(buf, (int)nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (rank != 0) mpz_import(x, nbytes, 1, 1, 0, 0, buf);
    free(buf);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    unsigned long alpha = (argc > 1) ? (unsigned long)atol(argv[1]) : 16;

    mpz_t M, c;
    mpz_inits(M, c, NULL);
    if (rank == 0) {
        gmp_randstate_t st; gmp_randinit_mt(st); gmp_randseed_ui(st, 2718281);
        mpz_ui_pow_ui(M, 2, 255); mpz_nextprime(M, M);
        mpz_urandomm(c, st, M);
        gmp_randclear(st);
    }
    size_t nbytes = 0;
    if (rank == 0) nbytes = (mpz_sizeinbase(M, 2) + 7) / 8 + 1;
    { unsigned long nb = (unsigned long)nbytes; MPI_Bcast(&nb, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); nbytes = nb; }
    bcast_mpz(M, nbytes, rank);
    bcast_mpz(c, nbytes, rank);

    crt_ctx_t ctx;
    crt_ctx_init(&ctx, M, alpha);
    int lo = (int)((long)ctx.k * rank       / nranks);
    int hi = (int)((long)ctx.k * (rank + 1) / nranks);

    // one CRT step: x <- c^alpha mod M
    mpz_t x; mpz_init_set(x, c);
    crt_step(x, &ctx, M, lo, hi, rank, nranks);

    int ok = 1;
    if (rank == 0) {
        mpz_t ref; mpz_init(ref);
        mpz_powm_ui(ref, c, alpha, M);          // exact serial reference
        ok = (mpz_cmp(x, ref) == 0);
        printf("nranks=%d  alpha=%lu  primes=%d  prod=%zu bits  %s\n",
               nranks, alpha, ctx.k, mpz_sizeinbase(ctx.prod, 2),
               ok ? "PASS" : "FAIL (crt_step != mpz_powm)");
        mpz_clear(ref);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    crt_ctx_free(&ctx);
    mpz_clears(M, c, x, NULL);
    MPI_Finalize();
    return ok ? 0 : 1;
}

// Edit: 2026-07-12
