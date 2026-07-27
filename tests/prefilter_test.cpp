// =============================================================================
// prefilter_test.cpp — Phase 0 correctness gate for the simple pre-filter.
//
// Gate (from wiki/planning/simple-prefilter-plan-2026-07-07.md, Phase 0):
//   * parallel argmin (prefilter_simple_select) == single-rank brute-force argmin
//   * identical winning index j at nranks in {1, 2, 4}
//   * the selected y_j is smaller than the table mean (sanity of the heuristic)
//
// Build:  mpicxx -O2 -std=c++17 -I src/common tests/prefilter_test.cpp \
//                src/common/prefilter.cpp src/common/factorization.cpp \
//                src/common/common.cpp -lgmp -o prefilter_test
// Run:    for np in 1 2 4; do mpirun -np $np ./prefilter_test; done
// (must print "PASS" and exit 0 for every np)
// =============================================================================

#include "prefilter.h"
#include "factorization.h"

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
        unsigned char *tmp =
            (unsigned char *)mpz_export(NULL, &written, 1, 1, 0, 0, x);
        if (tmp && written > 0 && written <= nbytes)
            memcpy(buf + nbytes - written, tmp, written);
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

    const int R = 4096;            // |Re|
    const unsigned long e = 5;     // MinRoot exponent

    // --- modulus M (256-bit prime) + a random challenge, agreed on all ranks ---
    mpz_t M, challenge;
    mpz_inits(M, challenge, NULL);
    if (rank == 0) {
        gmp_randstate_t st; gmp_randinit_mt(st); gmp_randseed_ui(st, 12345);
        mpz_ui_pow_ui(M, 2, 255);
        mpz_nextprime(M, M);
        mpz_urandomm(challenge, st, M);
        gmp_randclear(st);
    }
    size_t nbytes = 0;
    if (rank == 0) nbytes = (mpz_sizeinbase(M, 2) + 7) / 8 + 1;
    { unsigned long nb = (unsigned long)nbytes;
      MPI_Bcast(&nb, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); nbytes = nb; }
    bcast_mpz(M, nbytes, rank);
    bcast_mpz(challenge, nbytes, rank);

    // --- blinding table, replicated on every rank (r_blind agreed via bcast) ---
    blinding_pair_t *table = kf_blinding_table_alloc(R, M, e);
    if (!table) { fprintf(stderr, "[rank %d] table alloc failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    // kf_blinding_table_alloc seeds from time(NULL) ^ address, so a fresh table is
    // drawn on every invocation. That is right for the solvers but wrong here: the
    // gate above requires the winning index j to be IDENTICAL at nranks in {1,2,4},
    // and those are three SEPARATE mpirun invocations. With a time-seeded table each
    // run searches a different candidate set, so j legitimately differs and the
    // cross-nranks comparison becomes untestable. Re-draw rank 0's multipliers from a
    // FIXED seed and broadcast them, so the whole table is reproducible run to run.
    if (rank == 0) {
        gmp_randstate_t st; gmp_randinit_mt(st); gmp_randseed_ui(st, 987654321UL);
        mpz_t r;  mpz_init(r);
        for (int i = 0; i < R; i++) {
            do { mpz_urandomm(r, st, M); } while (mpz_sgn(r) == 0);
            mpz_powm_ui(table[i].r_blind, r, e, M);   // r_blind holds r^e mod M
        }
        mpz_clear(r); gmp_randclear(st);
    }
    for (int i = 0; i < R; i++) bcast_mpz(table[i].r_blind, nbytes, rank);

    // --- partition [0, R) across ranks (same scheme as the solvers) ---
    int lo = (int)((long)R * rank / nranks);
    int hi = (int)((long)R * (rank + 1) / nranks);

    mpz_t y_par; mpz_init(y_par);
    int j_par = -1;
    prefilter_simple_select(y_par, &j_par, challenge, table, R, lo, hi, M, rank, nranks);

    int ok = 1;
    if (rank == 0) {
        // brute-force reference over the whole table
        mpz_t y_bf; mpz_init(y_bf);
        int j_bf = -1;
        prefilter_argmin_local(y_bf, &j_bf, challenge, table, 0, R, M);

        if (j_par != j_bf || mpz_cmp(y_par, y_bf) != 0) {
            printf("FAIL: parallel (j=%d) != brute force (j=%d)\n", j_par, j_bf);
            ok = 0;
        }

        // heuristic sanity: winner strictly below the table mean bit-length
        mpz_t sum, mean, y; mpz_inits(sum, mean, y, NULL);
        for (int i = 0; i < R; i++) {
            mpz_mul(y, challenge, table[i].r_blind);
            mpz_mod(y, y, M);
            mpz_add(sum, sum, y);
        }
        mpz_tdiv_q_ui(mean, sum, (unsigned long)R);
        int win_bits  = (int)mpz_sizeinbase(y_bf, 2);
        int mean_bits = (int)mpz_sizeinbase(mean, 2);
        if (mpz_cmp(y_bf, mean) >= 0) {
            printf("FAIL: winner not below table mean (win=%d bits, mean=%d bits)\n",
                   win_bits, mean_bits);
            ok = 0;
        }
        printf("nranks=%d  R=%d  winner j=%d  win=%d bits  mean=%d bits  M=%d bits\n",
               nranks, R, j_bf, win_bits, mean_bits, (int)mpz_sizeinbase(M, 2));
        printf("%s\n", ok ? "PASS" : "PASS-FAILED");
        mpz_clears(sum, mean, y, y_bf, NULL);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    kf_blinding_table_free(table, R);
    mpz_clears(M, challenge, y_par, NULL);
    MPI_Finalize();
    return ok ? 0 : 1;
}

// Edit: 2026-07-12
