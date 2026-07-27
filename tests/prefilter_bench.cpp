// =============================================================================
// prefilter_bench.cpp — is the simple pre-filter PROFITABLE?
//
// prefilter_test.cpp answers "is the parallel selection correct?".  This answers
// the different, and for the thesis more interesting, question: does selecting a
// candidate cost less than the sieving that selection saves?
//
// The argument it closes (see wiki simple-prefilter / thesis sub4:prefilter):
//   Selecting argmin over |Re| candidates shortens the challenge by ~log2|Re|
//   bits.  Shorter challenge -> smaller u = |x|/log2(B) -> larger Dickman rho(u)
//   -> fewer expected blinding attempts.  Writing C_pf for one selection round
//   and C_sieve for one full B-smooth sieve, pre-filtering wins iff
//
//        2^(k-b) * (C_pf + C_sieve)  <  2^k * C_sieve
//   i.e.      C_pf  <  (2^b - 1) * C_sieve
//
//   where b is the number of BITS OF SEARCH saved (not bits off the candidate).
//   For N=512, B=2^35 the standard estimate is b ~ 2, giving C_pf < 3*C_sieve.
//   So the whole question reduces to a ratio this program measures directly.
//
// What it reports, for a sweep of |Re|:
//   * bit saving vs |M| and vs the sample mean, with mean +- std over T trials
//     (one draw is a single sample of a minimum -- a curve is the real evidence)
//   * the predicted log2|Re|, for comparison
//   * C_pf: wall-clock of one selection round
//   * C_pf / C_sieve against the break-even threshold
//
// It touches NOTHING else: a standalone binary, linked against prefilter.cpp
// only.  No solver is rebuilt and no benchmark result can change.
//
// Build:
//   mpicxx -O2 -std=c++17 -I src/common tests/prefilter_bench.cpp \
//          src/common/prefilter.cpp src/common/factorization.cpp \
//          src/common/common.cpp -lgmp -o prefilter_bench
// Run:
//   mpirun -np 4 ./prefilter_bench [--bits 256] [--trials 20] [--sieve-ms 790]
// =============================================================================

#include "prefilter.h"
#include "factorization.h"

#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    int    mod_bits  = 256;    // modulus size
    int    trials    = 20;     // independent draws per |Re|
    double sieve_ms  = 790.0;  // measured k32 single-rank sieve, 24576b (factor_ms at N1)
    const unsigned long e = 5;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--bits")     && i+1 < argc) mod_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trials")   && i+1 < argc) trials   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sieve-ms") && i+1 < argc) sieve_ms = atof(argv[++i]);
    }

    const int R_LIST[] = { 64, 256, 1024, 4096, 16384 };
    const int N_R      = (int)(sizeof(R_LIST) / sizeof(R_LIST[0]));

    // --- modulus M, agreed on all ranks -------------------------------------
    mpz_t M; mpz_init(M);
    if (rank == 0) {
        mpz_ui_pow_ui(M, 2, (unsigned long)(mod_bits - 1));
        mpz_nextprime(M, M);
    }
    size_t nbytes = 0;
    if (rank == 0) nbytes = (mpz_sizeinbase(M, 2) + 7) / 8 + 1;
    { unsigned long nb = (unsigned long)nbytes;
      MPI_Bcast(&nb, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); nbytes = nb; }
    bcast_mpz(M, nbytes, rank);
    const int M_bits = (int)mpz_sizeinbase(M, 2);

    if (rank == 0) {
        printf("============================================================\n");
        printf("prefilter_bench   nranks=%d  |M|=%d bits  trials=%d\n",
               nranks, M_bits, trials);
        printf("C_sieve reference = %.1f ms  (override with --sieve-ms)\n", sieve_ms);
        printf("============================================================\n\n");
        printf("%7s %9s %16s %16s %11s %10s\n",
               "|Re|", "log2|Re|", "saving vs |M|", "saving vs mean", "C_pf (ms)", "C_pf/C_s");
        printf("%7s %9s %16s %16s %11s %10s\n",
               "-----", "--------", "---------------", "---------------",
               "----------", "---------");
    }

    for (int ri = 0; ri < N_R; ri++) {
        const int R = R_LIST[ri];

        double sum_sv_M = 0, sum_sv_M2 = 0;   // saving vs |M|
        double sum_sv_mu = 0;                 // saving vs sample mean
        double sum_cpf = 0;

        for (int t = 0; t < trials; t++) {
            // --- fresh challenge + blinding table, deterministic per trial ---
            mpz_t challenge; mpz_init(challenge);
            blinding_pair_t *table = kf_blinding_table_alloc(R, M, e);
            if (!table) { fprintf(stderr, "[rank %d] table alloc failed\n", rank);
                          MPI_Abort(MPI_COMM_WORLD, 1); }

            if (rank == 0) {
                gmp_randstate_t st; gmp_randinit_mt(st);
                gmp_randseed_ui(st, 1000003UL * (unsigned long)(t + 1) + (unsigned long)R);
                mpz_urandomm(challenge, st, M);
                mpz_t r; mpz_init(r);
                for (int i = 0; i < R; i++) {
                    do { mpz_urandomm(r, st, M); } while (mpz_sgn(r) == 0);
                    mpz_powm_ui(table[i].r_blind, r, e, M);   // r_blind holds r^e mod M
                }
                mpz_clear(r); gmp_randclear(st);
            }
            bcast_mpz(challenge, nbytes, rank);
            for (int i = 0; i < R; i++) bcast_mpz(table[i].r_blind, nbytes, rank);

            // --- timed selection round (this is C_pf) ------------------------
            int lo = (int)((long)R * rank / nranks);
            int hi = (int)((long)R * (rank + 1) / nranks);

            mpz_t y_win; mpz_init(y_win);
            int j_win = -1;

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            prefilter_simple_select(y_win, &j_win, challenge, table, R,
                                    lo, hi, M, rank, nranks);
            MPI_Barrier(MPI_COMM_WORLD);
            double cpf_ms = (MPI_Wtime() - t0) * 1000.0;

            if (rank == 0) {
                // sample mean of the candidate set, for the second saving column
                mpz_t sum, mean, y; mpz_inits(sum, mean, y, NULL);
                for (int i = 0; i < R; i++) {
                    mpz_mul(y, challenge, table[i].r_blind);
                    mpz_mod(y, y, M);
                    mpz_add(sum, sum, y);
                }
                mpz_tdiv_q_ui(mean, sum, (unsigned long)R);

                double sv_M  = (double)M_bits - (double)mpz_sizeinbase(y_win, 2);
                double sv_mu = (double)mpz_sizeinbase(mean, 2)
                             - (double)mpz_sizeinbase(y_win, 2);
                sum_sv_M  += sv_M;  sum_sv_M2 += sv_M * sv_M;
                sum_sv_mu += sv_mu;
                sum_cpf   += cpf_ms;
                mpz_clears(sum, mean, y, NULL);
            }

            mpz_clear(y_win);
            kf_blinding_table_free(table, R);
            mpz_clear(challenge);
        }

        if (rank == 0) {
            double mu_M  = sum_sv_M / trials;
            double var   = sum_sv_M2 / trials - mu_M * mu_M;
            double sd    = var > 0 ? sqrt(var) : 0.0;
            double mu_mu = sum_sv_mu / trials;
            double cpf   = sum_cpf / trials;
            printf("%7d %9.0f %10.2f+-%-4.2f %16.2f %11.3f %10.4f\n",
                   R, log2((double)R), mu_M, sd, mu_mu, cpf, cpf / sieve_ms);
        }
    }

    // --- verdict -------------------------------------------------------------
    if (rank == 0) {
        printf("\n------------------------------------------------------------\n");
        printf("Break-even: pre-filtering pays iff C_pf < (2^b - 1) * C_sieve,\n");
        printf("where b is the BITS OF SEARCH saved. For N=512, B=2^35 the\n");
        printf("standard estimate is b ~ 2, i.e. C_pf < %.1f ms here.\n", 3.0 * sieve_ms);
        printf("Compare the C_pf column above against that threshold.\n");
        printf("------------------------------------------------------------\n");
        printf("NOTE: 'saving vs |M|' is the quantity the theory predicts --\n");
        printf("the minimum of |Re| roughly-uniform values on [0,M) sits near\n");
        printf("M/|Re|, so the saving should track log2|Re|. 'saving vs mean'\n");
        printf("is reported for continuity with prefilter_test and runs ~1 bit\n");
        printf("lower, since the sample mean is itself below |M|.\n");
    }

    mpz_clear(M);
    MPI_Finalize();
    return 0;
}

// Edit: 2026-07-27
