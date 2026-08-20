#include "prefilter.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// prefilter_argmin_local — exact serial argmin over table[lo, hi).
// ---------------------------------------------------------------------------
void prefilter_argmin_local(mpz_t y_best, int *idx_best,
                            const mpz_t challenge,
                            const blinding_pair_t *table,
                            int lo, int hi,
                            const mpz_t modulo)
{
    // Sentinel: y_best = M (> every real y_i in [0, M)), idx = -1.
    mpz_set(y_best, modulo);
    *idx_best = -1;

    mpz_t y;
    mpz_init(y);
    for (int i = lo; i < hi; i++)
    {
        // y_i = challenge * r_blind[i] mod M
        mpz_mul(y, challenge, table[i].r_blind);
        mpz_mod(y, y, modulo);
        // Strict '<' keeps the LOWEST index on ties, matching an ascending
        // brute-force scan.
        if (mpz_cmp(y, y_best) < 0)
        {
            mpz_set(y_best, y);
            *idx_best = i;
        }
    }
    mpz_clear(y);
}

// ---------------------------------------------------------------------------
// prefilter_simple_select — MPI global argmin (exact, gather-to-rank-0).
// ---------------------------------------------------------------------------
int prefilter_simple_select(mpz_t y_out, int *j_out,
                            const mpz_t challenge,
                            const blinding_pair_t *table, int table_size,
                            int lo, int hi,
                            const mpz_t modulo,
                            int rank, int nranks)
{
    (void)table_size; // range is given by [lo, hi); table_size kept for API clarity

    // -- local argmin (exact) --
    mpz_t y_local;
    mpz_init(y_local);
    int idx_local = -1;
    prefilter_argmin_local(y_local, &idx_local, challenge, table, lo, hi, modulo);

    // Fixed-width big-endian buffer, one per rank. Sized to hold any value < M
    // (the sentinel equals M, hence the +1 byte of headroom).
    size_t nbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;

    unsigned char *sendbuf = (unsigned char *)calloc(nbytes, 1);
    if (!sendbuf)
        MPI_Abort(MPI_COMM_WORLD, 1);
    {
        size_t written = 0;
        unsigned char *tmp =
            (unsigned char *)mpz_export(NULL, &written, 1, 1, 0, 0, y_local);
        if (tmp && written > 0 && written <= nbytes)
            memcpy(sendbuf + nbytes - written, tmp, written);
        if (tmp)
            free(tmp);
    }

    unsigned char *recvbuf = NULL;
    int *recv_idx = NULL;
    if (rank == 0)
    {
        recvbuf = (unsigned char *)malloc(nbytes * (size_t)nranks);
        recv_idx = (int *)malloc(sizeof(int) * (size_t)nranks);
        if (!recvbuf || !recv_idx)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Gather(sendbuf, (int)nbytes, MPI_UNSIGNED_CHAR,
               recvbuf, (int)nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&idx_local, 1, MPI_INT,
               recv_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // -- rank 0 picks the exact global minimum --
    int win_idx = -1;
    if (rank == 0)
    {
        mpz_t y_win, y_r;
        mpz_init_set(y_win, modulo); // +inf sentinel
        mpz_init(y_r);
        for (int r = 0; r < nranks; r++)
        {
            if (recv_idx[r] < 0)
                continue; // rank r owned no entry
            mpz_import(y_r, nbytes, 1, 1, 0, 0, recvbuf + (size_t)r * nbytes);
            if (mpz_cmp(y_r, y_win) < 0)
            {
                mpz_set(y_win, y_r);
                win_idx = recv_idx[r];
            }
        }
        // Pack the winning value back into sendbuf for the broadcast.
        memset(sendbuf, 0, nbytes);
        size_t written = 0;
        unsigned char *tmp =
            (unsigned char *)mpz_export(NULL, &written, 1, 1, 0, 0, y_win);
        if (tmp && written > 0 && written <= nbytes)
            memcpy(sendbuf + nbytes - written, tmp, written);
        if (tmp)
            free(tmp);
        mpz_clears(y_win, y_r, NULL);
    }

    // -- broadcast winner (value + index) to all ranks --
    MPI_Bcast(&win_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sendbuf, (int)nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    mpz_import(y_out, nbytes, 1, 1, 0, 0, sendbuf);

    if (rank == 0)
    {
        free(recvbuf);
        free(recv_idx);
    }
    free(sendbuf);
    mpz_clear(y_local);

    *j_out = win_idx;
    return win_idx;
}