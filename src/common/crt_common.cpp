// =============================================================================
// crt_common.cpp — CRT-based modular exponentiation. See crt_common.h.
// =============================================================================

#include "crt_common.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// 64-bit modular exponentiation (base, exp, mod all < 2^32 -> __int128 mulmod).
// ---------------------------------------------------------------------------
static uint64_t powmod_u64(uint64_t base, uint64_t exp, uint64_t mod)
{
    uint64_t result = 1 % mod;
    base %= mod;
    while (exp) {
        if (exp & 1) result = (uint64_t)((unsigned __int128)result * base % mod);
        base = (uint64_t)((unsigned __int128)base * base % mod);
        exp >>= 1;
    }
    return result;
}

// ---------------------------------------------------------------------------
// crt_ctx_init
// ---------------------------------------------------------------------------
void crt_ctx_init(crt_ctx_t *ctx, const mpz_t modulo, unsigned long alpha)
{
    ctx->alpha = alpha;

    // bound = M^alpha  (strictly > (M-1)^alpha >= acc^alpha for any acc < M).
    mpz_t bound;
    mpz_init(bound);
    mpz_pow_ui(bound, modulo, alpha);

    // Collect distinct ~31-bit primes (start at 2^30) until prod > bound.
    int cap = 16;
    ctx->primes = (uint32_t *)malloc(cap * sizeof(uint32_t));
    ctx->k = 0;

    mpz_init_set_ui(ctx->prod, 1);
    mpz_t p;
    mpz_init_set_ui(p, 1u << 30);

    while (mpz_cmp(ctx->prod, bound) <= 0) {
        mpz_nextprime(p, p);
        if (mpz_cmp_ui(p, 0xFFFFFFFFUL) >= 0) {
            // Would need a 32-bit+ prime — alpha too large for this modulus.
            fprintf(stderr,
                "crt_ctx_init: prime set exhausted 32-bit range (alpha=%lu, "
                "|M|=%zu bits). Lower alpha.\n",
                alpha, mpz_sizeinbase(modulo, 2));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (ctx->k == cap) {
            cap *= 2;
            ctx->primes = (uint32_t *)realloc(ctx->primes, cap * sizeof(uint32_t));
        }
        ctx->primes[ctx->k++] = (uint32_t)mpz_get_ui(p);
        mpz_mul_ui(ctx->prod, ctx->prod, mpz_get_ui(p));
    }
    mpz_clear(p);
    mpz_clear(bound);

    // Precompute K_i = M_i * P_i, where P_i = prod / p_i and M_i = P_i^-1 mod p_i.
    ctx->K = (mpz_t *)malloc((size_t)ctx->k * sizeof(mpz_t));
    mpz_t Pi, Mi, pm;
    mpz_inits(Pi, Mi, pm, NULL);
    for (int i = 0; i < ctx->k; i++) {
        mpz_divexact_ui(Pi, ctx->prod, ctx->primes[i]); // P_i (exact)
        mpz_set_ui(pm, ctx->primes[i]);
        mpz_invert(Mi, Pi, pm);                          // M_i = P_i^-1 mod p_i
        mpz_init(ctx->K[i]);
        mpz_mul(ctx->K[i], Mi, Pi);                      // K_i = M_i * P_i
    }
    mpz_clears(Pi, Mi, pm, NULL);
}

void crt_ctx_free(crt_ctx_t *ctx)
{
    for (int i = 0; i < ctx->k; i++) mpz_clear(ctx->K[i]);
    free(ctx->K);
    free(ctx->primes);
    mpz_clear(ctx->prod);
    ctx->K = NULL;
    ctx->primes = NULL;
    ctx->k = 0;
}

// ---------------------------------------------------------------------------
// export_fixed / import_fixed — big-endian fixed-width mpz <-> byte buffer.
// ---------------------------------------------------------------------------
static void export_fixed(unsigned char *buf, size_t nbytes, const mpz_t x)
{
    memset(buf, 0, nbytes);
    size_t written = 0;
    unsigned char *tmp = (unsigned char *)mpz_export(NULL, &written, 1, 1, 0, 0, x);
    if (tmp && written > 0 && written <= nbytes)
        memcpy(buf + nbytes - written, tmp, written);
    if (tmp) free(tmp);
}

// ---------------------------------------------------------------------------
// crt_step: c_inout <- c_inout^alpha mod M
// ---------------------------------------------------------------------------
void crt_step(mpz_t c_inout, const crt_ctx_t *ctx, const mpz_t modulo,
              int lo, int hi, int rank, int nranks)
{
    // -- local partial reconstruction sum over this rank's prime slice --
    mpz_t local;
    mpz_init_set_ui(local, 0);
    for (int i = lo; i < hi; i++) {
        uint64_t pi = ctx->primes[i];
        uint64_t ci = (uint64_t)mpz_tdiv_ui(c_inout, (unsigned long)pi); // c mod p_i
        uint64_t ri = powmod_u64(ci, ctx->alpha, pi);                    // c_i^alpha mod p_i
        if (ri) mpz_addmul_ui(local, ctx->K[i], (unsigned long)ri);      // += r_i * K_i
    }
    mpz_mod(local, local, ctx->prod); // keep partial < prod (sum mod prod later)

    // -- gather partials to rank 0, sum, reconstruct, reduce mod M --
    size_t nbytes = (mpz_sizeinbase(ctx->prod, 2) + 7) / 8 + 1;
    unsigned char *sendbuf = (unsigned char *)malloc(nbytes);
    if (!sendbuf) MPI_Abort(MPI_COMM_WORLD, 1);
    export_fixed(sendbuf, nbytes, local);

    unsigned char *recvbuf = NULL;
    if (rank == 0) {
        recvbuf = (unsigned char *)malloc(nbytes * (size_t)nranks);
        if (!recvbuf) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Gather(sendbuf, (int)nbytes, MPI_UNSIGNED_CHAR,
               recvbuf, (int)nbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Result of the step, packed big-endian into a |M|-sized buffer for bcast.
    size_t mbytes = (mpz_sizeinbase(modulo, 2) + 7) / 8 + 1;
    unsigned char *resbuf = (unsigned char *)malloc(mbytes);
    if (!resbuf) MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank == 0) {
        mpz_t total, part;
        mpz_init_set_ui(total, 0);
        mpz_init(part);
        for (int r = 0; r < nranks; r++) {
            mpz_import(part, nbytes, 1, 1, 0, 0, recvbuf + (size_t)r * nbytes);
            mpz_add(total, total, part);
        }
        mpz_mod(total, total, ctx->prod); // == c^alpha exactly (0 <= c^alpha < prod)
        mpz_mod(total, total, modulo);    // c^alpha mod M
        export_fixed(resbuf, mbytes, total);
        mpz_clears(total, part, NULL);
    }

    MPI_Bcast(resbuf, (int)mbytes, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    mpz_import(c_inout, mbytes, 1, 1, 0, 0, resbuf);

    if (rank == 0) free(recvbuf);
    free(sendbuf);
    free(resbuf);
    mpz_clear(local);
}

// Edit: 2026-07-12
