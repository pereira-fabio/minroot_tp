#include "generate.h"
#include "puzzle_io.h"
#include "crypto.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Generate a prime of exactly `bits` bits with gcd(e, p-1) = 1. */
static void gen_prime(mpz_t p, unsigned long e, int bits, gmp_randstate_t rng) {
    mpz_t g, phi;
    mpz_init(g);
    mpz_init(phi);

    while (1) {
        mpz_urandomb(p, rng, (mp_bitcnt_t)bits);
        mpz_setbit(p, (mp_bitcnt_t)(bits - 1)); /* force correct bit length */
        mpz_nextprime(p, p);

        if ((int)mpz_sizeinbase(p, 2) != bits)
            continue;

        /* Require gcd(e, p-1) = 1 so e-th roots are unique in F_p. */
        mpz_sub_ui(phi, p, 1);
        mpz_set_ui(g, e);
        mpz_gcd(g, g, phi);
        if (mpz_cmp_ui(g, 1) == 0)
            break;
    }
    mpz_clear(g);
    mpz_clear(phi);
}

int cmd_generate(int bits, unsigned long e, unsigned long T,
                 const char *message, const char *outpath) {
    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, (unsigned long)time(NULL) ^ (unsigned long)clock());

    puzzle_t pz;
    puzzle_init(&pz);
    pz.e = e;
    pz.T = T;

    /* 1. Prime p */
    printf("Generating %d-bit prime p with gcd(%lu, p-1) = 1 ...\n", bits, e);
    gen_prime(pz.p, e, bits, rng);
    gmp_printf("  p  = %Zx\n", pz.p);

    /* 2. Pick the FINAL MinRoot state (x_T, y_T) at random.
     *
     *    MinRoot is trapdoor-free by design. There is no setup secret that
     *    lets the setter skip ahead (unlike an RSA time-lock), and the rounds
     *    do NOT collapse into a single exponentiation the way a plain iterated
     *    e-th root does -- the +y_i and +i feed-forward destroys that. So the
     *    setter genuinely cannot walk the chain forward cheaply.
     *
     *    It does not have to. The MinRoot round inverts cheaply: walking
     *    BACKWARDS costs one e-th POWER per step instead of one e-th ROOT.
     *    So choose the end state and walk back to the start.
     */
    mpz_t x_T, y_T, xc, yc, x_prev, y_prev;
    mpz_init(x_T); mpz_init(y_T);
    mpz_init(xc);  mpz_init(yc);
    mpz_init(x_prev); mpz_init(y_prev);

    do {
        mpz_urandomm(x_T, rng, pz.p);
    } while (mpz_cmp_ui(x_T, 0) == 0);
    mpz_urandomm(y_T, rng, pz.p);

    /* 3. Walk backwards T rounds to recover (x_0, y_0).
     *
     *    Forward round i:  x_{i+1} = (x_i + y_i)^{1/e} mod p
     *                      y_{i+1} = x_i + i           mod p
     *
     *    Inverting round i:  x_i = y_{i+1} - i
     *                        y_i = x_{i+1}^e - x_i
     *
     *    (Check: x_i + y_i = x_{i+1}^e, so (x_i+y_i)^{1/e} = x_{i+1}; and
     *     x_i + i = y_{i+1}. Both forward relations hold.)
     */
    printf("Walking %lu MinRoot rounds backwards (e-th powers, cheap) ...\n", T);
    mpz_set(xc, x_T);
    mpz_set(yc, y_T);
    for (unsigned long i = T; i-- > 0; ) {
        mpz_sub_ui(x_prev, yc, i);              /* x_i = y_{i+1} - i        */
        mpz_mod(x_prev, x_prev, pz.p);
        mpz_powm_ui(y_prev, xc, e, pz.p);       /* y_i = x_{i+1}^e - x_i    */
        mpz_sub(y_prev, y_prev, x_prev);
        mpz_mod(y_prev, y_prev, pz.p);
        mpz_set(xc, x_prev);
        mpz_set(yc, y_prev);
    }
    mpz_set(pz.x0, xc);
    mpz_set(pz.y0, yc);

    /* 4. AES-256 key = SHA-256(x_T) */
    unsigned char key[32];
    sha256_mpz(x_T, key);

    /* 5. Random IV from /dev/urandom */
    FILE *ur = fopen("/dev/urandom", "rb");
    if (!ur || fread(pz.iv, 1, 16, ur) != 16) {
        fprintf(stderr, "Error: could not read /dev/urandom\n");
        if (ur) fclose(ur);
        return 1;
    }
    fclose(ur);

    /* 6. Encrypt message */
    size_t msg_len = strlen(message);
    pz.ciphertext  = aes256_encrypt(
        (const unsigned char *)message, msg_len, key, pz.iv, &pz.ciphertext_len);

    /* 7. Write puzzle file */
    if (puzzle_write(&pz, outpath) != 0) {
        fprintf(stderr, "Error: could not write to %s\n", outpath);
        return 1;
    }

    printf("\nPuzzle written to: %s\n", outpath);
    printf("  Prime bits   : %d\n",   bits);
    printf("  Exponent e   : %lu\n",  e);
    printf("  Iterations T : %lu\n",  T);
    printf("  Message len  : %zu bytes\n", msg_len);
    printf("\nSolve sequentially : minroot-timelock solve %s\n",  outpath);
    printf("Solve via attack   : minroot-timelock attack %s\n", outpath);

    mpz_clear(x_T); mpz_clear(y_T); mpz_clear(xc); mpz_clear(yc);
    mpz_clear(x_prev); mpz_clear(y_prev);
    puzzle_free(&pz);
    gmp_randclear(rng);
    return 0;
}

// Edit: 2026-07-28
