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

    /* 2. Random starting value x0 in [1, p-1] */
    do {
        mpz_urandomm(pz.x0, rng, pz.p);
    } while (mpz_cmp_ui(pz.x0, 0) == 0);

    /* 3. Compute x_T using the setter's shortcut:
     *      x_T = x0 ^ (e^{-T} mod (p-1))  mod p
     *    This is one powm instead of T powms.
     *    Derivation: x_k = x_{k-1}^(e^{-1}) mod p, so x_T = x0^(e^{-T}) mod p.
     *    e^{-T} mod phi = (e^T mod phi)^{-1} mod phi.
     */
    mpz_t phi, e_mpz, e_T, e_T_inv, x_T;
    mpz_init(phi);
    mpz_init_set_ui(e_mpz, e);
    mpz_init(e_T);
    mpz_init(e_T_inv);
    mpz_init(x_T);

    mpz_sub_ui(phi, pz.p, 1);                    /* phi = p - 1         */
    mpz_powm_ui(e_T, e_mpz, T, phi);             /* e_T = e^T mod phi   */

    if (mpz_invert(e_T_inv, e_T, phi) == 0) {
        fprintf(stderr, "Error: e^T not invertible mod p-1. "
                        "Choose a different prime or exponent.\n");
        mpz_clear(phi); mpz_clear(e_mpz); mpz_clear(e_T);
        mpz_clear(e_T_inv); mpz_clear(x_T);
        puzzle_free(&pz);
        gmp_randclear(rng);
        return 1;
    }

    printf("Computing x_T via shortcut (1 powm instead of %lu) ...\n", T);
    mpz_powm(x_T, pz.x0, e_T_inv, pz.p);        /* x_T = x0^(e_T_inv) mod p */

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

    mpz_clear(phi); mpz_clear(e_mpz); mpz_clear(e_T);
    mpz_clear(e_T_inv); mpz_clear(x_T);
    puzzle_free(&pz);
    gmp_randclear(rng);
    return 0;
}
