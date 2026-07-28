#include "solve.h"
#include "puzzle_io.h"
#include "crypto.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int cmd_solve(const char *puzzle_path) {
    puzzle_t pz;
    puzzle_init(&pz);
    if (puzzle_read(&pz, puzzle_path) != 0) {
        fprintf(stderr, "Error: could not read puzzle from %s\n", puzzle_path);
        return 1;
    }

    printf("Sequential solver\n");
    printf("  T = %lu iterations, e = %lu\n", pz.T, pz.e);
    printf("  |p| = %zu bits\n\n", mpz_sizeinbase(pz.p, 2));

    mpz_t phi, e_mpz, e_inv, x, y, s, x_old;
    mpz_init(phi);
    mpz_init_set_ui(e_mpz, pz.e);
    mpz_init(e_inv);
    mpz_init_set(x, pz.x0);
    mpz_init_set(y, pz.y0);
    mpz_init(s);
    mpz_init(x_old);

    mpz_sub_ui(phi, pz.p, 1);
    if (mpz_invert(e_inv, e_mpz, phi) == 0) {
        fprintf(stderr, "Error: e not invertible mod p-1\n");
        return 1;
    }

    struct timespec t0, tnow;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Report every 5% of iterations (at least every 1 step). */
    unsigned long report = pz.T / 20;
    if (report < 1) report = 1;

    /* MinRoot forward round -- the honest, inherently sequential direction:
     *   x_{i+1} = (x_i + y_i)^{1/e} mod p     <- the ONLY expensive operation
     *   y_{i+1} = x_i + i           mod p
     * The two additions are free; the root is what the delay rests on. There
     * is no shortcut here, which is the whole point of the construction. */
    for (unsigned long i = 0; i < pz.T; i++) {
        mpz_add(s, x, y);
        mpz_mod(s, s, pz.p);
        mpz_set(x_old, x);
        mpz_powm(x, s, e_inv, pz.p);   /* x_{i+1} = (x_i + y_i)^{1/e} */
        mpz_add_ui(y, x_old, i);       /* y_{i+1} = x_i + i           */
        mpz_mod(y, y, pz.p);

        if ((i + 1) % report == 0 || i + 1 == pz.T) {
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            double elapsed = (tnow.tv_sec - t0.tv_sec)
                           + (tnow.tv_nsec - t0.tv_nsec) * 1e-9;
            double frac    = (double)(i + 1) / (double)pz.T;
            double eta     = (frac > 0.0) ? (elapsed / frac - elapsed) : 0.0;
            printf("  [%5.1f%%]  step %lu/%lu  elapsed %.2fs  ETA %.2fs\n",
                   frac * 100.0, i + 1, pz.T, elapsed, eta);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tnow);
    double total = (tnow.tv_sec - t0.tv_sec)
                 + (tnow.tv_nsec - t0.tv_nsec) * 1e-9;

    /* Derive key and decrypt */
    unsigned char key[32];
    sha256_mpz(x, key);

    size_t        pt_len;
    unsigned char *plaintext = aes256_decrypt(
        pz.ciphertext, pz.ciphertext_len, key, pz.iv, &pt_len);

    if (!plaintext) {
        fprintf(stderr, "\nDecryption failed — wrong key. "
                        "The solve is incorrect.\n");
        mpz_clear(phi); mpz_clear(e_mpz); mpz_clear(e_inv);
        mpz_clear(x); mpz_clear(y); mpz_clear(s); mpz_clear(x_old);
        puzzle_free(&pz);
        return 1;
    }

    printf("\n=== Solved in %.3f s ===\n", total);
    printf("Message: %s\n", (char *)plaintext);

    free(plaintext);
    mpz_clear(phi); mpz_clear(e_mpz); mpz_clear(e_inv);
    mpz_clear(x); mpz_clear(y); mpz_clear(s); mpz_clear(x_old);
    puzzle_free(&pz);
    return 0;
}

// Edit: 2026-07-28
