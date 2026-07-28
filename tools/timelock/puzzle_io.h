#pragma once
#include <gmp.h>
#include <stddef.h>

typedef struct {
    mpz_t        p;               /* prime field modulus */
    mpz_t        x0;              /* MinRoot start state, first component  */
    mpz_t        y0;              /* MinRoot start state, second component */
    unsigned long e;              /* root exponent (typically 5) */
    unsigned long T;              /* number of sequential root iterations */
    unsigned char iv[16];         /* AES-256-CBC IV */
    unsigned char *ciphertext;    /* encrypted message (heap-allocated) */
    size_t        ciphertext_len;
} puzzle_t;

void puzzle_init(puzzle_t *pz);
void puzzle_free(puzzle_t *pz);

/* Returns 0 on success, -1 on error. */
int puzzle_write(const puzzle_t *pz, const char *path);
int puzzle_read(puzzle_t *pz, const char *path);

// Edit: 2026-07-28
