#include "puzzle_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void puzzle_init(puzzle_t *pz) {
    mpz_init(pz->p);
    mpz_init(pz->x0);
    pz->e = 5;
    pz->T = 0;
    memset(pz->iv, 0, 16);
    pz->ciphertext     = NULL;
    pz->ciphertext_len = 0;
}

void puzzle_free(puzzle_t *pz) {
    mpz_clear(pz->p);
    mpz_clear(pz->x0);
    free(pz->ciphertext);
    pz->ciphertext     = NULL;
    pz->ciphertext_len = 0;
}

/* ---- hex helpers ---- */

static void bytes_to_hex(const unsigned char *b, size_t n, char *out) {
    for (size_t i = 0; i < n; i++)
        sprintf(out + 2 * i, "%02x", b[i]);
    out[2 * n] = '\0';
}

static size_t hex_to_bytes(const char *hex, unsigned char *out) {
    size_t n = strlen(hex) / 2;
    for (size_t i = 0; i < n; i++) {
        unsigned int v = 0;
        sscanf(hex + 2 * i, "%02x", &v);
        out[i] = (unsigned char)v;
    }
    return n;
}

/* ---- I/O ---- */

int puzzle_write(const puzzle_t *pz, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;

    char *p_hex  = mpz_get_str(NULL, 16, pz->p);
    char *x0_hex = mpz_get_str(NULL, 16, pz->x0);

    fprintf(f, "MINROOT-TIMELOCK-V1\n");
    fprintf(f, "p: %s\n",   p_hex);
    fprintf(f, "e: %lu\n",  pz->e);
    fprintf(f, "x0: %s\n",  x0_hex);
    fprintf(f, "T: %lu\n",  pz->T);

    char iv_hex[33];
    bytes_to_hex(pz->iv, 16, iv_hex);
    fprintf(f, "iv: %s\n", iv_hex);

    char *ct_hex = malloc(pz->ciphertext_len * 2 + 1);
    bytes_to_hex(pz->ciphertext, pz->ciphertext_len, ct_hex);
    fprintf(f, "ciphertext: %s\n", ct_hex);

    free(p_hex);
    free(x0_hex);
    free(ct_hex);
    fclose(f);
    return 0;
}

int puzzle_read(puzzle_t *pz, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[131072];   /* large enough for a 24576-bit hex modulus */

    /* header */
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    if (strncmp(line, "MINROOT-TIMELOCK-V1", 19) != 0) { fclose(f); return -1; }

    char val[131072];
    while (fgets(line, sizeof(line), f)) {
        /* strip trailing newline */
        line[strcspn(line, "\r\n")] = '\0';

        if (sscanf(line, "p: %s",  val) == 1) { mpz_set_str(pz->p,  val, 16); }
        else if (sscanf(line, "e: %lu",  &pz->e) == 1) { /* done */ }
        else if (sscanf(line, "x0: %s", val) == 1) { mpz_set_str(pz->x0, val, 16); }
        else if (sscanf(line, "T: %lu",  &pz->T) == 1) { /* done */ }
        else if (sscanf(line, "iv: %s",  val) == 1) {
            hex_to_bytes(val, pz->iv);
        } else if (strncmp(line, "ciphertext: ", 12) == 0) {
            const char *hex = line + 12;
            size_t hex_len       = strlen(hex);
            pz->ciphertext_len   = hex_len / 2;
            pz->ciphertext       = malloc(pz->ciphertext_len);
            hex_to_bytes(hex, pz->ciphertext);
        }
    }
    fclose(f);
    return 0;
}
