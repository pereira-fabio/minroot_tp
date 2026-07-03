#pragma once
#include <gmp.h>
#include <stddef.h>

/* SHA-256 of the big-endian byte representation of x. out must be 32 bytes. */
void sha256_mpz(const mpz_t x, unsigned char out[32]);

/* AES-256-CBC encrypt. key=32 bytes, iv=16 bytes.
   Returns heap-allocated ciphertext; caller frees. Sets *ct_len. */
unsigned char *aes256_encrypt(const unsigned char *plaintext, size_t pt_len,
                               const unsigned char key[32],
                               const unsigned char iv[16],
                               size_t *ct_len);

/* AES-256-CBC decrypt. Returns heap-allocated plaintext (null-terminated);
   caller frees. Sets *pt_len. Returns NULL on failure. */
unsigned char *aes256_decrypt(const unsigned char *ciphertext, size_t ct_len,
                               const unsigned char key[32],
                               const unsigned char iv[16],
                               size_t *pt_len);
