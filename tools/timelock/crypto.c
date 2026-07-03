#include "crypto.h"
#include <openssl/evp.h>
#include <stdlib.h>
#include <string.h>

void sha256_mpz(const mpz_t x, unsigned char out[32]) {
    /* Export x as big-endian bytes */
    size_t nbytes = (mpz_sizeinbase(x, 2) + 7) / 8;
    if (nbytes == 0) nbytes = 1;

    unsigned char *buf = calloc(nbytes, 1);
    size_t count = 0;
    mpz_export(buf, &count, 1, 1, 1, 0, x);

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    EVP_DigestUpdate(ctx, buf, count == 0 ? 1 : count);
    unsigned int outlen = 32;
    EVP_DigestFinal_ex(ctx, out, &outlen);
    EVP_MD_CTX_free(ctx);
    free(buf);
}

unsigned char *aes256_encrypt(const unsigned char *pt, size_t pt_len,
                               const unsigned char key[32],
                               const unsigned char iv[16],
                               size_t *ct_len) {
    /* Output buffer: plaintext + one extra AES block for PKCS7 padding */
    unsigned char *out = malloc(pt_len + 32);
    int len1 = 0, len2 = 0;

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv);
    EVP_EncryptUpdate(ctx, out, &len1, pt, (int)pt_len);
    EVP_EncryptFinal_ex(ctx, out + len1, &len2);
    EVP_CIPHER_CTX_free(ctx);

    *ct_len = (size_t)(len1 + len2);
    return out;
}

unsigned char *aes256_decrypt(const unsigned char *ct, size_t ct_len,
                               const unsigned char key[32],
                               const unsigned char iv[16],
                               size_t *pt_len) {
    unsigned char *out = malloc(ct_len + 16);
    int len1 = 0, len2 = 0;

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv);

    if (!EVP_DecryptUpdate(ctx, out, &len1, ct, (int)ct_len)) {
        EVP_CIPHER_CTX_free(ctx);
        free(out);
        return NULL;
    }
    if (!EVP_DecryptFinal_ex(ctx, out + len1, &len2)) {
        EVP_CIPHER_CTX_free(ctx);
        free(out);
        return NULL;
    }
    EVP_CIPHER_CTX_free(ctx);

    *pt_len = (size_t)(len1 + len2);
    out[*pt_len] = '\0';   /* null-terminate for convenience */
    return out;
}
