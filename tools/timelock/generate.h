#pragma once

/* Generate a MinRoot time-lock puzzle and write it to outpath.
 * bits    : bit-length of the prime modulus p
 * e       : root exponent (must be coprime to p-1; typically 3 or 5)
 * T       : number of sequential e-th root iterations
 * message : plaintext to encrypt
 * outpath : destination .mrt file
 * Returns 0 on success. */
int cmd_generate(int bits, unsigned long e, unsigned long T,
                 const char *message, const char *outpath);
