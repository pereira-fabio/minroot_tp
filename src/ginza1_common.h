#ifndef GINZA1_COMMON_H
#define GINZA1_COMMON_H

#include <gmp.h>

#include "ginza1_factor_list.h"

void read_mpz_from_file(mpz_t num, const char *filename);
int extract_k_from_filename(const char *path);
bool parse_unsigned_long_arg(const char *text, unsigned long *value);
double wall_time_seconds();

bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo);
bool compute_cube_root_direct_powm(mpz_t result, const mpz_t challenge, const mpz_t modulo);
void compute_cube_root_from_factors(mpz_t result, const factor_list_t *factors, const mpz_t modulo);
void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge, const mpz_t modulo);
void print_factorization_summary(const mpz_t value, const factor_list_t *factors);

#endif
