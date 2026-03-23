#ifndef GINZA1_SOLVER_H
#define GINZA1_SOLVER_H

#include <gmp.h>

#include "ginza1_factor_list.h"

typedef struct
{
    unsigned long smoothness_bound;
    unsigned long attempts_per_worker;
} solver_options_t;

void solve_k16(const mpz_t challenge, factor_list_t *factors);
void solve_k32(const mpz_t challenge, factor_list_t *factors);
void solve_k64(const mpz_t challenge, factor_list_t *factors, const solver_options_t *options);
void solve_unknown_k(int k, const mpz_t challenge, factor_list_t *factors);

#endif
