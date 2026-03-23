#ifndef GINZA1_FACTORIZATION_H
#define GINZA1_FACTORIZATION_H

#include <gmp.h>

#include "ginza1_factor_list.h"

unsigned long *generate_primes_up_to(unsigned long limit, int *count);
bool is_b_smooth(const mpz_t n, const unsigned long *primes, int prime_count, int *exponents);

void factor_challenge_trial_division(const mpz_t challenge, const unsigned long *primes, int prime_count, factor_list_t *factors);
void factor_challenge_pollard_with_progress(const mpz_t challenge, factor_list_t *factors);

#endif
