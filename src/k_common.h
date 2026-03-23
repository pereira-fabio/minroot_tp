#include <gmp.h>

typedef struct
{
    mpz_t *items;
    int count;
    int capacity;
} factor_list_t;

void factor_list_init(factor_list_t *list);
void factor_list_clear(factor_list_t *list);
void factor_list_append(factor_list_t *list, const mpz_t value);

unsigned long *generate_primes_up_to(unsigned long limit, int *count);
void read_mpz_from_file(mpz_t num, const char *filename);
bool compute_cube_root_mod_prime(mpz_t result, const mpz_t value, const mpz_t modulo);
void compute_cube_root_from_factors(mpz_t result, const factor_list_t *factors, const mpz_t modulo);
void verify_cube_root(const mpz_t candidate_cuberoot, const mpz_t challenge_mod, const mpz_t modulo);

int count_exponent(mpz_t n, unsigned long prime);