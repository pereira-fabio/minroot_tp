#pragma once
#include <stddef.h>

typedef struct {
    unsigned long *data;
    size_t         count;
    unsigned long  bound;
} prime_table_t;

/* Sieve of Eratosthenes up to B. B must be <= 1<<24 (16M) to keep sieve in RAM. */
void prime_table_init(prime_table_t *pt, unsigned long B);
void prime_table_free(prime_table_t *pt);
