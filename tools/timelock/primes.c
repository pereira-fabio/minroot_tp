#include "primes.h"
#include <stdlib.h>
#include <string.h>

void prime_table_init(prime_table_t *pt, unsigned long B) {
    unsigned char *sieve = calloc(B + 1, 1);
    sieve[0] = 1;
    if (B >= 1) sieve[1] = 1;
    for (unsigned long i = 2; i * i <= B; i++) {
        if (!sieve[i]) {
            for (unsigned long j = i * i; j <= B; j += i)
                sieve[j] = 1;
        }
    }

    size_t count = 0;
    for (unsigned long i = 2; i <= B; i++)
        if (!sieve[i]) count++;

    pt->data  = malloc(count * sizeof(unsigned long));
    pt->count = 0;
    pt->bound = B;

    for (unsigned long i = 2; i <= B; i++)
        if (!sieve[i]) pt->data[pt->count++] = i;

    free(sieve);
}

void prime_table_free(prime_table_t *pt) {
    free(pt->data);
    pt->data  = NULL;
    pt->count = 0;
    pt->bound = 0;
}
