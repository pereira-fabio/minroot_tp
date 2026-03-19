#ifndef GINZA1_FACTOR_LIST_H
#define GINZA1_FACTOR_LIST_H

#include <gmp.h>

typedef struct
{
    mpz_t *items;
    int count;
    int capacity;
} factor_list_t;

void factor_list_init(factor_list_t *list);
void factor_list_append(factor_list_t *list, const mpz_t value);
void factor_list_append_ui(factor_list_t *list, unsigned long value);
void factor_list_append_ui_power(factor_list_t *list, unsigned long value, int exponent);
void factor_list_clear(factor_list_t *list);

#endif
