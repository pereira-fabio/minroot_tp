#include "ginza1_factor_list.h"

#include <stdio.h>
#include <stdlib.h>

void factor_list_init(factor_list_t *list)
{
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

void factor_list_append(factor_list_t *list, const mpz_t value)
{
    if (list->count == list->capacity)
    {
        int new_capacity = (list->capacity == 0) ? 16 : list->capacity * 2;
        mpz_t *new_items = (mpz_t *)realloc(list->items, (size_t)new_capacity * sizeof(mpz_t));
        if (!new_items)
        {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }

    mpz_init_set(list->items[list->count], value);
    list->count++;
}

void factor_list_append_ui(factor_list_t *list, unsigned long value)
{
    mpz_t factor;
    mpz_init_set_ui(factor, value);
    factor_list_append(list, factor);
    mpz_clear(factor);
}

void factor_list_append_ui_power(factor_list_t *list, unsigned long value, int exponent)
{
    for (int i = 0; i < exponent; i++)
    {
        factor_list_append_ui(list, value);
    }
}

void factor_list_clear(factor_list_t *list)
{
    for (int i = 0; i < list->count; i++)
    {
        mpz_clear(list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}
