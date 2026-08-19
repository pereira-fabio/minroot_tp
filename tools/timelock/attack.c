#include "attack.h"
#include "puzzle_io.h"
#include "crypto.h"
#include "primes.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Trial-divide y (modified in place) by primes in the table.
 * exponents[j] receives the multiplicity of primes->data[j] in y.
 * Returns 1 if y is fully B-smooth (y == 1 after division), 0 otherwise. */
static int trial_divide(mpz_t y,
                        const prime_table_t *primes,
                        unsigned long *exponents)
{
    mpz_t q, rem;
    mpz_init(q);
    mpz_init(rem);

    for (size_t j = 0; j < primes->count; j++)
        exponents[j] = 0;

    for (size_t j = 0; j < primes->count; j++)
    {
        unsigned long p_j = primes->data[j];

        /* Early exit: if remaining y < p_j^2 and y > 1 it has no small factors */
        if (mpz_cmp_ui(y, p_j * p_j) < 0)
            break;

        while (1)
        {
            mpz_tdiv_qr_ui(q, rem, y, p_j);
            if (mpz_cmp_ui(rem, 0) != 0)
                break;
            mpz_set(y, q);
            exponents[j]++;
        }

        if (mpz_cmp_ui(y, 1) == 0)
            break;
    }

    // After the loop the residual y is 1, or (via the y < p_j^2 early-exit) a
    // single prime with no factor below the current p_j. That residual prime is
    // still B-smooth if it is <= B (i.e. present in the prime table) — the old
    // code wrongly rejected these, which collapsed the hit rate from ~rho(u) to
    // ~rho(2u) and made the attack ~1000x too slow. Record its exponent so the
    // reconstruction includes it.
    int smooth;
    if (mpz_cmp_ui(y, 1) == 0)
    {
        smooth = 1;
    }
    else
    {
        unsigned long maxp = primes->data[primes->count - 1]; // largest prime <= B
        if (mpz_cmp_ui(y, maxp) <= 0)
        {
            unsigned long yv = mpz_get_ui(y);  // fits: y <= maxp <= B
            size_t lo = 0, hi = primes->count; // binary search for yv
            while (lo < hi)
            {
                size_t mid = lo + (hi - lo) / 2;
                if (primes->data[mid] < yv)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo < primes->count && primes->data[lo] == yv)
            {
                exponents[lo] += 1;
                smooth = 1;
            }
            else
            {
                smooth = 0; // y <= B but not a tabled prime (shouldn't occur)
            }
        }
        else
        {
            smooth = 0; // residual cofactor > B -> not B-smooth
        }
    }
    mpz_clear(q);
    mpz_clear(rem);
    return smooth;
}

int cmd_attack(const char *puzzle_path, unsigned long B)
{
    puzzle_t pz;
    puzzle_init(&pz);
    if (puzzle_read(&pz, puzzle_path) != 0)
    {
        fprintf(stderr, "Error: could not read puzzle from %s\n", puzzle_path);
        return 1;
    }

    size_t prime_bits = mpz_sizeinbase(pz.p, 2);
    printf("Attack solver (single-threaded smoothness)\n");
    printf("  T = %lu iterations, e = %lu, |p| = %zu bits, B = %lu\n\n",
           pz.T, pz.e, prime_bits, B);

    /* Warn if B is likely too small to find smooth blindings in reasonable time.
     * Rule of thumb: u = prime_bits / log2(B) should be <= 6 for practical use. */
    {
        double log2B = 0.0;
        unsigned long tmp = B;
        while (tmp >>= 1)
            log2B++;
        double u = (double)prime_bits / log2B;
        if (u > 6.0)
        {
            printf("  WARNING: u = |p|/log2(B) = %.1f > 6. Hit rate will be very low.\n"
                   "  Consider using a smaller prime (-b 64) or larger B.\n\n",
                   u);
        }
    }

    /* Build prime table */
    printf("  Building prime sieve up to B=%lu ...\n", B);
    prime_table_t primes;
    prime_table_init(&primes, B);
    printf("  %zu primes in table.\n", primes.count);

    /* Precompute e_inv mod (p-1) */
    mpz_t phi, e_mpz, e_inv;
    mpz_init(phi);
    mpz_init_set_ui(e_mpz, pz.e);
    mpz_init(e_inv);
    mpz_sub_ui(phi, pz.p, 1);
    if (mpz_invert(e_inv, e_mpz, phi) == 0)
    {
        fprintf(stderr, "Error: e not invertible mod p-1\n");
        return 1;
    }

    /* Precompute cbrt table: cbrt_table[j] = primes[j]^(e_inv) mod p.
     * This is the e-th root of each small prime in the field F_p. */
    printf("  Precomputing cbrt table (%zu entries) ...\n", primes.count);
    mpz_t *cbrt_table = malloc(primes.count * sizeof(mpz_t));
    mpz_t p_j_mpz;
    mpz_init(p_j_mpz);
    for (size_t j = 0; j < primes.count; j++)
    {
        mpz_init(cbrt_table[j]);
        mpz_set_ui(p_j_mpz, primes.data[j]);
        mpz_powm(cbrt_table[j], p_j_mpz, e_inv, pz.p);
    }
    mpz_clear(p_j_mpz);
    printf("  cbrt table ready.\n\n");

    /* RNG for blinding scalars */
    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, (unsigned long)time(NULL) ^ (unsigned long)clock());

    unsigned long *exponents = malloc(primes.count * sizeof(unsigned long));

    mpz_t x, r, r_e, r_inv, y, y_work, result, check;
    mpz_t st_y, s, x_old; /* MinRoot second state, round challenge */
    mpz_init_set(x, pz.x0);
    mpz_init_set(st_y, pz.y0);
    mpz_init(s);
    mpz_init(x_old);
    mpz_init(r);
    mpz_init(r_e);
    mpz_init(r_inv);
    mpz_init(y);
    mpz_init(y_work);
    mpz_init(result);
    mpz_init(check);

    struct timespec t0, tnow;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    unsigned long total_attempts = 0;
    unsigned long report = pz.T / 10;
    if (report < 1)
        report = 1;

    for (unsigned long step = 0; step < pz.T; step++)
    {
        unsigned long attempts_this_step = 0;

        /* MinRoot round: this step must invert s = x_i + y_i, not x_i. The
         * feed-forward is additive and free; the e-th root is the whole cost,
         * and it is the only part the smoothness attack touches. */
        mpz_add(s, x, st_y);
        mpz_mod(s, s, pz.p);

        if (mpz_cmp_ui(s, 0) == 0)
        {
            /* s = 0 has the unique e-th root 0. The blinding loop would spin
             * forever on it (0 is never B-smooth). Probability ~1/p. */
            mpz_set_ui(result, 0);
        }
        else
            while (1)
            {
                attempts_this_step++;
                total_attempts++;

                /* Random r in [2, p-2] */
                do
                {
                    mpz_urandomm(r, rng, pz.p);
                } while (mpz_cmp_ui(r, 1) <= 0);

                /* Blinded challenge: y = s * r^e mod p */
                mpz_powm(r_e, r, e_mpz, pz.p);
                mpz_mul(y, s, r_e);
                mpz_mod(y, y, pz.p);

                /* Trial division on a working copy */
                mpz_set(y_work, y);
                int smooth = trial_divide(y_work, &primes, exponents);
                if (!smooth)
                    continue;

                /* Reconstruct y^(1/e) mod p = prod(cbrt_table[j]^exp[j]) mod p */
                mpz_set_ui(result, 1);
                for (size_t j = 0; j < primes.count; j++)
                {
                    if (exponents[j] == 0)
                        continue;
                    mpz_powm_ui(check, cbrt_table[j], exponents[j], pz.p);
                    mpz_mul(result, result, check);
                    mpz_mod(result, result, pz.p);
                }

                /* Unblind: x_{step+1} = result * r^{-1} mod p */
                mpz_invert(r_inv, r, pz.p);
                mpz_mul(result, result, r_inv);
                mpz_mod(result, result, pz.p);

                /* Correctness gate: result^e mod p must equal s = x_i + y_i */
                mpz_powm(check, result, e_mpz, pz.p);
                if (mpz_cmp(check, s) != 0)
                {
                    /* Should never happen if the cbrt table is correct. */
                    fprintf(stderr, "  [BUG] step %lu: verify failed — retrying\n", step);
                    continue;
                }

                break;
            }

        /* Advance the MinRoot state: y_{i+1} = x_i + i, then x_{i+1} = root. */
        mpz_set(x_old, x);
        mpz_set(x, result);
        mpz_add_ui(st_y, x_old, step);
        mpz_mod(st_y, st_y, pz.p);

        if ((step + 1) % report == 0 || step + 1 == pz.T)
        {
            clock_gettime(CLOCK_MONOTONIC, &tnow);
            double elapsed = (tnow.tv_sec - t0.tv_sec) + (tnow.tv_nsec - t0.tv_nsec) * 1e-9;
            printf("  step %5lu/%lu  attempts=%lu  total_attempts=%lu  elapsed=%.2fs\n",
                   step + 1, pz.T, attempts_this_step, total_attempts, elapsed);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tnow);
    double total = (tnow.tv_sec - t0.tv_sec) + (tnow.tv_nsec - t0.tv_nsec) * 1e-9;

    /* Derive key and decrypt */
    unsigned char key[32];
    sha256_mpz(x, key);

    size_t pt_len;
    unsigned char *plaintext = aes256_decrypt(
        pz.ciphertext, pz.ciphertext_len, key, pz.iv, &pt_len);

    if (!plaintext)
    {
        fprintf(stderr, "\nDecryption failed — wrong final x_T. "
                        "Attack produced incorrect result.\n");
        return 1;
    }

    printf("\n=== Attack solved in %.3f s ===\n", total);
    printf("Total blinding attempts : %lu  (avg %.1f per step)\n",
           total_attempts, (double)total_attempts / (double)pz.T);
    printf("Message: %s\n", (char *)plaintext);

    free(plaintext);
    free(exponents);
    for (size_t j = 0; j < primes.count; j++)
        mpz_clear(cbrt_table[j]);
    free(cbrt_table);
    mpz_clear(phi);
    mpz_clear(e_mpz);
    mpz_clear(e_inv);
    mpz_clear(x);
    mpz_clear(st_y);
    mpz_clear(s);
    mpz_clear(x_old);
    mpz_clear(r);
    mpz_clear(r_e);
    mpz_clear(r_inv);
    mpz_clear(y);
    mpz_clear(y_work);
    mpz_clear(result);
    mpz_clear(check);
    prime_table_free(&primes);
    gmp_randclear(rng);
    puzzle_free(&pz);
    return 0;
}