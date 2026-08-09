/* ---------------------------------------------------------------------------
 * gen_prime_table.c — build a streamed prime table for an ARBITRARY bound 2^k.
 *
 * WHY THIS EXISTS
 * ---------------
 * The solvers ship two hard-coded tables: k32 over (2^16, 2^32] and k35 over
 * (2^16, 2^35]. Those are the two endpoints the thesis measures, and they land
 * on opposite sides of the GPU cache boundary -- k16 fits in L1 and wins, k32
 * misses everything and loses. The interesting question is where the boundary
 * actually sits, and answering it needs tables at the intermediate k.
 *
 * analysis/cache_residency.py predicts the crossover from prime counts alone
 * (largest resident k is 30 on an L40 at 4 GPUs, 26 on a V100). This tool
 * produces the tables that turn that prediction into a measurement.
 *
 * FORMAT
 * ------
 * Byte-identical to kf_generate_and_save_prime_table, so the existing loaders
 * read these files with no change:
 *
 *     u64  magic  = 0x4B33325052494D53  ("K32PRIMS")
 *     u64  count  = number of primes that follow
 *     u32  primes[count]                 ascending, all in (2^16, 2^k]
 *
 * The magic is kept as "K32PRIMS" deliberately: it identifies the FORMAT, not
 * the bound, and changing it would break ksc_load_prime_slice for no gain.
 *
 * RANGE
 * -----
 * Lower bound is fixed at 65537. Primes at or below 65536 are served by the
 * dense k16 cbrt table (Gap 8) and are never streamed, so including them would
 * both waste space and double-count during factoring.
 *
 * k is limited to [17, 32]. Below 17 the range is empty or nearly so and the
 * k16 dense table is the right structure anyway; above 32 a prime no longer
 * fits in u32 and the caller wants the u64 path (kf_generate_and_save_prime_
 * table_u64, as used by k35).
 *
 * BUILD
 *     gcc -O2 -o gen_prime_table tools/gen_prime_table.c
 * USAGE
 *     ./gen_prime_table 20                  -> data/k20_primes.bin
 *     ./gen_prime_table 20 /path/out.bin
 *     ./gen_prime_table --all               -> k17..k31 (skips k32, ships already)
 * ------------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <time.h>

#define SEG_SIZE     (1 << 19)                      /* 512 KiB, matches KF_SEG_SIZE  */
#define WRITE_BATCH  65536                          /* matches KF_WRITE_BATCH        */
#define MAGIC        0x4B33325052494D53ULL          /* "K32PRIMS", format id         */
#define LOWER        65537ULL                       /* primes below use the k16 table */

/* Primes up to 65536, for sieving the segments. */
static unsigned *small_primes(int *count)
{
    const int limit = 65536;
    char *s = calloc((size_t)limit + 1, 1);
    memset(s, 1, (size_t)limit + 1);
    s[0] = s[1] = 0;
    for (int i = 2; (long long)i * i <= limit; i++)
        if (s[i])
            for (int j = i * i; j <= limit; j += i)
                s[j] = 0;
    *count = 0;
    for (int i = 2; i <= limit; i++) if (s[i]) (*count)++;
    unsigned *p = malloc((size_t)(*count) * sizeof(unsigned));
    int n = 0;
    for (int i = 2; i <= limit; i++) if (s[i]) p[n++] = (unsigned)i;
    free(s);
    return p;
}

static int generate(int k, const char *path)
{
    if (k < 17 || k > 32) {
        fprintf(stderr, "k must be in [17,32] (got %d); "
                        "use the k16 dense table below, the u64 path above\n", k);
        return 1;
    }
    /* 2^32 - 1 is the largest value a u32 prime can take; for k = 32 the bound
       is UINT32_MAX rather than 2^32 so the arithmetic stays in range. */
    uint64_t upper = (k == 32) ? (uint64_t)UINT32_MAX : ((uint64_t)1 << k);

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot create %s\n", path); return 1; }

    uint64_t magic = MAGIC, count = 0;
    fwrite(&magic, 8, 1, f);
    long count_pos = ftell(f);
    fwrite(&count, 8, 1, f);          /* placeholder, rewritten at the end */

    int spc;
    unsigned *sp = small_primes(&spc);
    char *seg = malloc(SEG_SIZE);
    uint32_t *batch = malloc(WRITE_BATCH * sizeof(uint32_t));
    int bi = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (uint64_t lo = LOWER; lo <= upper; lo += SEG_SIZE) {
        uint64_t hi = lo + SEG_SIZE - 1;
        if (hi > upper) hi = upper;
        size_t len = (size_t)(hi - lo + 1);

        memset(seg, 1, len);
        for (int i = 0; i < spc; i++) {
            uint64_t p = sp[i], start = ((lo + p - 1) / p) * p;
            if (start == p) start += p;
            for (uint64_t j = start; j <= hi; j += p) seg[j - lo] = 0;
        }
        for (uint64_t c = lo; c <= hi; c++) {
            if (!seg[c - lo]) continue;
            batch[bi++] = (uint32_t)c;
            count++;
            if (bi == WRITE_BATCH) { fwrite(batch, 4, WRITE_BATCH, f); bi = 0; }
        }
    }
    if (bi > 0) fwrite(batch, 4, (size_t)bi, f);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    fseek(f, count_pos, SEEK_SET);
    fwrite(&count, 8, 1, f);
    fclose(f);
    free(seg); free(batch); free(sp);

    double mb = (double)(16 + count * 4) / 1e6;
    printf("k%-2d  %12" PRIu64 " primes in (2^16, %s]  ->  %8.3f MB  [%.0f ms]  %s\n",
           k, count, (k == 32 ? "2^32-1" : "2^k"), mb, ms, path);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <k>            [outfile]   generate one table (k in [17,32])\n"
            "       %s --all          [outdir]    generate k17..k31\n", argv[0], argv[0]);
        return 1;
    }
    mkdir("data", 0755);

    if (strcmp(argv[1], "--all") == 0) {
        const char *dir = (argc > 2) ? argv[2] : "data";
        char path[512];
        for (int k = 17; k <= 31; k++) {
            snprintf(path, sizeof(path), "%s/k%d_primes.bin", dir, k);
            if (generate(k, path)) return 1;
        }
        printf("\nk32 skipped: data/k32_primes.bin already ships with the solvers.\n");
        return 0;
    }

    int k = atoi(argv[1]);
    char path[512];
    if (argc > 2) snprintf(path, sizeof(path), "%s", argv[2]);
    else          snprintf(path, sizeof(path), "data/k%d_primes.bin", k);
    return generate(k, path);
}
