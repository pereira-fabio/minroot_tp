// =============================================================================
// delta_table.cpp — delta/gap-encoded prime table (CPU side)
//
// EXPERIMENT — not part of the attack. See experiments/delta/README.md.
// Interface and rationale: delta_table.h.
// =============================================================================

#include "delta_table.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>

static inline size_t kf_delta_full_block_bytes(int width, uint32_t B)
{
    return (size_t)width + (size_t)(B - 1);
}

// --- one-time converters: flat table -> delta table ------------------------

bool kf_convert_prime_table_to_delta(const char *flat_path, const char *delta_path,
                                     const char *log_prefix)
{
    const char *lp = log_prefix ? log_prefix : "";
    FILE *in = fopen(flat_path, "rb");
    if (!in)
    {
        fprintf(stderr, "%sCannot open %s\n", lp, flat_path);
        return false;
    }

    uint64_t magic, count;
    if (fread(&magic, 8, 1, in) != 1 || magic != KF_PRIME_TABLE_MAGIC ||
        fread(&count, 8, 1, in) != 1)
    {
        fprintf(stderr, "%sBad flat table header in %s\n", lp, flat_path);
        fclose(in);
        return false;
    }

    FILE *out = fopen(delta_path, "wb");
    if (!out)
    {
        fprintf(stderr, "%sCannot create %s\n", lp, delta_path);
        fclose(in);
        return false;
    }

    uint64_t dmagic = KF_DELTA_MAGIC_U32;
    uint32_t B = KF_DELTA_BLOCK, width = 4;
    fwrite(&dmagic, 8, 1, out);
    fwrite(&count, 8, 1, out);
    fwrite(&B, 4, 1, out);
    fwrite(&width, 4, 1, out);

    uint32_t *buf = (uint32_t *)malloc((size_t)B * 4);
    uint8_t *gb = (uint8_t *)malloc((size_t)B);
    uint64_t written = 0;
    unsigned maxgh = 0;
    bool ok = true;
    while (written < count)
    {
        size_t want = (size_t)((count - written < B) ? (count - written) : B);
        if (fread(buf, 4, want, in) != want)
        {
            ok = false;
            break;
        }
        fwrite(&buf[0], 4, 1, out); // anchor
        for (size_t i = 1; i < want; i++)
        {
            uint32_t gap = buf[i] - buf[i - 1];
            unsigned gh = gap >> 1;
            if ((gap & 1u) || gh > 255)
            {
                fprintf(stderr, "%sgap/2=%u at prime %u exceeds one byte — delta scheme invalid\n",
                        lp, gh, buf[i]);
                ok = false;
                break;
            }
            gb[i - 1] = (uint8_t)gh;
            if (gh > maxgh)
                maxgh = gh;
        }
        if (!ok)
            break;
        if (want > 1)
            fwrite(gb, 1, want - 1, out);
        written += want;
    }
    free(buf);
    free(gb);
    fclose(in);
    fclose(out);
    if (!ok)
    {
        remove(delta_path);
        return false;
    }
    printf("%sConverted %s (%" PRIu64 " primes) -> %s  [1 byte/prime, max gap/2=%u, ~%.0f MB]\n",
           lp, flat_path, count, delta_path, maxgh,
           (double)(KF_DELTA_HDR + count) / 1e6);
    return true;
}

bool kf_convert_prime_table_to_delta_u64(const char *flat_path, const char *delta_path,
                                         const char *log_prefix)
{
    const char *lp = log_prefix ? log_prefix : "";
    FILE *in = fopen(flat_path, "rb");
    if (!in)
    {
        fprintf(stderr, "%sCannot open %s\n", lp, flat_path);
        return false;
    }

    uint64_t magic, count;
    if (fread(&magic, 8, 1, in) != 1 || magic != K35U_PRIME_TABLE_MAGIC ||
        fread(&count, 8, 1, in) != 1)
    {
        fprintf(stderr, "%sBad u64 flat table header in %s\n", lp, flat_path);
        fclose(in);
        return false;
    }

    FILE *out = fopen(delta_path, "wb");
    if (!out)
    {
        fprintf(stderr, "%sCannot create %s\n", lp, delta_path);
        fclose(in);
        return false;
    }

    uint64_t dmagic = KF_DELTA_MAGIC_U64;
    uint32_t B = KF_DELTA_BLOCK, width = 8;
    fwrite(&dmagic, 8, 1, out);
    fwrite(&count, 8, 1, out);
    fwrite(&B, 4, 1, out);
    fwrite(&width, 4, 1, out);

    uint64_t *buf = (uint64_t *)malloc((size_t)B * 8);
    uint8_t *gb = (uint8_t *)malloc((size_t)B);
    uint64_t written = 0;
    unsigned maxgh = 0;
    bool ok = true;
    while (written < count)
    {
        size_t want = (size_t)((count - written < B) ? (count - written) : B);
        if (fread(buf, 8, want, in) != want)
        {
            ok = false;
            break;
        }
        fwrite(&buf[0], 8, 1, out); // anchor
        for (size_t i = 1; i < want; i++)
        {
            uint64_t gap = buf[i] - buf[i - 1];
            uint64_t gh = gap >> 1;
            if ((gap & 1ull) || gh > 255)
            {
                fprintf(stderr, "%sgap/2=%" PRIu64 " at prime %" PRIu64 " exceeds one byte — delta scheme invalid\n",
                        lp, gh, buf[i]);
                ok = false;
                break;
            }
            gb[i - 1] = (uint8_t)gh;
            if ((unsigned)gh > maxgh)
                maxgh = (unsigned)gh;
        }
        if (!ok)
            break;
        if (want > 1)
            fwrite(gb, 1, want - 1, out);
        written += want;
    }
    free(buf);
    free(gb);
    fclose(in);
    fclose(out);
    if (!ok)
    {
        remove(delta_path);
        return false;
    }
    printf("%sConverted %s (%" PRIu64 " primes) -> %s  [1 byte/prime, max gap/2=%u, ~%.0f MB]\n",
           lp, flat_path, count, delta_path, maxgh,
           (double)(KF_DELTA_HDR + count) / 1e6);
    return true;
}

// --- slice loading ---------------------------------------------------------

// Read the anchor (first prime) of block g at its O(1) file offset.
static uint64_t kf_delta_read_anchor(FILE *f, int width, uint32_t B, uint64_t g)
{
    off_t off = (off_t)KF_DELTA_HDR + (off_t)g * (off_t)kf_delta_full_block_bytes(width, B);
    fseeko(f, off, SEEK_SET);
    if (width == 8)
    {
        uint64_t a;
        return (fread(&a, 8, 1, f) == 1) ? a : UINT64_MAX;
    }
    uint32_t a;
    return (fread(&a, 4, 1, f) == 1) ? (uint64_t)a : UINT64_MAX;
}

// First block index whose anchor is strictly greater than x (== n_blocks if none).
static uint64_t kf_delta_first_gt(FILE *f, int width, uint32_t B,
                                  uint64_t n_blocks, uint64_t x)
{
    uint64_t lo = 0, hi = n_blocks;
    while (lo < hi)
    {
        uint64_t mid = lo + (hi - lo) / 2;
        if (kf_delta_read_anchor(f, width, B, mid) <= x)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

bool kf_load_delta_slice(kf_delta_slice_t *out, const char *path,
                         uint64_t range_lo, uint64_t range_hi)
{
    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f)
        return false;

    uint64_t magic, count;
    uint32_t B, width;
    if (fread(&magic, 8, 1, f) != 1 ||
        (magic != KF_DELTA_MAGIC_U32 && magic != KF_DELTA_MAGIC_U64) ||
        fread(&count, 8, 1, f) != 1 || fread(&B, 4, 1, f) != 1 ||
        fread(&width, 4, 1, f) != 1)
    {
        fclose(f);
        return false;
    }

    out->total_count = count;
    out->block_size = B;
    out->width = (int)width;
    out->range_lo = range_lo;
    out->range_hi = range_hi;

    if (count == 0)
    {
        out->bytes = (uint8_t *)malloc(1);
        fclose(f);
        return true; // empty but valid
    }

    uint64_t n_blocks_total = (count + B - 1) / B;

    // Block g_start possibly contains range_lo; block g_end possibly contains range_hi.
    uint64_t g_start = kf_delta_first_gt(f, width, B, n_blocks_total, range_lo);
    if (g_start > 0)
        g_start--; // step back to the block that may hold range_lo
    uint64_t first_gt_hi = kf_delta_first_gt(f, width, B, n_blocks_total, range_hi);

    if (first_gt_hi == 0 || first_gt_hi <= g_start)
    {
        // Whole table above range_hi, or empty intersection.
        out->bytes = (uint8_t *)malloc(1);
        out->n_blocks = 0;
        out->first_block_index = g_start;
        fclose(f);
        return true;
    }
    uint64_t g_end = first_gt_hi - 1;

    size_t full = kf_delta_full_block_bytes(width, B);
    uint64_t nb = g_end - g_start + 1;
    uint64_t last_len = count - (n_blocks_total - 1) * B; // length of the file's final block
    size_t bytes_len;
    if (g_end == n_blocks_total - 1)
        bytes_len = (size_t)(nb - 1) * full + ((size_t)width + (size_t)(last_len - 1));
    else
        bytes_len = (size_t)nb * full;

    uint8_t *bytes = (uint8_t *)malloc(bytes_len);
    if (!bytes)
    {
        fclose(f);
        return false;
    }
    fseeko(f, (off_t)KF_DELTA_HDR + (off_t)g_start * (off_t)full, SEEK_SET);
    if (fread(bytes, 1, bytes_len, f) != bytes_len)
    {
        free(bytes);
        fclose(f);
        return false;
    }
    fclose(f);

    out->bytes = bytes;
    out->n_blocks = (uint32_t)nb;
    out->first_block_index = g_start;
    return true;
}

void kf_delta_slice_free(kf_delta_slice_t *s)
{
    if (s && s->bytes)
    {
        free(s->bytes);
        s->bytes = nullptr;
    }
}

// --- trial division over a delta slice -------------------------------------
//
// Structure mirrors kf_trial_divide_table exactly: a shared cofactor stripped
// only under cof_lock, a thread-private local_cof for the lock-free pre-check,
// and a `done` early-out once the cofactor reaches 1.  The only differences are
// (a) the outer loop walks blocks and decodes each into a stack buffer, and
// (b) primes outside [range_lo, range_hi] are skipped (the slice over-reads at
// most one block at each end, so this restores exact per-rank boundaries).

void kf_trial_divide_delta(const mpz_t n, const kf_delta_slice_t *slice,
                           kfactor_list_t *found,
                           int64_t *thread_tested, int *thread_found,
                           int nthreads_out)
{
    if (!slice || slice->n_blocks == 0)
        return;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    kfactor_list_t *tf = (kfactor_list_t *)malloc((size_t)nthreads * sizeof(kfactor_list_t));
    for (int t = 0; t < nthreads; t++)
        kfl_init(&tf[t]);

    mpz_t cofactor;
    mpz_init_set(cofactor, n);
    omp_lock_t cof_lock;
    omp_init_lock(&cof_lock);
    volatile int done = 0;

    const uint32_t B = slice->block_size;
    const int width = slice->width;
    const size_t full = kf_delta_full_block_bytes(width, B);
    const uint64_t total = slice->total_count;
    const uint64_t lo = slice->range_lo, hi = slice->range_hi;

#ifdef _OPENMP
#pragma omp parallel
    {
        int t = omp_get_thread_num();
        mpz_t local_cof;
        omp_set_lock(&cof_lock);
        mpz_init_set(local_cof, cofactor);
        omp_unset_lock(&cof_lock);
        uint64_t *buf = (uint64_t *)malloc((size_t)B * sizeof(uint64_t));

#pragma omp for schedule(static)
        for (uint32_t j = 0; j < slice->n_blocks; j++)
        {
            if (done)
                continue;
            uint64_t g = slice->first_block_index + j;
            uint64_t base = g * B;
            uint32_t len = (base + B <= total) ? B : (uint32_t)(total - base);
            const uint8_t *r = slice->bytes + (size_t)j * full;
            uint64_t cur;
            if (width == 8)
            {
                memcpy(&cur, r, 8);
                r += 8;
            }
            else
            {
                uint32_t c32;
                memcpy(&c32, r, 4);
                cur = c32;
                r += 4;
            }
            buf[0] = cur;
            for (uint32_t k = 1; k < len; k++)
            {
                cur += (uint64_t)(*r++) << 1;
                buf[k] = cur;
            }
            for (uint32_t k = 0; k < len; k++)
            {
                uint64_t p = buf[k];
                if (p < lo || p > hi)
                    continue; // exact per-rank boundary
                if (thread_tested && t < nthreads_out)
                    thread_tested[t]++;
                if (mpz_tdiv_ui(local_cof, (unsigned long)p) != 0)
                    continue;
                omp_set_lock(&cof_lock);
                int e = 0;
                while (mpz_divisible_ui_p(cofactor, (unsigned long)p))
                {
                    mpz_divexact_ui(cofactor, cofactor, (unsigned long)p);
                    e++;
                }
                if (mpz_cmp_ui(cofactor, 1) == 0)
                    done = 1;
                mpz_set(local_cof, cofactor);
                omp_unset_lock(&cof_lock);
                if (e > 0)
                {
                    kfl_append_ul(&tf[t], p, e);
                    if (thread_found && t < nthreads_out)
                        thread_found[t]++;
                }
            }
        }
        free(buf);
        mpz_clear(local_cof);
    } // omp parallel
#else
    {
        int t = 0;
        uint64_t *buf = (uint64_t *)malloc((size_t)B * sizeof(uint64_t));
        for (uint32_t j = 0; j < slice->n_blocks && !done; j++)
        {
            uint64_t g = slice->first_block_index + j;
            uint64_t base = g * B;
            uint32_t len = (base + B <= total) ? B : (uint32_t)(total - base);
            const uint8_t *r = slice->bytes + (size_t)j * full;
            uint64_t cur;
            if (width == 8)
            {
                memcpy(&cur, r, 8);
                r += 8;
            }
            else
            {
                uint32_t c32;
                memcpy(&c32, r, 4);
                cur = c32;
                r += 4;
            }
            buf[0] = cur;
            for (uint32_t k = 1; k < len; k++)
            {
                cur += (uint64_t)(*r++) << 1;
                buf[k] = cur;
            }
            for (uint32_t k = 0; k < len; k++)
            {
                uint64_t p = buf[k];
                if (p < lo || p > hi)
                    continue;
                if (thread_tested && t < nthreads_out)
                    thread_tested[t]++;
                if (mpz_tdiv_ui(cofactor, (unsigned long)p) != 0)
                    continue;
                int e = 0;
                while (mpz_divisible_ui_p(cofactor, (unsigned long)p))
                {
                    mpz_divexact_ui(cofactor, cofactor, (unsigned long)p);
                    e++;
                }
                if (mpz_cmp_ui(cofactor, 1) == 0)
                    done = 1;
                if (e > 0)
                {
                    kfl_append_ul(&tf[t], p, e);
                    if (thread_found && t < nthreads_out)
                        thread_found[t]++;
                }
            }
        }
        free(buf);
    }
#endif

    omp_destroy_lock(&cof_lock);
    mpz_clear(cofactor);

    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            kfl_append_mpz(found, tf[t].items[i].prime, tf[t].items[i].exponent);
        kfl_free(&tf[t]);
    }
    free(tf);
}
