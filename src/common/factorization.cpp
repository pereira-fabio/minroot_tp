// _GNU_SOURCE (before any system header) exposes sched_getaffinity / CPU_COUNT,
// used to size the OpenMP reconstruction to the cores actually granted to this
// process (see ksc_available_cpus).
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "factorization.h"
#include "common.h"

#if defined(__linux__)
#include <sched.h>
#endif

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <cerrno> // for errno in kf_generate_and_save_prime_table_u64

// ---------------------------------------------------------------------------
// k32 — Small prime generator
// ---------------------------------------------------------------------------

unsigned *kf_generate_small_primes(int *count)
{
    const int limit = 65536;
    char *sieve = (char *)calloc((size_t)(limit + 1), 1);
    memset(sieve, 1, (size_t)(limit + 1));
    sieve[0] = sieve[1] = 0;
    for (int i = 2; (long long)i * i <= limit; i++)
        if (sieve[i])
            for (int j = i * i; j <= limit; j += i)
                sieve[j] = 0;
    *count = 0;
    for (int i = 2; i <= limit; i++)
        if (sieve[i])
            (*count)++;
    unsigned *primes = (unsigned *)malloc((size_t)(*count) * sizeof(unsigned));
    int idx = 0;
    for (int i = 2; i <= limit; i++)
        if (sieve[i])
            primes[idx++] = (unsigned)i;
    free(sieve);
    return primes;
}

// ---------------------------------------------------------------------------
// k32 — Prime table generation
// ---------------------------------------------------------------------------

bool kf_generate_and_save_prime_table(const char *path,
                                      int sp_count, const unsigned *sp,
                                      const char *log_prefix)
{
    printf("%sGenerating prime table [65537, 2^32] -> %s...\n", log_prefix ? log_prefix : "", path);
    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "Cannot create %s\n", path);
        return false;
    }

    uint64_t magic = KF_PRIME_TABLE_MAGIC,
             count = 0;
    fwrite(&magic, 8, 1, f);
    long count_pos = ftell(f); // remember where to write the final count
    fwrite(&count, 8, 1, f);   // placeholder

    char *seg = (char *)malloc(KF_SEG_SIZE);
    uint32_t *batch = (uint32_t *)malloc(KF_WRITE_BATCH * sizeof(uint32_t));
    int bi = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t lo = 65537ULL; lo <= (uint64_t)UINT32_MAX; lo += KF_SEG_SIZE)
    {
        uint64_t hi = lo + KF_SEG_SIZE - 1;
        if (hi > (uint64_t)UINT32_MAX)
            hi = (uint64_t)UINT32_MAX;
        size_t seg_len = (size_t)(hi - lo + 1);

        memset(seg, 1, seg_len);
        for (int i = 0; i < sp_count; i++)
        {
            uint64_t p = sp[i];
            uint64_t start = ((lo + p - 1) / p) * p;
            if (start == p)
                start += p;
            for (uint64_t j = start; j <= hi; j += p)
                seg[j - lo] = 0;
        }
        for (uint64_t c = lo; c <= hi; c++)
        {
            if (!seg[c - lo])
                continue;
            batch[bi++] = (uint32_t)c;
            count++;
            if (bi == KF_WRITE_BATCH)
            {
                fwrite(batch, 4, (size_t)KF_WRITE_BATCH, f);
                bi = 0;
            }
        }
    }
    if (bi > 0)
        fwrite(batch, 4, (size_t)bi, f);
    free(seg);
    free(batch);

    // write actual prime count into the header placeholder
    fseek(f, count_pos, SEEK_SET);
    fwrite(&count, 8, 1, f);
    fclose(f);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("%sGenerated %" PRIu64 " primes in %.1f ms  (file ~%.0f MB)\n",
           log_prefix ? log_prefix : "", count, ms, (double)(16 + count * 4) / 1e6);
    return true;
}

// ---------------------------------------------------------------------------
// k32 — Load full prime table
// ---------------------------------------------------------------------------

uint32_t *kf_load_prime_table(uint64_t *out_count, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    uint64_t magic, count;
    if (fread(&magic, 8, 1, f) != 1 || magic != KF_PRIME_TABLE_MAGIC ||
        fread(&count, 8, 1, f) != 1)
    {
        fclose(f);
        return NULL;
    }
    uint32_t *primes = (uint32_t *)malloc(count * 4);
    if (!primes || fread(primes, 4, count, f) != count)
    {
        free(primes);
        fclose(f);
        return NULL;
    }
    fclose(f);
    printf("Loaded %" PRIu64 " primes from %s\n", count, path);
    *out_count = count;
    return primes;
}

// ---------------------------------------------------------------------------
// k32 — Binary search within prime table file (internal helper)
// ---------------------------------------------------------------------------

static uint64_t prime_file_lower_bound(FILE *f, uint64_t total_count,
                                       uint32_t target)
{
    uint64_t lo = 0, hi = total_count;
    while (lo < hi)
    {
        uint64_t mid = lo + (hi - lo) / 2;
        uint32_t p;
        fseek(f, (long)(16LL + (long long)mid * 4LL), SEEK_SET);
        if (fread(&p, 4, 1, f) != 1)
            break;
        if (p < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// k32 — Load rank-specific prime slice (MPI path)
// ---------------------------------------------------------------------------

uint32_t *kf_load_prime_slice(uint64_t *slice_count, const char *path,
                              uint32_t range_lo, uint32_t range_hi)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    uint64_t magic, total_count;
    if (fread(&magic, 8, 1, f) != 1 || magic != KF_PRIME_TABLE_MAGIC ||
        fread(&total_count, 8, 1, f) != 1)
    {
        fclose(f);
        return NULL;
    }

    uint64_t start_idx = prime_file_lower_bound(f, total_count, range_lo);
    uint64_t end_idx;
    if (range_hi >= (uint32_t)UINT32_MAX)
        end_idx = total_count;
    else
        end_idx = prime_file_lower_bound(f, total_count, range_hi + 1U);

    *slice_count = end_idx - start_idx;
    if (*slice_count == 0)
    {
        fclose(f);
        return (uint32_t *)malloc(4);
    } // empty, not NULL

    uint32_t *slice = (uint32_t *)malloc(*slice_count * 4);
    if (!slice)
    {
        fclose(f);
        return NULL;
    }

    fseek(f, (long)(16LL + (long long)start_idx * 4LL), SEEK_SET);
    if (fread(slice, 4, *slice_count, f) != *slice_count)
    {
        free(slice);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return slice;
}

// ---------------------------------------------------------------------------
// Shared pieces of the k32 trial-division loops
// ---------------------------------------------------------------------------

static_assert(sizeof(unsigned long) == 8,
              "paired trial division packs two primes into one unsigned long");

// Strip every power of p from the shared cofactor under the lock and resync
// the caller's snapshot. Returns the exponent (0 for a stale snapshot hit).
static int kf_strip_prime(mpz_t cofactor, mpz_t local_cof, omp_lock_t *lock,
                          unsigned long p, bool *now_one)
{
    omp_set_lock(lock);
    int e = 0;
    while (mpz_divisible_ui_p(cofactor, p))
    {
        mpz_divexact_ui(cofactor, cofactor, p);
        e++;
    }
    *now_one = (mpz_cmp_ui(cofactor, 1) == 0);
    // Sync thread-local snapshot so subsequent pre-checks stay accurate
    // (reflects factors removed by all threads so far).
    mpz_set(local_cof, cofactor);
    omp_unset_lock(lock);
    return e;
}

// Divisibility pre-check for a pair of candidates (b == 0: a alone).
//
// mpz_tdiv_ui walks every limb of the snapshot, and up to a 62-bit divisor
// its cost is set by |y|, not by the divisor. Two primes whose product is
// below 2^62 share one walk: r = y mod (a*b), then a | y iff a | r.
//
// The 2^62 cap matters: GMP's fast remainder loop needs two spare bits in the
// divisor, so a 63-bit divisor costs ~1.7x and a 64-bit one ~3.2x a 62-bit one
// (measured at 24576 bits). Pairing two primes near 2^32 would be SLOWER than
// testing them apart, so only primes below ~2^31 pair; the rest, and k35's
// upper range, take one walk each exactly as before.
#define KF_PAIR_LIMIT (1UL << 62)
static inline void kf_pair_check(const mpz_t local_cof, uint64_t a, uint64_t b,
                                 bool *hit_a, bool *hit_b)
{
    unsigned long ab;
    if (b != 0 && !__builtin_mul_overflow((unsigned long)a, (unsigned long)b, &ab) &&
        ab < KF_PAIR_LIMIT)
    {
        unsigned long r = mpz_tdiv_ui(local_cof, ab);
        *hit_a = (r % a == 0);
        *hit_b = (r % b == 0);
        return;
    }
    *hit_a = (mpz_tdiv_ui(local_cof, (unsigned long)a) == 0);
    *hit_b = (b != 0 && mpz_tdiv_ui(local_cof, (unsigned long)b) == 0);
}

// ---------------------------------------------------------------------------
// kf_cost_split — see factorization.h
//
// Constants: 2^24-integer windows near 2^28 (paired) and 2^32 (unpaired), one
// thread. The sieve alone (64-bit y) costs ~2.1 ns per integer; one unpaired
// test 10.7 / 42 / 59 / 221 ns at 1024 / 4096 / 8192 / 24576 bits (fit:
// 5 + 0.55 * limbs); a paired test 0.29-0.46 of that (0.45 used, the large-
// size value where trial division dominates).
// ---------------------------------------------------------------------------

#define KF_COST_STEPS 4096

// Boundary k of the split: the x where the cumulative cost reaches k/parts of
// the total, by midpoint-rule integration over KF_COST_STEPS equal cells and
// linear interpolation inside the cell. Pure arithmetic on the arguments, so
// every rank and thread gets bit-identical boundaries.
static uint64_t kf_cost_boundary(uint64_t lo, uint64_t hi, int parts, int k,
                                 int limbs, bool sieve)
{
    if (k <= 0)
        return lo;
    if (k >= parts)
        return hi + 1;
    const double a = (double)lo, width = (double)(hi - lo + 1);
    const double cell = width / KF_COST_STEPS;
    const double fixed = sieve ? KF_SIEVE_NS : 0.0;
    const double test = KF_TD_BASE_NS + KF_TD_LIMB_NS * limbs;
    const bool flat = !sieve && kf_get_td_mode() == KF_TD_GCD;
    double cum[KF_COST_STEPS + 1];
    cum[0] = 0.0;
    for (int i = 0; i < KF_COST_STEPS; i++)
    {
        double x = a + (i + 0.5) * cell;
        double w = (x < 2147483648.0) ? KF_PAIRED_COST : 1.0;
        cum[i + 1] = cum[i] + (flat ? 1.0 : fixed + w * test / log(x < 3.0 ? 3.0 : x)) * cell;
    }
    double target = cum[KF_COST_STEPS] * k / parts;
    int i = 0;
    while (i < KF_COST_STEPS - 1 && cum[i + 1] < target)
        i++;
    double frac = (cum[i + 1] > cum[i]) ? (target - cum[i]) / (cum[i + 1] - cum[i]) : 0.0;
    uint64_t b = lo + (uint64_t)((i + frac) * cell);
    if (b < lo)
        b = lo;
    if (b > hi + 1)
        b = hi + 1;
    return b;
}

void kf_cost_split(uint64_t lo, uint64_t hi, int parts, int idx, int limbs, bool sieve,
                   uint64_t *out_lo, uint64_t *out_hi)
{
    uint64_t b0 = kf_cost_boundary(lo, hi, parts, idx, limbs, sieve);
    uint64_t b1 = kf_cost_boundary(lo, hi, parts, idx + 1, limbs, sieve);
    if (b1 <= b0)
    {
        *out_lo = 1; // empty piece
        *out_hi = 0;
        return;
    }
    *out_lo = b0;
    *out_hi = b1 - 1;
}

// ---------------------------------------------------------------------------
// Checkpoint between trial-division phases (see factorization.h). Called by
// every thread of the team. The master exports this rank's new factors (each
// thread's list past exported[t]), runs `share`, and divides every returned
// factor out of the shared cofactor; own factors are already gone and simply
// fail the divisibility test. Then every thread refreshes its snapshot.
// ---------------------------------------------------------------------------

static void kf_share_checkpoint(kf_share_fn share, mpz_t cofactor, omp_lock_t *lock,
                                kfactor_list_t *tf, int nthreads, int *exported,
                                mpz_t local_cof, volatile int *done)
{
#ifdef _OPENMP
#pragma omp barrier
#pragma omp master
#endif
    {
        if (share)
        {
            kfactor_list_t mine, all;
            kfl_init(&mine);
            kfl_init(&all);
            for (int t = 0; t < nthreads; t++)
            {
                for (int i = exported[t]; i < tf[t].count; i++)
                    kfl_append_mpz(&mine, tf[t].items[i].prime, tf[t].items[i].exponent);
                exported[t] = tf[t].count;
            }
            share(&mine, &all);
            omp_set_lock(lock);
            for (int i = 0; i < all.count; i++)
                while (mpz_divisible_p(cofactor, all.items[i].prime))
                    mpz_divexact(cofactor, cofactor, all.items[i].prime);
            if (mpz_cmp_ui(cofactor, 1) == 0)
                *done = 1;
            omp_unset_lock(lock);
            kfl_free(&mine);
            kfl_free(&all);
        }
    }
#ifdef _OPENMP
#pragma omp barrier
#endif
    omp_set_lock(lock);
    mpz_set(local_cof, cofactor);
    omp_unset_lock(lock);
}

// ---------------------------------------------------------------------------
// k32 — Trial division: sieve method
// ---------------------------------------------------------------------------

void kf_trial_divide_sieve(const mpz_t n,
                           uint64_t range_lo, uint64_t range_hi,
                           int sp_count, const unsigned *sp,
                           kfactor_list_t *found,
                           int64_t *thread_tested, int *thread_found,
                           int nthreads_out, kf_share_fn share)
{
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    kfactor_list_t *tf = (kfactor_list_t *)malloc(
        (size_t)nthreads * sizeof(kfactor_list_t));
    for (int t = 0; t < nthreads; t++)
        kfl_init(&tf[t]);
    int *exported = (int *)calloc((size_t)nthreads, sizeof(int));

    // Shared cofactor for early termination.
    // Each thread divides out its found factors under a mutex.
    // When cofactor reaches 1, all factors have been found — stop scanning.
    mpz_t cofactor;
    mpz_init_set(cofactor, n);
    omp_lock_t cof_lock;
    omp_init_lock(&cof_lock);
    volatile int done = 0; // set to 1 when cofactor == 1

#ifdef _OPENMP
#pragma omp parallel
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int t = 0, nt = 1;
#endif
        // Threads split the rank's range by cost, like ranks split the whole.
        uint64_t t_lo, t_hi;
        kf_cost_split(range_lo, range_hi, nt, t, (int)mpz_size(n), true, &t_lo, &t_hi);

        char *seg = (char *)malloc(KF_SEG_SIZE);

        // Thread-local snapshot of cofactor for the divisibility pre-check.
        //
        // The shared mpz_t cofactor is only modified under cof_lock, but the
        // pre-check `mpz_tdiv_ui(cofactor, c)` previously read it lock-free.
        // Concurrent reads of a GMP mpz_t while another thread writes its limbs
        // is a data race: mpz_divexact_ui can change limb count and pointers,
        // so a concurrent mpz_tdiv_ui may read partially-updated state and return
        // a wrong remainder, causing a prime factor to be silently missed.
        //
        // Fix: each thread keeps local_cof, a thread-private copy used for the
        // pre-check. It is initialized under the lock and re-synced whenever
        // this thread acquires the lock and processes a factor. This guarantees:
        //   - No data race (local_cof is thread-private; cofactor only read under lock)
        //   - No missed factors: local_cof >= actual cofactor always (factors only
        //     decrease cofactor), so if c divides actual cofactor it also divides
        //     local_cof — no false negatives from the pre-check.
        //   - Occasional false positives (local_cof stale → pre-check passes but
        //     re-check inside lock fails) are harmless: e=0, nothing appended.
        mpz_t local_cof;
        omp_set_lock(&cof_lock);
        mpz_init_set(local_cof, cofactor);
        omp_unset_lock(&cof_lock);

        // The segment holds odd numbers only: seg[k] stands for lo + 2k. Even
        // candidates other than 2 are never prime, so storing them doubled the
        // memset, the scan and the work of the p = 2 marking pass for nothing.
        // 2 itself, when in range, is queued as the first candidate.
        uint64_t first_odd = t_lo | 1;

        // Incremental segment offsets: compute the first odd multiple of each
        // odd small prime once, then carry it forward across segments in steps
        // of 2p. This removes the per-prime, per-segment division from the hot
        // loop — a leading cost for the large k35 range.
        uint64_t *next_mult = (uint64_t *)malloc((size_t)sp_count * sizeof(uint64_t));
        for (int i = 0; i < sp_count; i++)
        {
            uint64_t p = sp[i];
            if (p == 2)
            {
                next_mult[i] = UINT64_MAX; // no odd multiples to mark
                continue;
            }
            uint64_t start = ((first_odd + p - 1) / p) * p;
            if ((start & 1) == 0)
                start += p;
            if (start == p) // never sieve the prime itself out
                start += 2 * p;
            next_mult[i] = start;
        }

        // Candidates are tested in pairs (kf_pair_check); `pend` holds the
        // first of a pair until its partner arrives, 0 when none is waiting.
        uint64_t pend = (t_lo <= 2 && 2 <= t_hi) ? 2 : 0;
        auto test_pair = [&](uint64_t a, uint64_t b)
        {
            bool hit_a, hit_b;
            kf_pair_check(local_cof, a, b, &hit_a, &hit_b);
            uint64_t cand[2] = {a, b};
            bool hit[2] = {hit_a, hit_b};
            for (int k = 0; k < 2; k++)
            {
                if (!hit[k])
                    continue;
                bool now_one = false;
                int e = kf_strip_prime(cofactor, local_cof, &cof_lock,
                                       (unsigned long)cand[k], &now_one);
                if (now_one)
                    done = 1;
                if (e > 0)
                    kfl_append_ul(&tf[t], (unsigned long)cand[k], e);
            }
        };

        // The thread's odd candidates are scanned in KF_SHARE_PHASES phases
        // with a cofactor checkpoint between them. Phase k covers the k-th
        // equal slice of the candidates, so every thread does the same share
        // of its work per phase; whole segments per phase would round apart
        // when a thread has only a few, and the checkpoint barrier would then
        // serialise threads (measured 2x slower on 1-2 segments per thread).
        // Segments end early at phase boundaries; next_mult carries across
        // any contiguous run of odd starts. Every thread passes every
        // checkpoint, even with nothing left to scan or `done` set.
        const uint64_t span = 2 * (uint64_t)KF_SEG_SIZE;
        const uint64_t n_odd = (t_hi >= first_odd && t_lo <= t_hi)
                                   ? (t_hi - first_odd) / 2 + 1
                                   : 0;
        for (int phase = 0; phase < KF_SHARE_PHASES; phase++)
        {
            if (phase > 0)
                kf_share_checkpoint(share, cofactor, &cof_lock, tf, nthreads,
                                    exported, local_cof, &done);
            const uint64_t i_lo = n_odd * (uint64_t)phase / KF_SHARE_PHASES;
            const uint64_t i_hi = n_odd * (uint64_t)(phase + 1) / KF_SHARE_PHASES;
            if (i_hi <= i_lo)
                continue;
            const uint64_t ph_lo = first_odd + 2 * i_lo;
            const uint64_t ph_hi = first_odd + 2 * (i_hi - 1);
            for (uint64_t lo = ph_lo; lo <= ph_hi && !done; lo += span)
            {
                uint64_t hi = lo + span - 2;
                if (hi > ph_hi)
                    hi = ph_hi;
                size_t seg_len = (size_t)((hi - lo) / 2 + 1);

                memset(seg, 1, seg_len);
                if (lo == 1)
                    seg[0] = 0; // 1 is not a prime
                for (int i = 0; i < sp_count; i++)
                {
                    uint64_t j = next_mult[i];
                    if (j > hi)
                        continue;
                    uint64_t step = 2 * (uint64_t)sp[i];
                    for (; j <= hi; j += step)
                        seg[(j - lo) >> 1] = 0;
                    next_mult[i] = j; // first odd multiple > hi — carry to next segment
                }
                for (size_t k = 0; k < seg_len; k++)
                {
                    if (!seg[k])
                        continue;
                    uint64_t c = lo + 2 * (uint64_t)k;
                    if (pend == 0)
                    {
                        pend = c;
                        continue;
                    }
                    test_pair(pend, c);
                    pend = 0;
                }
            }
        }
        if (pend != 0 && !done)
            test_pair(pend, 0);

        mpz_clear(local_cof);
        free(next_mult);
        free(seg);
    }

    omp_destroy_lock(&cof_lock);
    mpz_clear(cofactor);

    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            kfl_append_mpz(found, tf[t].items[i].prime, tf[t].items[i].exponent);
        kfl_free(&tf[t]);
    }
    free(tf);
    free(exported);
}

// ---------------------------------------------------------------------------
// Table method, gcd route — see factorization.h
// ---------------------------------------------------------------------------

static kf_td_mode_t g_td_mode = KF_TD_GCD;

void kf_set_td_mode(kf_td_mode_t mode) { g_td_mode = mode; }
kf_td_mode_t kf_get_td_mode(void) { return g_td_mode; }

bool kf_parse_td_arg(int *argc, char **argv)
{
    bool ok = true;
    int w = 1;
    for (int i = 1; i < *argc; i++)
    {
        if (strcmp(argv[i], "--td") == 0 && i + 1 < *argc)
        {
            const char *m = argv[++i];
            if (strcmp(m, "gcd") == 0)
                g_td_mode = KF_TD_GCD;
            else if (strcmp(m, "scan") == 0)
                g_td_mode = KF_TD_SCAN;
            else
            {
                fprintf(stderr, "Unknown --td '%s' (use gcd or scan)\n", m);
                ok = false;
            }
            continue;
        }
        argv[w++] = argv[i];
    }
    *argc = w;
    argv[w] = nullptr;
    return ok;
}

// Products of one prime table, one per chunk. Chunks are equal-width in value,
// which is equal in product bits (see kf_cost_split). There are at least as
// many as threads, and none holds more than KF_GCD_CHUNK primes: a product's
// build time grows faster than its size, and a reduction costs the product's
// bits whatever the chunking, so small chunks only add one cheap gcd each
// (~7 ns per prime at 24576 bits against ~200 ns per prime for the scan).
#define KF_GCD_CHUNK 65536
struct kf_table_products
{
    const uint32_t *primes;
    int64_t count;
    int nchunks;
    int64_t *bounds; // chunk c is primes[bounds[c], bounds[c+1])
    mpz_t *prods;
};

static kf_table_products *g_tp = nullptr;
static int g_tp_n = 0;

// Product of primes[a, b) by a balanced tree (linear accumulation would cost
// quadratic time on a 1.5M-bit result).
static void kf_prime_product(mpz_t r, const uint32_t *primes, int64_t a, int64_t b)
{
    if (b - a <= 16)
    {
        mpz_set_ui(r, 1);
        for (int64_t i = a; i < b; i++)
            mpz_mul_ui(r, r, primes[i]);
        return;
    }
    mpz_t l;
    mpz_init(l);
    int64_t m = a + (b - a) / 2;
    kf_prime_product(l, primes, a, m);
    kf_prime_product(r, primes, m, b);
    mpz_mul(r, r, l);
    mpz_clear(l);
}

static const kf_table_products *kf_table_products_get(const uint32_t *primes,
                                                      int64_t count)
{
    for (int i = 0; i < g_tp_n; i++)
        if (g_tp[i].primes == primes && g_tp[i].count == count)
            return &g_tp[i];

    int nchunks = 1;
#ifdef _OPENMP
    nchunks = omp_get_max_threads();
#endif
    if ((count + KF_GCD_CHUNK - 1) / KF_GCD_CHUNK > nchunks)
        nchunks = (int)((count + KF_GCD_CHUNK - 1) / KF_GCD_CHUNK);
    if (nchunks > count)
        nchunks = count > 0 ? (int)count : 1;
    kf_table_products tp;
    tp.primes = primes;
    tp.count = count;
    tp.nchunks = nchunks;
    tp.bounds = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nchunks + 1));
    tp.prods = (mpz_t *)malloc(sizeof(mpz_t) * (size_t)nchunks);
    tp.bounds[0] = 0;
    tp.bounds[nchunks] = count;
    for (int c = 1; c < nchunks; c++)
    {
        // First prime >= the c-th equal-width value boundary.
        double v = primes[0] + (double)(primes[count - 1] - primes[0]) * c / nchunks;
        int64_t lo = tp.bounds[c - 1], hi = count;
        while (lo < hi)
        {
            int64_t mid = lo + (hi - lo) / 2;
            if ((double)primes[mid] < v)
                lo = mid + 1;
            else
                hi = mid;
        }
        tp.bounds[c] = lo;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int c = 0; c < nchunks; c++)
    {
        mpz_init(tp.prods[c]);
        kf_prime_product(tp.prods[c], primes, tp.bounds[c], tp.bounds[c + 1]);
    }
    g_tp = (kf_table_products *)realloc(g_tp, sizeof(kf_table_products) * (size_t)(g_tp_n + 1));
    g_tp[g_tp_n] = tp;
    return &g_tp[g_tp_n++];
}

double kf_table_products_prepare(const uint32_t *primes, int64_t prime_count)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    if (g_td_mode == KF_TD_GCD && primes && prime_count > 0)
        kf_table_products_get(primes, prime_count);
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t0)
        .count();
}

void kf_table_products_release(const uint32_t *primes)
{
    for (int i = 0; i < g_tp_n; i++)
    {
        if (g_tp[i].primes != primes)
            continue;
        for (int c = 0; c < g_tp[i].nchunks; c++)
            mpz_clear(g_tp[i].prods[c]);
        free(g_tp[i].prods);
        free(g_tp[i].bounds);
        g_tp[i--] = g_tp[--g_tp_n];
    }
}

static void kf_trial_divide_gcd(const mpz_t n, const kf_table_products *tp,
                                kfactor_list_t *found,
                                int64_t *thread_tested, int *thread_found,
                                int nthreads_out)
{
    if (mpz_cmp_ui(n, 1) <= 0)
        return;
    const int nchunks = tp->nchunks;
    kfactor_list_t *cf = (kfactor_list_t *)malloc(sizeof(kfactor_list_t) * (size_t)nchunks);
    for (int c = 0; c < nchunks; c++)
        kfl_init(&cf[c]);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif
        mpz_t g;
        mpz_init(g);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int c = 0; c < nchunks; c++)
        {
            const int64_t a = tp->bounds[c], b = tp->bounds[c + 1];
            if (thread_tested && t < nthreads_out)
                thread_tested[t] += b - a;
            mpz_tdiv_r(g, tp->prods[c], n);
            mpz_gcd(g, n, g); // product of this chunk's primes dividing n
            // Split g, which is small: one word-sized remainder per prime,
            // stopping once every prime of g is accounted for.
            for (int64_t i = a; i < b && mpz_cmp_ui(g, 1) != 0; i++)
            {
                unsigned long p = tp->primes[i];
                if (mpz_tdiv_ui(g, p) != 0)
                    continue;
                mpz_divexact_ui(g, g, p);
                kfl_append_ul(&cf[c], p, count_exponent(n, p));
                if (thread_found && t < nthreads_out)
                    thread_found[t]++;
            }
        }
        mpz_clear(g);
    }

    for (int c = 0; c < nchunks; c++)
    {
        for (int i = 0; i < cf[c].count; i++)
            kfl_append_mpz(found, cf[c].items[i].prime, cf[c].items[i].exponent);
        kfl_free(&cf[c]);
    }
    free(cf);
}

// ---------------------------------------------------------------------------
// k32 — Trial division: table method
// ---------------------------------------------------------------------------

void kf_trial_divide_table(const mpz_t n,
                           const uint32_t *primes, int64_t prime_count,
                           kfactor_list_t *found,
                           int64_t *thread_tested, int *thread_found,
                           int nthreads_out, kf_share_fn share)
{
    if (g_td_mode == KF_TD_GCD)
    {
        // Always return here in gcd mode, even for an empty slice: the scan
        // below calls `share` collectively and must not run on some ranks only.
        if (prime_count > 0)
            kf_trial_divide_gcd(n, kf_table_products_get(primes, prime_count), found,
                                thread_tested, thread_found, nthreads_out);
        return;
    }

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    kfactor_list_t *tf = (kfactor_list_t *)malloc(
        (size_t)nthreads * sizeof(kfactor_list_t));
    for (int t = 0; t < nthreads; t++)
        kfl_init(&tf[t]);
    int *exported = (int *)calloc((size_t)nthreads, sizeof(int));

    // Shared cofactor for early termination.
    mpz_t cofactor;
    mpz_init_set(cofactor, n);
    omp_lock_t cof_lock;
    omp_init_lock(&cof_lock);
    volatile int64_t stop_idx = prime_count; // threads stop when idx >= stop_idx
    volatile int done = 0;                   // cofactor == 1 via shared factors

    // Thread-local snapshot of cofactor for the pre-check — same fix as
    // kf_trial_divide_sieve.  The shared cofactor is only written under
    // cof_lock; reading it lock-free from another thread is a data race: GMP's
    // mpz_divexact_ui can change _mp_size and realloc _mp_d, so a concurrent
    // mpz_tdiv_ui may see partially-updated state and return a corrupt value.
    // Worst case: cofactor appears as 0, making mpz_divisible_ui_p(0,p)=true
    // and mpz_divexact_ui(0,p)=0 inside the lock → infinite loop.
    // Fix: each thread keeps local_cof (thread-private), initialised under the
    // lock, re-synced after each factor is stripped.  local_cof >= actual
    // cofactor always, so no false negatives; stale hits just waste one lock
    // acquisition (e=0, nothing appended).
    //
    // Primes are tested two at a time (kf_pair_check): iteration j covers
    // primes[2j] and primes[2j+1], sharing one walk when their product allows.
    // Pairs below 2^31 cost about half those above, so a slice that straddles
    // it is handed out in dynamic chunks rather than equal static blocks.
    int64_t npairs = (prime_count + 1) / 2;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif

        mpz_t local_cof;
        omp_set_lock(&cof_lock);
        mpz_init_set(local_cof, cofactor);
        omp_unset_lock(&cof_lock);

        for (int phase = 0; phase < KF_SHARE_PHASES; phase++)
        {
            if (phase > 0)
                kf_share_checkpoint(share, cofactor, &cof_lock, tf, nthreads,
                                    exported, local_cof, &done);
            const int64_t j_lo = npairs * phase / KF_SHARE_PHASES;
            const int64_t j_hi = npairs * (phase + 1) / KF_SHARE_PHASES;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
            for (int64_t j = j_lo; j < j_hi; j++)
            {
                int64_t i0 = 2 * j;
                if (i0 >= stop_idx || done)
                    continue;
                int64_t np = (i0 + 1 < prime_count) ? 2 : 1;
                uint64_t cand[2] = {(uint64_t)primes[i0],
                                    np == 2 ? (uint64_t)primes[i0 + 1] : 0};
                if (thread_tested && t < nthreads_out)
                    thread_tested[t] += np;

                bool hit[2];
                kf_pair_check(local_cof, cand[0], cand[1], &hit[0], &hit[1]);
                for (int k = 0; k < np; k++)
                {
                    if (!hit[k])
                        continue;
                    bool now_one = false;
                    int e = kf_strip_prime(cofactor, local_cof, &cof_lock,
                                           (unsigned long)cand[k], &now_one);
                    if (now_one)
                        stop_idx = i0 + k + 1;
                    if (e > 0)
                    {
                        kfl_append_ul(&tf[t], (unsigned long)cand[k], e);
                        if (thread_found && t < nthreads_out)
                            thread_found[t]++;
                    }
                }
            }
        } // phases

        mpz_clear(local_cof);
    } // omp parallel

    omp_destroy_lock(&cof_lock);
    mpz_clear(cofactor);

    for (int t = 0; t < nthreads; t++)
    {
        for (int i = 0; i < tf[t].count; i++)
            kfl_append_mpz(found, tf[t].items[i].prime, tf[t].items[i].exponent);
        kfl_free(&tf[t]);
    }
    free(tf);
    free(exported);
}

// ---------------------------------------------------------------------------
// k16 — Horner trial division
// ---------------------------------------------------------------------------

void kf_trial_divide_k16(const unsigned char *challenge_bytes, size_t nbytes,
                         const mpz_t challenge,
                         const unsigned long *primes, int prime_count,
                         int *exponents)
{
    // OpenMP over the prime array: challenge_bytes and challenge are read-only.
    // exponents[i] is written only by the thread handling index i.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < prime_count; i++)
    {
        unsigned long rem = 0;
        for (size_t b = 0; b < nbytes; b++)
            rem = (rem * 256 + challenge_bytes[b]) % primes[i];
        if (rem != 0)
        {
            exponents[i] = 0;
            continue;
        }
        exponents[i] = count_exponent(challenge, primes[i]);
    }
}

// ---------------------------------------------------------------------------
// k64 — Miller-Rabin primality test
// ---------------------------------------------------------------------------

bool kf_is_prime(const mpz_t n)
{
    return mpz_probab_prime_p(n, 15) > 0;
}

// ---------------------------------------------------------------------------
// k64 — Pollard's rho (Brent variant)
// ---------------------------------------------------------------------------

bool kf_pollard_rho_brent(mpz_t factor, const mpz_t n, unsigned long c_val)
{
    if (mpz_cmp_ui(n, 1) == 0)
        return false;
    if (mpz_divisible_ui_p(n, 2))
    {
        mpz_set_ui(factor, 2);
        return true;
    }

    mpz_t x, y, ys, c, q, diff, g;
    mpz_inits(x, y, ys, c, q, diff, g, NULL);
    mpz_set_ui(c, c_val);

    mpz_set_ui(x, 2);
    mpz_set_ui(y, 2);
    mpz_set_ui(g, 1);

    const long m = 128; // batch GCD size
    long r = 1;

    while (mpz_cmp_ui(g, 1) == 0)
    {
        mpz_set(x, y);
        mpz_set_ui(q, 1); // reset accumulator for this Brent cycle

        // Advance y by r steps (the "giant step")
        for (long i = 0; i < r; i++)
        {
            mpz_mul(y, y, y);
            mpz_add(y, y, c);
            mpz_mod(y, y, n);
        }

        long k = 0;
        while (k < r && mpz_cmp_ui(g, 1) == 0)
        {
            mpz_set(ys, y); // save position for possible backtracking

            for (long i = 0; i < m && i < r - k; i++)
            {
                mpz_mul(y, y, y);
                mpz_add(y, y, c);
                mpz_mod(y, y, n);
                mpz_sub(diff, x, y);
                mpz_abs(diff, diff);
                mpz_mul(q, q, diff);
                mpz_mod(q, q, n);
            }
            mpz_gcd(g, q, n);
            k += m;
        }
        r *= 2;
    }

    // g == n: the batch product happened to be divisible by n (rare).
    // Backtrack from ys using individual GCD calls to find the factor.
    if (mpz_cmp(g, n) == 0)
    {
        mpz_set_ui(g, 1);
        while (mpz_cmp_ui(g, 1) == 0)
        {
            mpz_mul(ys, ys, ys);
            mpz_add(ys, ys, c);
            mpz_mod(ys, ys, n);
            mpz_sub(diff, x, ys);
            mpz_abs(diff, diff);
            mpz_gcd(g, diff, n);
        }
    }

    bool ok = (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, n) < 0);
    if (ok)
        mpz_set(factor, g);

    mpz_clears(x, y, ys, c, q, diff, g, NULL);
    return ok;
}

// ---------------------------------------------------------------------------
// k64 — Recursive complete factorization
// ---------------------------------------------------------------------------

void kf_factor_recursive(const mpz_t n, kfactor_list_t *factors,
                         unsigned long c_start)
{
    if (mpz_cmp_ui(n, 1) == 0)
        return;

    if (kf_is_prime(n))
    {
#pragma omp critical
        kfl_append_mpz(factors, n, 1);
        return;
    }

    // Small trial division fallback — handles tiny factors that Pollard's rho
    // is slow on (c=1 degeneracy), and avoids wasted parallel launches.
    for (unsigned long p = 2; p < 1000; p++)
    {
        if (mpz_divisible_ui_p(n, p))
        {
            mpz_t fp;
            mpz_init_set_ui(fp, p);
            mpz_t co;
            mpz_init(co);
            mpz_divexact_ui(co, n, p);
            kf_factor_recursive(fp, factors, c_start);
            kf_factor_recursive(co, factors, c_start + 31337);
            mpz_clears(fp, co, NULL);
            return;
        }
    }

    // Pollard's rho — threads race with independent c values.
    mpz_t found_factor;
    mpz_init(found_factor);
    volatile int success = 0;

#ifdef _OPENMP
#pragma omp parallel shared(found_factor, success)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int tid = 0, nt = 1;
#endif
        mpz_t local_factor;
        mpz_init(local_factor);
        // Each thread gets a disjoint range of c values:
        // thread 0: c_start+1, c_start+1+nt, c_start+1+2nt, ...
        // thread 1: c_start+2, c_start+2+nt, ...
        unsigned long c = c_start + (unsigned long)(tid + 1);
        const unsigned long limit = c_start + 200000UL * (unsigned long)nt;

        while (c <= limit)
        {
#pragma omp flush(success)
            if (success)
                break;

            // Skip c=0 and c=n-2 (degenerate cycles for Pollard's rho)
            mpz_t bad;
            mpz_init(bad);
            mpz_sub_ui(bad, n, 2);
            bool skip = (c == 0 || (mpz_fits_ulong_p(bad) &&
                                    mpz_get_ui(bad) == c));
            mpz_clear(bad);

            if (!skip && kf_pollard_rho_brent(local_factor, n, c))
            {
#pragma omp critical
                {
                    if (!success)
                    {
                        mpz_set(found_factor, local_factor);
                        success = 1;
                    }
                }
#pragma omp flush(success)
            }
            c += (unsigned long)nt;
        }
        mpz_clear(local_factor);
    } // end parallel region

    if (success)
    {
        mpz_t cofactor;
        mpz_init(cofactor);
        mpz_divexact(cofactor, n, found_factor);
        // Recursive calls use distinct c_start offsets to avoid re-using
        // the same c values that just succeeded (avoids correlated failures).
        kf_factor_recursive(found_factor, factors, c_start + 7919);
        kf_factor_recursive(cofactor, factors, c_start + 104729);
        mpz_clear(cofactor);
    }
    else
    {
        fprintf(stderr, "WARNING: Could not factor %zu-bit number\n",
                mpz_sizeinbase(n, 2));
    }
    mpz_clear(found_factor);
}

// ---------------------------------------------------------------------------
// k64 — Consolidate raw factors
// ---------------------------------------------------------------------------

void kf_factor_challenge(const mpz_t challenge, kfactor_list_t *out)
{
    // Collect raw prime factors (may have duplicates for repeated primes)
    kfactor_list_t raw;
    kfl_init(&raw);
    kf_factor_recursive(challenge, &raw, 1);

    // Consolidate: merge duplicate primes into (prime, exponent) pairs.
    // Mark merged entries by setting prime to 0.
    for (int i = 0; i < raw.count; i++)
    {
        if (mpz_cmp_ui(raw.items[i].prime, 0) == 0)
            continue;
        int exp = 1;
        for (int j = i + 1; j < raw.count; j++)
        {
            if (mpz_cmp(raw.items[i].prime, raw.items[j].prime) == 0)
            {
                exp++;
                mpz_set_ui(raw.items[j].prime, 0); // mark merged
            }
        }
        kfl_append_mpz(out, raw.items[i].prime, exp);
    }
    kfl_free(&raw);
}
// ===========================================================================
// cbrt cache — persistent (prime → cbrt_mod_M) lookup table
// ===========================================================================

void kf_cbrt_cache_init(cbrt_cache_t *cache)
{
    cache->items = NULL;
    cache->count = 0;
    cache->capacity = 0;
    cache->slots = NULL;
    cache->slot_mask = -1;
}

void kf_cbrt_cache_clear(cbrt_cache_t *cache)
{
    for (int i = 0; i < cache->count; i++)
        mpz_clear(cache->items[i].cbrt);
    free(cache->items);
    free(cache->slots);
    kf_cbrt_cache_init(cache);
}

// Fibonacci hashing: primes are odd and clustered, so mix before masking.
static inline int cbrt_slot_of(unsigned long prime, int mask)
{
    return (int)(((uint64_t)prime * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
}

// Rebuild the index at `nslots` (a power of two > count) from items[].
static void cbrt_cache_reindex(cbrt_cache_t *cache, int nslots)
{
    free(cache->slots);
    cache->slots = (int *)calloc((size_t)nslots, sizeof(int));
    cache->slot_mask = nslots - 1;
    for (int i = 0; i < cache->count; i++)
    {
        int s = cbrt_slot_of(cache->items[i].prime, cache->slot_mask);
        while (cache->slots[s])
            s = (s + 1) & cache->slot_mask;
        cache->slots[s] = i + 1;
    }
}

// ---------------------------------------------------------------------------
// Per-thread lookup counters. See kf_cbrt_stat_t in factorization.h.
// ---------------------------------------------------------------------------
static kf_cbrt_stat_t g_cbrt_stats[KF_CBRT_MAX_THREADS];
static kf_recon_prof_t g_recon_prof;

static inline kf_cbrt_stat_t *kf_cbrt_slot(void)
{
#ifdef _OPENMP
    int t = omp_get_thread_num();
    if (t < 0 || t >= KF_CBRT_MAX_THREADS)
        t = 0;
    return &g_cbrt_stats[t];
#else
    return &g_cbrt_stats[0];
#endif
}

void kf_cbrt_stats_reset(void)
{
    memset(g_cbrt_stats, 0, sizeof(g_cbrt_stats));
}

void kf_cbrt_stats_total(long *hits, long *misses, long *probes)
{
    long h = 0, m = 0, p = 0;
    for (int i = 0; i < KF_CBRT_MAX_THREADS; i++)
    {
        h += g_cbrt_stats[i].hits;
        m += g_cbrt_stats[i].misses;
        p += g_cbrt_stats[i].probes;
    }
    if (hits)
        *hits = h;
    if (misses)
        *misses = m;
    if (probes)
        *probes = p;
}

int kf_cbrt_stats_threads_used(void)
{
    int n = 0;
    for (int i = 0; i < KF_CBRT_MAX_THREADS; i++)
        if (g_cbrt_stats[i].hits || g_cbrt_stats[i].misses)
            n++;
    return n;
}

const kf_cbrt_stat_t *kf_cbrt_stats_raw(void) { return g_cbrt_stats; }

void kf_cbrt_stats_report(const char *tag)
{
    long h = 0, m = 0, p = 0;
    kf_cbrt_stats_total(&h, &m, &p);
    long total = h + m;
    if (total == 0)
    {
        printf("\n--- cbrt cache stats (%s): no lookups recorded ---\n",
               tag ? tag : "");
        return;
    }

    printf("\n--- cbrt cache reads per core  (%s) ---\n", tag ? tag : "");
    printf("  %-8s %12s %12s %12s %10s %12s\n",
           "thread", "lookups", "hits", "misses", "hit%", "probes");
    for (int i = 0; i < KF_CBRT_MAX_THREADS; i++)
    {
        long th = g_cbrt_stats[i].hits, tm = g_cbrt_stats[i].misses;
        if (th == 0 && tm == 0)
            continue;
        long tt = th + tm;
        printf("  %-8d %12ld %12ld %12ld %9.2f%% %12ld\n",
               i, tt, th, tm, 100.0 * (double)th / (double)tt,
               g_cbrt_stats[i].probes);
    }
    printf("  %-8s %12ld %12ld %12ld %9.2f%% %12ld\n",
           "TOTAL", total, h, m, 100.0 * (double)h / (double)total, p);
    printf("  cores used: %d   mean probes/lookup: %.1f\n",
           kf_cbrt_stats_threads_used(), (double)p / (double)total);

    // Machine-parseable: one line per thread, then the aggregate.
    for (int i = 0; i < KF_CBRT_MAX_THREADS; i++)
    {
        long th = g_cbrt_stats[i].hits, tm = g_cbrt_stats[i].misses;
        if (th == 0 && tm == 0)
            continue;
        printf("CBRTSTAT|%s|%d|%ld|%ld|%ld\n",
               tag ? tag : "", i, th, tm, g_cbrt_stats[i].probes);
    }
    printf("CBRTSTAT|%s|ALL|%ld|%ld|%ld\n", tag ? tag : "", h, m, p);
    fflush(stdout);
}

void kf_recon_prof_reset(void)
{
    g_recon_prof.parallel_ms = 0.0;
    g_recon_prof.serial_ms = 0.0;
    g_recon_prof.factors = 0;
    g_recon_prof.calls = 0;
}

const kf_recon_prof_t *kf_recon_prof_get(void) { return &g_recon_prof; }

int kf_cbrt_cache_find(const cbrt_cache_t *cache, unsigned long prime)
{
    // One counter update per LOOKUP, not per comparison, so the instrumentation
    // does not perturb the scan it measures.
    kf_cbrt_stat_t *st = kf_cbrt_slot();
    if (cache->slot_mask < 0)
    {
        st->misses++;
        return -1;
    }
    int s = cbrt_slot_of(prime, cache->slot_mask);
    long probes = 1;
    for (int v; (v = cache->slots[s]) != 0; s = (s + 1) & cache->slot_mask, probes++)
        if (cache->items[v - 1].prime == prime)
        {
            st->probes += probes;
            st->hits++;
            return v - 1;
        }
    st->probes += probes;
    st->misses++;
    return -1;
}

void kf_cbrt_cache_append(cbrt_cache_t *cache, unsigned long prime,
                          const mpz_t val)
{
    if (cache->count == cache->capacity)
    {
        int nc = (cache->capacity == 0) ? 64 : cache->capacity * 2;
        cache->items = (cbrt_entry_t *)realloc(
            cache->items, (size_t)nc * sizeof(cbrt_entry_t));
        cache->capacity = nc;
    }
    cache->items[cache->count].prime = prime;
    mpz_init_set(cache->items[cache->count].cbrt, val);
    cache->count++;

    // Keep the index at most half full; growing rebuilds it from items[].
    if (2 * cache->count > cache->slot_mask + 1)
    {
        int ns = 128;
        while (ns < 2 * cache->count)
            ns *= 2;
        cbrt_cache_reindex(cache, ns);
    }
    else
    {
        int s = cbrt_slot_of(prime, cache->slot_mask);
        while (cache->slots[s])
            s = (s + 1) & cache->slot_mask;
        cache->slots[s] = cache->count;
    }
}

bool kf_cbrt_cache_load(cbrt_cache_t *cache, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return false;

    int stored = 0;
    if (fscanf(f, "%d", &stored) != 1 || stored < 0)
    {
        fclose(f);
        return false;
    }

    kf_cbrt_cache_clear(cache);
    kf_cbrt_cache_init(cache);

    for (int i = 0; i < stored; i++)
    {
        unsigned long p = 0;
        if (fscanf(f, "%lu", &p) != 1)
        {
            fclose(f);
            kf_cbrt_cache_clear(cache);
            kf_cbrt_cache_init(cache);
            return false;
        }
        mpz_t v;
        mpz_init(v);
        if (mpz_inp_str(v, f, 10) == 0)
        {
            mpz_clear(v);
            fclose(f);
            kf_cbrt_cache_clear(cache);
            kf_cbrt_cache_init(cache);
            return false;
        }
        kf_cbrt_cache_append(cache, p, v);
        mpz_clear(v);
    }

    fclose(f);
    printf("Loaded %d cached cbrt entries from %s\n", cache->count, path);
    return true;
}

bool kf_cbrt_cache_save(const cbrt_cache_t *cache, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f)
    {
        fprintf(stderr, "Cannot write cbrt cache to %s\n", path);
        return false;
    }
    fprintf(f, "%d\n", cache->count);
    for (int i = 0; i < cache->count; i++)
    {
        fprintf(f, "%lu ", cache->items[i].prime);
        mpz_out_str(f, 10, cache->items[i].cbrt);
        fputc('\n', f);
    }
    fclose(f);
    printf("Saved %d cached cbrt entries to %s\n", cache->count, path);
    return true;
}

// ===========================================================================
// k32 — reconstruction: result = ∏ cbrt(p_i)^e_i  mod M
// ===========================================================================

void kf_compute_cbrt_from_factors(mpz_t result,
                                  const kfactor_list_t *found,
                                  const mpz_t modulo,
                                  cbrt_cache_t *cache,
                                  bool *cache_dirty,
                                  unsigned long exponent)
{
    // Precompute e_inv = exponent^{-1} mod (M-1) once for this modulo.
    mpz_t phi, e_inv, e_base;
    mpz_inits(phi, e_inv, e_base, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);
    mpz_invert(e_inv, e_base, phi);
    mpz_clears(phi, e_base, NULL);

    mpz_set_ui(result, 1);

    for (int i = 0; i < found->count; i++)
    {
        unsigned long p = mpz_get_ui(found->items[i].prime);
        int idx = kf_cbrt_cache_find(cache, p);

        mpz_t cbrt_p;
        mpz_init(cbrt_p);

        if (idx >= 0)
        {
            mpz_set(cbrt_p, cache->items[idx].cbrt);
        }
        else
        {
            // Cache miss: compute p^e_inv mod M and store for next time.
            mpz_t p_mpz;
            mpz_init_set_ui(p_mpz, p);
            mpz_powm(cbrt_p, p_mpz, e_inv, modulo);
            mpz_clear(p_mpz);
            kf_cbrt_cache_append(cache, p, cbrt_p);
            if (cache_dirty)
                *cache_dirty = true;
        }

        if (found->items[i].exponent > 1)
        {
            mpz_t powered;
            mpz_init(powered);
            mpz_powm_ui(powered, cbrt_p,
                        (unsigned long)found->items[i].exponent, modulo);
            mpz_mul(result, result, powered);
            mpz_clear(powered);
        }
        else
        {
            mpz_mul(result, result, cbrt_p);
        }
        mpz_mod(result, result, modulo);
        mpz_clear(cbrt_p);
    }

    mpz_clear(e_inv);
}

// ---------------------------------------------------------------------------
// Number of CPU cores actually available to THIS process. Uses the scheduler
// affinity mask (sched_getaffinity), so under SLURM it returns exactly the cores
// the job was granted (its cpuset/cgroup), not the whole node — no need to set
// OMP_NUM_THREADS or --cpus-per-task by hand. Falls back to the OpenMP processor
// count, then to 1. Cached after the first call.
// ---------------------------------------------------------------------------
static int ksc_available_cpus(void)
{
    static int cached = 0;
    if (cached > 0)
        return cached;
    int n = 0;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
        n = CPU_COUNT(&set);
#endif
#ifdef _OPENMP
    if (n <= 0)
        n = omp_get_num_procs();
#endif
    if (n <= 0)
        n = 1;
    cached = n;
    return n;
}

// ---------------------------------------------------------------------------
// kf_product_mod_tree — result = vals[0] * ... * vals[n-1] mod M, in parallel
//
// A serial accumulator is a chain of n-1 dependent |M|-bit mulmods. With a
// warm cbrt cache that chain IS reconstruction (~800 links for k32 at 24576
// bits). Multiplication mod M is associative and commutative, so each thread
// multiplies one contiguous chunk, then the per-thread partials are combined
// pairwise: n/T + ceil(log2 T) dependent mulmods instead of n-1, with the
// same total work. The result is identical to the serial product.
//
// Each thread gets at least two values, and small products stay serial: below
// KF_TREE_MIN_WORK (n * limbs^2, ~1 ms of serial mulmods) threads cannot pay
// for themselves. Waking a team costs microseconds on a quiet machine, but on a
// busy one a thread descheduled at a barrier stalls the region for a scheduler
// time slice (~10 ms observed) — 1000x the 15 us a 1024-bit, 33-term product
// needs. Thread count is read inside the region because the runtime may grant
// fewer than asked.
// ---------------------------------------------------------------------------
#define KF_TREE_MIN_WORK 2.0e6

bool kf_parallel_worth(double mulmods, const mpz_t modulo)
{
    double limbs = (double)mpz_size(modulo);
    return mulmods * limbs * limbs >= KF_TREE_MIN_WORK;
}

void kf_product_mod_tree(mpz_t result, mpz_t *vals, int n,
                         const mpz_t modulo, int nthreads)
{
    int want = nthreads < n / 2 ? nthreads : n / 2;
    if (!kf_parallel_worth((double)n, modulo))
        want = 1;
#ifdef _OPENMP
    if (want >= 2)
    {
        mpz_t *part = (mpz_t *)malloc((size_t)want * sizeof(mpz_t));
#pragma omp parallel num_threads(want)
        {
            int T = omp_get_num_threads();
            int t = omp_get_thread_num();
            int lo = (int)((long)n * t / T);
            int hi = (int)((long)n * (t + 1) / T);
            mpz_init_set(part[t], vals[lo]);
            for (int i = lo + 1; i < hi; i++)
            {
                mpz_mul(part[t], part[t], vals[i]);
                mpz_mod(part[t], part[t], modulo);
            }
            for (int s = 1; s < T; s <<= 1)
            {
#pragma omp barrier
                if (t % (2 * s) == 0 && t + s < T)
                {
                    mpz_mul(part[t], part[t], part[t + s]);
                    mpz_mod(part[t], part[t], modulo);
                }
            }
#pragma omp barrier
#pragma omp single
            mpz_set(result, part[0]);
            mpz_clear(part[t]);
        }
        free(part);
        return;
    }
#endif
    (void)want;
    mpz_set_ui(result, 1);
    for (int i = 0; i < n; i++)
    {
        mpz_mul(result, result, vals[i]);
        mpz_mod(result, result, modulo);
    }
}

static kf_product_fn g_product_backend = kf_product_mod_tree;

void kf_set_product_backend(kf_product_fn fn)
{
    g_product_backend = fn ? fn : kf_product_mod_tree;
}

// ---------------------------------------------------------------------------
// Parallel (OpenMP) variant of kf_compute_cbrt_from_factors, for single-process
// solvers (the IRIS GPU port) where reconstruction is NOT distributed across MPI
// ranks as it is on AION. At large moduli a challenge can have hundreds of
// distinct prime factors, each needing a full |M|-bit modular exponentiation for
// its cube root; serially on one core this dominates the runtime (e.g. ~20 min
// for a 24576-bit k35 challenge). Here the per-factor cube roots (and their
// e_i-th powers) are computed concurrently, then combined by
// kf_product_mod_tree.
//
// Output is bit-identical to the serial version: multiplication mod M is
// associative and commutative, so parallelism changes only the order of the
// multiplications, not the product.
//
// Cache lookups are O(1) (hashed), so they run serially first; threads are
// only woken for real work — a cache miss is a full |M|-bit exponentiation —
// for the same reason kf_product_mod_tree gates its tree. The cache is never
// written inside the parallel region: newly computed roots are appended in the
// serial phase afterward. Without -fopenmp the pragma is ignored and the
// function runs serially (still correct).
void kf_compute_cbrt_from_factors_omp(mpz_t result,
                                      const kfactor_list_t *found,
                                      const mpz_t modulo,
                                      cbrt_cache_t *cache,
                                      bool *cache_dirty,
                                      unsigned long exponent)
{
    mpz_t phi, e_inv, e_base;
    mpz_inits(phi, e_inv, e_base, NULL);
    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);
    mpz_invert(e_inv, e_base, phi);
    mpz_clears(phi, e_base, NULL);

    int n = found->count;
    if (n <= 0)
    {
        mpz_set_ui(result, 1);
        mpz_clear(e_inv);
        return;
    }

    // Per-factor base cube root cbrt(p_i); miss[i] flags a cache miss.
    // parts[i] = cbrt(p_i)^e_i, the term that goes into the product.
    mpz_t *cbrts = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    mpz_t *parts = (mpz_t *)malloc((size_t)n * sizeof(mpz_t));
    char *miss = (char *)calloc((size_t)n, 1);

    // Use exactly the cores this process was granted; never more threads than
    // there are factors to root.
    int nthreads = ksc_available_cpus();
    if (nthreads > n)
        nthreads = n;

    auto t_p0 = std::chrono::high_resolution_clock::now();

    // Serial: look every factor up; count what is left to compute.
    int misses = 0, powers = 0;
    for (int i = 0; i < n; i++)
    {
        mpz_init(cbrts[i]);
        int idx = kf_cbrt_cache_find(cache, mpz_get_ui(found->items[i].prime));
        if (idx >= 0)
        {
            mpz_set(cbrts[i], cache->items[idx].cbrt);
        }
        else
        {
            miss[i] = 1;
            misses++;
        }
        powers += (found->items[i].exponent > 1);
    }
    // Two or more exponentiations always pay for threads; small powers
    // (a few mulmods each) only in bulk.
    bool par = misses >= 2 || kf_parallel_worth(4.0 * powers, modulo);

// Parallel when worth it: cube roots for the misses, then each term's power.
#pragma omp parallel for schedule(dynamic) num_threads(nthreads) if (par)
    for (int i = 0; i < n; i++)
    {
        if (miss[i])
        {
            mpz_t p_mpz;
            mpz_init_set_ui(p_mpz, mpz_get_ui(found->items[i].prime));
            mpz_powm(cbrts[i], p_mpz, e_inv, modulo); // p^(e_inv) mod M
            mpz_clear(p_mpz);
        }
        if (found->items[i].exponent > 1)
        {
            mpz_init(parts[i]);
            mpz_powm_ui(parts[i], cbrts[i],
                        (unsigned long)found->items[i].exponent, modulo);
        }
        else
        {
            mpz_init_set(parts[i], cbrts[i]);
        }
    }

    auto t_p1 = std::chrono::high_resolution_clock::now();

    // Combine: product tree over the terms (host threads, or the device tree
    // when a CUDA solver installed it with kf_set_product_backend).
    g_product_backend(result, parts, n, modulo, nthreads);

    // Serial: append newly computed roots to the cache, in index order.
    for (int i = 0; i < n; i++)
    {
        mpz_clear(parts[i]);
        if (miss[i])
        {
            unsigned long p = mpz_get_ui(found->items[i].prime);
            kf_cbrt_cache_append(cache, p, cbrts[i]);
            if (cache_dirty)
                *cache_dirty = true;
        }
        mpz_clear(cbrts[i]);
    }

    auto t_p2 = std::chrono::high_resolution_clock::now();
    g_recon_prof.parallel_ms +=
        std::chrono::duration<double, std::milli>(t_p1 - t_p0).count();
    g_recon_prof.serial_ms +=
        std::chrono::duration<double, std::milli>(t_p2 - t_p1).count();
    g_recon_prof.factors += n;
    g_recon_prof.calls++;

    free(cbrts);
    free(parts);
    free(miss);
    mpz_clear(e_inv);
}

// ===========================================================================
// blinding_pair_t table — precomputation phase
// ===========================================================================

blinding_pair_t *kf_blinding_table_alloc(int count, const mpz_t modulo,
                                         unsigned long exponent)
{
    blinding_pair_t *table =
        (blinding_pair_t *)malloc((size_t)count * sizeof(blinding_pair_t));
    if (!table)
        return NULL;

    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    gmp_randseed_ui(rng, (unsigned long)time(NULL) ^ (unsigned long)(uintptr_t)table);

    mpz_t r;
    mpz_init(r);

    for (int i = 0; i < count; i++)
    {
        mpz_inits(table[i].r, table[i].r_blind, table[i].r_inv, NULL);

        do
        {
            mpz_urandomm(r, rng, modulo);
        } while (mpz_cmp_ui(r, 2) < 0);

        mpz_set(table[i].r, r);
        mpz_powm_ui(table[i].r_blind, r, exponent, modulo); // r^e mod M
        mpz_invert(table[i].r_inv, r, modulo);              // r^{-1} mod M
    }

    mpz_clear(r);
    gmp_randclear(rng);
    return table;
}

void kf_blinding_table_free(blinding_pair_t *table, int count)
{
    if (!table)
        return;
    for (int i = 0; i < count; i++)
        mpz_clears(table[i].r, table[i].r_blind, table[i].r_inv, NULL);
    free(table);
}

// ===========================================================================
// k16 — Precompute and reconstruct
// ===========================================================================

void kf_precompute_cbrt_table_k16(mpz_t *cbrt_table, const unsigned long *primes,
                                  int prime_count, const mpz_t modulo,
                                  unsigned long exponent)
{
    mpz_t phi, e_inv, e_base;
    mpz_inits(phi, e_inv, e_base, NULL);

    mpz_sub_ui(phi, modulo, 1);
    mpz_set_ui(e_base, exponent);

    // e_inv = exponent^{-1} mod (M-1)
    if (!mpz_invert(e_inv, e_base, phi))
    {
        fprintf(stderr, "[ERROR] Modulo not suitable for %lu-th roots (gcd(%lu, M-1) != 1)\n",
                exponent, exponent);
        exit(1);
    }

    printf("[SETUP] Precomputing %d %lu-th roots...\n", prime_count, exponent);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < prime_count; i++)
    {
        mpz_t p;
        mpz_init_set_ui(p, primes[i]);
        mpz_init(cbrt_table[i]);
        mpz_powm(cbrt_table[i], p, e_inv, modulo);
        mpz_clear(p);
    }

    mpz_clears(phi, e_inv, e_base, NULL);
}

void kf_save_cbrt_table_k16(const mpz_t *cbrt_table, int prime_count, const char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return;
    fwrite(&prime_count, sizeof(int), 1, f);
    for (int i = 0; i < prime_count; i++)
    {
        mpz_out_raw(f, cbrt_table[i]);
    }
    fclose(f);
}

bool kf_load_cbrt_table_k16(mpz_t *cbrt_table, int prime_count, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        return false;
    int stored_count;
    if (fread(&stored_count, sizeof(int), 1, f) != 1 || stored_count != prime_count)
    {
        fclose(f);
        return false;
    }
    for (int i = 0; i < prime_count; i++)
    {
        mpz_init(cbrt_table[i]);
        if (mpz_inp_raw(cbrt_table[i], f) == 0)
        {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

/**
 * Reconstructs the final cube root using the homomorphic property:
 * result = prod(cbrt_table[i]^exponents[i]) mod M
 */
void kf_compute_cube_root_from_factors_k16(mpz_t result, const int *exponents,
                                           const mpz_t *cbrt_table, int prime_count,
                                           const mpz_t modulo)
{
    mpz_set_ui(result, 1);
    for (int i = 0; i < prime_count; i++)
    {
        if (exponents[i] == 0)
            continue;

        if (exponents[i] == 1)
        {
            mpz_mul(result, result, cbrt_table[i]);
        }
        else
        {
            mpz_t tmp;
            mpz_init(tmp);
            mpz_powm_ui(tmp, cbrt_table[i], (unsigned long)exponents[i], modulo);
            mpz_mul(result, result, tmp);
            mpz_clear(tmp);
        }
        mpz_mod(result, result, modulo);
    }
}
// ---------------------------------------------------------------------------
// kf_try_factor_b64smooth — factor n via Pollard's rho, check 2^64-smooth
// ---------------------------------------------------------------------------

bool kf_try_factor_b64smooth(const mpz_t n,
                             kfactor_list_t *factors,
                             unsigned long c_start)
{
    if (mpz_cmp_ui(n, 1) == 0)
        return true;

    // If n fits in 64 bits, it is automatically 2^64-smooth.
    // Use kf_factor_recursive to fully factor it.
    if (mpz_sizeinbase(n, 2) <= 64)
    {
        kf_factor_recursive(n, factors, c_start);
        return true;
    }

    // n > 2^64: test if it is prime (would mean not 2^64-smooth)
    if (kf_is_prime(n))
        return false; // prime factor > 2^64 — not smooth

    // n is composite and > 2^64 — find a factor via Pollard's rho
    mpz_t found;
    mpz_init(found);
    volatile int success = 0;

#ifdef _OPENMP
#pragma omp parallel shared(found, success)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
#else
    {
        int tid = 0, nt = 1;
#endif
        mpz_t local_f;
        mpz_init(local_f);
        unsigned long c = c_start + (unsigned long)(tid + 1);
        const unsigned long limit = c_start + 500000UL * (unsigned long)nt;

        while (c <= limit)
        {
#pragma omp flush(success)
            if (success)
                break;

            if (kf_pollard_rho_brent(local_f, n, c))
            {
#pragma omp critical
                if (!success)
                {
                    mpz_set(found, local_f);
                    success = 1;
                }
#pragma omp flush(success)
            }
            c += (unsigned long)nt;
        }
        mpz_clear(local_f);
    }

    bool ok = false;
    if (success)
    {
        // Check both halves for 2^64-smoothness recursively
        mpz_t cofactor;
        mpz_init(cofactor);
        mpz_divexact(cofactor, n, found);

        kfactor_list_t left, right;
        kfl_init(&left);
        kfl_init(&right);

        bool left_ok = kf_try_factor_b64smooth(found, &left, c_start + 7919);
        bool right_ok = kf_try_factor_b64smooth(cofactor, &right, c_start + 104729);
        ok = left_ok && right_ok;

        if (ok)
        {
            for (int i = 0; i < left.count; i++)
                kfl_append_mpz(factors, left.items[i].prime, left.items[i].exponent);
            for (int i = 0; i < right.count; i++)
                kfl_append_mpz(factors, right.items[i].prime, right.items[i].exponent);
        }
        kfl_free(&left);
        kfl_free(&right);
        mpz_clear(cofactor);
    }
    mpz_clear(found);
    return ok;
}

// =============================================================================
// k35 upper-range prime table  — uint64_t primes in (2^32, 2^35]
// =============================================================================

// ---------------------------------------------------------------------------
// kf_generate_and_save_prime_table_u64
// ---------------------------------------------------------------------------
bool kf_generate_and_save_prime_table_u64(
    const char *path,
    uint64_t range_lo, uint64_t range_hi,
    int sp_count, const unsigned *sp,
    const char *log_prefix)
{
    // Use the same 512 KiB segment as the sieve
    const uint64_t SEG = (uint64_t)KF_SEG_SIZE;

    FILE *f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "%sERROR: cannot create %s: %s\n",
                log_prefix, path, strerror(errno));
        return false;
    }

    // Write placeholder header (magic + count = 0)
    uint64_t header[2] = {K35U_PRIME_TABLE_MAGIC, 0};
    if (fwrite(header, sizeof(uint64_t), 2, f) != 2)
    {
        fclose(f);
        return false;
    }

    uint64_t total = 0;
    // Write primes in batches to reduce fwrite calls
    uint64_t batch[KF_WRITE_BATCH];
    int bsz = 0;

    // Temporary sieve buffer — one byte per candidate offset
    unsigned char *sieve = (unsigned char *)malloc(SEG);
    if (!sieve)
    {
        fclose(f);
        return false;
    }

    for (uint64_t seg_lo = range_lo; seg_lo <= range_hi; seg_lo += SEG)
    {
        uint64_t seg_hi = seg_lo + SEG - 1;
        if (seg_hi > range_hi)
            seg_hi = range_hi;
        uint64_t seg_sz = seg_hi - seg_lo + 1;

        // Mark all as prime (0 = prime candidate)
        memset(sieve, 0, seg_sz);

        // Sieve with each small prime
        for (int i = 0; i < sp_count; i++)
        {
            uint64_t p = (uint64_t)sp[i];
            // First multiple of p in segment
            uint64_t start = ((seg_lo + p - 1) / p) * p;
            if (start < p * 2)
                start = p * 2; // don't mark p itself
            for (uint64_t j = start; j <= seg_hi; j += p)
                sieve[j - seg_lo] = 1;
        }

        // Collect unmarked candidates
        for (uint64_t j = 0; j < seg_sz; j++)
        {
            if (!sieve[j])
            {
                batch[bsz++] = seg_lo + j;
                total++;
                if (bsz == KF_WRITE_BATCH)
                {
                    fwrite(batch, sizeof(uint64_t), KF_WRITE_BATCH, f);
                    bsz = 0;
                }
            }
        }

        // Progress: every 256 segments (~128 MB range)
        if (((seg_lo - range_lo) / SEG) % 256 == 0)
        {
            double pct = 100.0 * (double)(seg_lo - range_lo) / (double)(range_hi - range_lo);
            fprintf(stderr, "%s  %.1f%%  (%" PRIu64 " primes so far)\r",
                    log_prefix, pct, total);
            fflush(stderr);
        }
    }

    // Flush remaining batch
    if (bsz > 0)
        fwrite(batch, sizeof(uint64_t), bsz, f);

    // Update count in header
    fseeko(f, (off_t)sizeof(uint64_t), SEEK_SET);
    fwrite(&total, sizeof(uint64_t), 1, f);
    fclose(f);
    free(sieve);

    fprintf(stderr, "%s  100.0%%  %" PRIu64 " primes written to %s\n",
            log_prefix, total, path);
    return true;
}

// ---------------------------------------------------------------------------
// kf_load_prime_slice_u64
//
// Binary-searches the file for the first prime >= lo, then reads until > hi.
// Returns malloc'd array.  Caller must free().
// ---------------------------------------------------------------------------
uint64_t *kf_load_prime_slice_u64(int64_t *count_out, const char *path,
                                  uint64_t lo, uint64_t hi)
{
    *count_out = 0;

    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "[k35u] Cannot open %s\n", path);
        return nullptr;
    }

    uint64_t magic, total;
    if (fread(&magic, 8, 1, f) != 1 || fread(&total, 8, 1, f) != 1 || magic != K35U_PRIME_TABLE_MAGIC)
    {
        fprintf(stderr, "[k35u] Bad header in %s\n", path);
        fclose(f);
        return nullptr;
    }

    if (total == 0)
    {
        fclose(f);
        return nullptr;
    }

    // Binary search: find index of first prime >= lo
    const off_t DATA_OFF = 2 * (off_t)sizeof(uint64_t);
    int64_t left = 0, right = (int64_t)total - 1, start_idx = (int64_t)total;
    while (left <= right)
    {
        int64_t mid = left + (right - left) / 2;
        uint64_t val;
        fseeko(f, DATA_OFF + (off_t)mid * 8, SEEK_SET);
        if (fread(&val, 8, 1, f) != 1)
            break;
        if (val < lo)
        {
            left = mid + 1;
        }
        else
        {
            start_idx = mid;
            right = mid - 1;
        }
    }

    if (start_idx == (int64_t)total)
    {
        fclose(f);
        return nullptr;
    }

    // Count entries in [lo, hi]
    fseeko(f, DATA_OFF + (off_t)start_idx * 8, SEEK_SET);
    int64_t count = 0;
    {
        uint64_t buf[4096];
        int64_t idx = start_idx;
        while (idx < (int64_t)total)
        {
            size_t rd = fread(buf, 8, 4096, f);
            if (rd == 0)
                break;
            for (size_t k = 0; k < rd; k++)
            {
                if (buf[k] > hi)
                {
                    idx = (int64_t)total;
                    break;
                }
                count++;
                idx++;
            }
        }
    }

    if (count == 0)
    {
        fclose(f);
        return nullptr;
    }

    // Load the slice
    uint64_t *slice = (uint64_t *)malloc((size_t)count * sizeof(uint64_t));
    if (!slice)
    {
        fclose(f);
        return nullptr;
    }
    fseeko(f, DATA_OFF + (off_t)start_idx * 8, SEEK_SET);
    if ((int64_t)fread(slice, 8, (size_t)count, f) != count)
    {
        free(slice);
        fclose(f);
        return nullptr;
    }
    fclose(f);

    *count_out = count;
    return slice;
}

// ---------------------------------------------------------------------------
// kf_trial_divide_table_u64
//
// Same logic as kf_trial_divide_table but for uint64_t primes.
// On 64-bit Linux, unsigned long == uint64_t so mpz_divisible_ui_p works.
// ---------------------------------------------------------------------------
void kf_trial_divide_table_u64(
    const mpz_t n,
    const uint64_t *primes, int64_t prime_count,
    kfactor_list_t *found,
    int64_t *thread_tested, int *thread_found, int nthreads_out)
{
    if (!primes || prime_count <= 0)
        return;

    // Use all available OMP threads unless caller specifies count
    int nt = nthreads_out > 0 ? nthreads_out : 1;
#ifdef _OPENMP
    if (nthreads_out <= 0)
        nt = omp_get_max_threads();
#endif

#pragma omp parallel num_threads(nt)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        kfactor_list_t local;
        kfl_init(&local);
        int64_t tested = 0;
        int found_cnt = 0;

        // Thread-private shrinking copy of n (Bug 13 fix).  Each thread strips
        // out only the primes it personally finds, so later divisibility tests
        // run against a smaller number (cheaper mpz_divisible_ui_p).  This is
        // lock-free and cannot miss a factor: a prime p is tested by exactly
        // one thread (static partition over distinct primes), and at the moment
        // that thread tests p, local_n still contains p at full multiplicity
        // (only OTHER primes this thread already found were removed).  So
        // p | n  <=>  p | local_n  at the test point.
        mpz_t local_n;
        mpz_init_set(local_n, n);

#pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < prime_count; i++)
        {
            unsigned long p = (unsigned long)primes[i];
            tested++;
            if (mpz_divisible_ui_p(local_n, p))
            {
                // Count exponent, stripping p from this thread's private copy.
                int e = 0;
                while (mpz_divisible_ui_p(local_n, p))
                {
                    mpz_divexact_ui(local_n, local_n, p);
                    e++;
                }
                kfl_append_ul(&local, p, e);
                found_cnt++;
            }
        }
        mpz_clear(local_n);

        // Merge local into shared found
#pragma omp critical
        {
            for (int i = 0; i < local.count; i++)
                kfl_append_ul(found,
                              (unsigned long)mpz_get_ui(local.items[i].prime),
                              local.items[i].exponent);
            if (thread_tested && tid < nthreads_out)
                thread_tested[tid] += tested;
            if (thread_found && tid < nthreads_out)
                thread_found[tid] += found_cnt;
        }
        kfl_free(&local);
    }
}
