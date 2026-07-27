// delta_table_test.cpp — correctness gate for the delta/gap-encoded prime table.
//
// Verifies the new factorization.cpp delta path against the existing flat path:
//   1. round-trip     — decode(delta) is bit-identical to the source primes
//   2. slice boundary — a delta sub-range serves exactly the primes in it
//                        (two disjoint sub-ranges partition with no overlap/miss)
//   3. equivalence    — trial-dividing a real smooth n via the delta slice
//                        yields the identical factorization as the flat table
// for BOTH widths: u32 (k32 range) and u64 (k35 upper range).
//
// MPI-free — builds with g++ (no mpicxx). Build via the Makefile target
//   make delta_table_test        (uses g++, not mpicxx)
// then run ./delta_table_test  (prints PASS/FAIL; exit 0 on PASS).

#include "factorization.h"
#include <gmp.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>

static int g_fail = 0;
#define CHECK(cond, name)                                  \
    do                                                     \
    {                                                      \
        if (cond)                                          \
            printf("  [PASS] %s\n", name);                 \
        else                                               \
        {                                                  \
            printf("  [FAIL] %s\n", name);                 \
            g_fail = 1;                                    \
        }                                                  \
    } while (0)

// --- prime generation (self-contained) -------------------------------------

static std::vector<uint32_t> small_primes(uint32_t L)
{
    std::vector<char> s(L + 1, 1);
    s[0] = s[1] = 0;
    for (uint32_t i = 2; (uint64_t)i * i <= L; i++)
        if (s[i])
            for (uint32_t j = i * i; j <= L; j += i)
                s[j] = 0;
    std::vector<uint32_t> out;
    for (uint32_t i = 2; i <= L; i++)
        if (s[i])
            out.push_back(i);
    return out;
}

template <typename T>
static std::vector<T> sieve_range(uint64_t lo, uint64_t hi,
                                  const std::vector<uint32_t> &sp)
{
    std::vector<T> out;
    const uint64_t SEG = 1 << 19;
    std::vector<char> seg(SEG);
    for (uint64_t s = lo; s <= hi; s += SEG)
    {
        uint64_t e = std::min(s + SEG - 1, hi);
        size_t len = (size_t)(e - s + 1);
        std::fill(seg.begin(), seg.begin() + len, 1);
        for (uint32_t p : sp)
        {
            if ((uint64_t)p * p > e)
                break;
            uint64_t start = ((s + p - 1) / p) * p;
            if (start == p)
                start += p;
            for (uint64_t j = start; j <= e; j += p)
                seg[j - s] = 0;
        }
        for (uint64_t c = s; c <= e; c++)
            if (seg[c - s])
                out.push_back((T)c);
    }
    return out;
}

// --- flat-file writers (match factorization.cpp on-disk formats) ------------

static void write_flat_u32(const char *path, const std::vector<uint32_t> &p)
{
    FILE *f = fopen(path, "wb");
    uint64_t magic = KF_PRIME_TABLE_MAGIC, count = p.size();
    fwrite(&magic, 8, 1, f);
    fwrite(&count, 8, 1, f);
    fwrite(p.data(), 4, p.size(), f);
    fclose(f);
}

static void write_flat_u64(const char *path, const std::vector<uint64_t> &p)
{
    FILE *f = fopen(path, "wb");
    uint64_t magic = K35U_PRIME_TABLE_MAGIC, count = p.size();
    fwrite(&magic, 8, 1, f);
    fwrite(&count, 8, 1, f);
    fwrite(p.data(), 8, p.size(), f);
    fclose(f);
}

// --- decode a full delta slice back to a prime vector (round-trip check) ----

static std::vector<uint64_t> decode_all(const kf_delta_slice_t &s)
{
    std::vector<uint64_t> out;
    size_t full = (size_t)s.width + (size_t)(s.block_size - 1);
    for (uint32_t j = 0; j < s.n_blocks; j++)
    {
        uint64_t g = s.first_block_index + j;
        uint64_t base = g * s.block_size;
        uint32_t len = (base + s.block_size <= s.total_count)
                           ? s.block_size
                           : (uint32_t)(s.total_count - base);
        const uint8_t *r = s.bytes + (size_t)j * full;
        uint64_t cur;
        if (s.width == 8) { memcpy(&cur, r, 8); r += 8; }
        else { uint32_t c32; memcpy(&c32, r, 4); cur = c32; r += 4; }
        out.push_back(cur);
        for (uint32_t k = 1; k < len; k++) { cur += (uint64_t)(*r++) << 1; out.push_back(cur); }
    }
    return out;
}

// --- collect a kfactor_list into a (prime -> exponent) map ------------------

static std::map<uint64_t, int> to_map(const kfactor_list_t &fl)
{
    std::map<uint64_t, int> m;
    for (int i = 0; i < fl.count; i++)
        m[mpz_get_ui(fl.items[i].prime)] += fl.items[i].exponent;
    return m;
}

// Build a smooth n = product of the given primes (with multiplicity) into n,
// and return the truth (prime -> exponent) map.
static std::map<uint64_t, int> build_smooth(mpz_t n, const std::vector<uint64_t> &fac)
{
    std::map<uint64_t, int> truth;
    mpz_set_ui(n, 1);
    for (uint64_t p : fac)
    {
        mpz_mul_ui(n, n, (unsigned long)p);
        truth[p]++;
    }
    return truth;
}

int main()
{
    printf("=== delta_table_test ===\n");
    const char *FLAT32 = "delta_test_u32.bin", *DLT32 = "delta_test_u32.dlt";
    const char *FLAT64 = "delta_test_u64.bin", *DLT64 = "delta_test_u64.dlt";

    // =====================================================================
    // u32 (k32 range) — [65537, 2^22]
    // =====================================================================
    printf("\n-- u32 (k32-style) --\n");
    {
        auto sp = small_primes(70000);
        uint64_t lo = 65537, hi = (1u << 22);
        auto primes = sieve_range<uint32_t>(lo, hi, sp);
        printf("  generated %zu primes in [%llu, %llu]\n", primes.size(),
               (unsigned long long)lo, (unsigned long long)hi);
        std::vector<uint32_t> pv = primes;
        write_flat_u32(FLAT32, pv);
        CHECK(kf_convert_prime_table_to_delta(FLAT32, DLT32, "  "), "convert u32 flat -> delta");

        // (1) round-trip over the whole table
        kf_delta_slice_t full;
        CHECK(kf_load_delta_slice(&full, DLT32, 0, UINT64_MAX), "load full u32 delta slice");
        auto dec = decode_all(full);
        bool rt = (dec.size() == pv.size());
        for (size_t i = 0; rt && i < pv.size(); i++)
            rt = (dec[i] == pv[i]);
        CHECK(rt, "round-trip decode == source primes");
        kf_delta_slice_free(&full);

        // (2) slice boundary: two disjoint sub-ranges partition the primes.
        uint64_t mid = (lo + hi) / 2;
        kf_delta_slice_t sa, sb;
        kf_load_delta_slice(&sa, DLT32, lo, mid);
        kf_load_delta_slice(&sb, DLT32, mid + 1, hi);
        auto da = decode_all(sa), db = decode_all(sb);
        // filter each decoded slice to its declared range (over-read trimming)
        std::vector<uint64_t> ua, ub;
        for (uint64_t x : da) if (x >= lo && x <= mid) ua.push_back(x);
        for (uint64_t x : db) if (x >= mid + 1 && x <= hi) ub.push_back(x);
        std::vector<uint64_t> uni = ua;
        uni.insert(uni.end(), ub.begin(), ub.end());
        std::sort(uni.begin(), uni.end());
        bool part = (uni.size() == pv.size());
        for (size_t i = 0; part && i < pv.size(); i++)
            part = (uni[i] == pv[i]);
        CHECK(part, "two sub-ranges partition exactly (no overlap / no gap)");
        kf_delta_slice_free(&sa);
        kf_delta_slice_free(&sb);

        // (3) trial-division equivalence on a real smooth n.
        std::vector<uint64_t> fac;
        for (int i = 0; i < 30; i++)
        {
            uint64_t p = primes[(size_t)((uint64_t)rand() * primes.size() / ((uint64_t)RAND_MAX + 1))];
            fac.push_back(p);
            if (i % 3 == 0) fac.push_back(p); // some multiplicities
        }
        mpz_t n;
        mpz_init(n);
        auto truth = build_smooth(n, fac);

        // flat
        uint64_t sc;
        uint32_t *flatslice = kf_load_prime_slice(&sc, FLAT32, (uint32_t)lo, (uint32_t)hi);
        kfactor_list_t ff;
        kfl_init(&ff);
        kf_trial_divide_table(n, flatslice, (int64_t)sc, &ff);
        auto fmap = to_map(ff);

        // delta
        kf_delta_slice_t ds;
        kf_load_delta_slice(&ds, DLT32, lo, hi);
        kfactor_list_t df;
        kfl_init(&df);
        kf_trial_divide_delta(n, &ds, &df);
        auto dmap = to_map(df);

        CHECK(fmap == truth, "flat table recovers the exact factorization");
        CHECK(dmap == truth, "delta table recovers the exact factorization");
        CHECK(dmap == fmap, "delta factorization == flat factorization");

        kfl_free(&ff);
        kfl_free(&df);
        kf_delta_slice_free(&ds);
        free(flatslice);
        mpz_clear(n);

        // (4) alg3/alg4 case — a cofactor OUTSIDE [lo,hi] (the medium prime /
        // composite). Trial division must strip only the in-range B-smooth part
        // and leave the identical residual, so the `done` early-out never fires.
        mpz_t n2;
        mpz_init(n2);
        std::vector<uint64_t> inrange;
        for (int i = 0; i < 12; i++)
            inrange.push_back(primes[(size_t)((uint64_t)rand() * primes.size() / ((uint64_t)RAND_MAX + 1))]);
        auto truth2 = build_smooth(n2, inrange);
        mpz_mul_ui(n2, n2, 4294967311u); // × a prime > 2^32 (out of the k32 range)

        uint32_t *fs2 = kf_load_prime_slice(&sc, FLAT32, (uint32_t)lo, (uint32_t)hi);
        kfactor_list_t ff2; kfl_init(&ff2);
        kf_trial_divide_table(n2, fs2, (int64_t)sc, &ff2);
        auto fmap2 = to_map(ff2);

        kf_delta_slice_t ds2; kf_load_delta_slice(&ds2, DLT32, lo, hi);
        kfactor_list_t df2; kfl_init(&df2);
        kf_trial_divide_delta(n2, &ds2, &df2);
        auto dmap2 = to_map(df2);

        CHECK(fmap2 == truth2 && dmap2 == truth2,
              "out-of-range cofactor: both strip only the in-range part");
        CHECK(dmap2 == fmap2, "delta == flat with a residual cofactor > 1");

        kfl_free(&ff2); kfl_free(&df2); kf_delta_slice_free(&ds2); free(fs2);
        mpz_clear(n2);
    }

    // =====================================================================
    // u64 (k35 upper range) — (2^32, 2^32 + 4,000,000]
    // =====================================================================
    printf("\n-- u64 (k35-upper-style) --\n");
    {
        auto sp = small_primes(100000);
        uint64_t lo = (1ull << 32) + 1, hi = (1ull << 32) + 4000000ull;
        auto primes = sieve_range<uint64_t>(lo, hi, sp);
        printf("  generated %zu primes in (2^32, 2^32+4e6]\n", primes.size());
        write_flat_u64(FLAT64, primes);
        CHECK(kf_convert_prime_table_to_delta_u64(FLAT64, DLT64, "  "), "convert u64 flat -> delta");

        // round-trip
        kf_delta_slice_t full;
        CHECK(kf_load_delta_slice(&full, DLT64, 0, UINT64_MAX), "load full u64 delta slice");
        CHECK(full.width == 8, "u64 delta slice reports width 8");
        auto dec = decode_all(full);
        bool rt = (dec.size() == primes.size());
        for (size_t i = 0; rt && i < primes.size(); i++)
            rt = (dec[i] == primes[i]);
        CHECK(rt, "round-trip decode == source primes (u64)");
        kf_delta_slice_free(&full);

        // trial-division equivalence (flat_u64 vs delta)
        std::vector<uint64_t> fac;
        for (int i = 0; i < 25; i++)
        {
            uint64_t p = primes[(size_t)((uint64_t)rand() * primes.size() / ((uint64_t)RAND_MAX + 1))];
            fac.push_back(p);
            if (i % 4 == 0) fac.push_back(p);
        }
        mpz_t n;
        mpz_init(n);
        auto truth = build_smooth(n, fac);

        int64_t sc;
        uint64_t *flatslice = kf_load_prime_slice_u64(&sc, FLAT64, lo, hi);
        kfactor_list_t ff;
        kfl_init(&ff);
        kf_trial_divide_table_u64(n, flatslice, sc, &ff);
        auto fmap = to_map(ff);

        kf_delta_slice_t ds;
        kf_load_delta_slice(&ds, DLT64, lo, hi);
        kfactor_list_t df;
        kfl_init(&df);
        kf_trial_divide_delta(n, &ds, &df);
        auto dmap = to_map(df);

        CHECK(fmap == truth, "flat u64 table recovers the exact factorization");
        CHECK(dmap == truth, "delta u64 table recovers the exact factorization");
        CHECK(dmap == fmap, "delta factorization == flat factorization (u64)");

        kfl_free(&ff);
        kfl_free(&df);
        kf_delta_slice_free(&ds);
        free(flatslice);
        mpz_clear(n);
    }

    remove(FLAT32); remove(DLT32); remove(FLAT64); remove(DLT64);
    printf("\n=== %s ===\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
