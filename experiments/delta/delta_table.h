// =============================================================================
// delta_table.h — delta/gap-encoded prime table (CPU side)
//
// EXPERIMENT — not part of the attack. See experiments/delta/README.md.
//
// A sorted prime list compresses ~4-8x by storing gaps, not values: every prime
// gap below 2^35 is <= 456, so (gap/2) <= 228 < 256 fits in ONE byte. The
// layout keeps O(1) random *block* access, so the schedule(static) trial-
// division loop and the per-rank / per-GPU range split are all preserved.
//
// This was built to test whether a table small enough to stay resident in cache
// speeds trial division up. It does not, materially: 3.99x less memory buys
// 1.10x, because the cost is big-integer arithmetic rather than memory traffic.
// The code is kept because the measurement is reported in the thesis and the
// benchmark scripts still exercise it.
//
// File format (.dlt):
//   [u64 magic][u64 count][u32 block_size B][u32 width]      (width = 4 or 8)
//   then ceil(count/B) blocks; block g holds len = min(B, count - g*B) primes:
//     [anchor : width bytes = p[g*B]] [u8 gaphalf[len-1]]    p[i]=p[i-1]+2*gaphalf
//   Full blocks are fixed size (width + B-1 bytes), so block g starts at byte
//   HDR(24) + g*(width+B-1): O(1) seek. Each block re-anchors its first prime,
//   so blocks decode independently (embarrassingly parallel; one per warp).
//
//   width 4  <- k32 flat table  [65537, 2^32]   (813 MB  -> ~204 MB, ~4x)
//   width 8  <- k35 upper table (2^32, 2^35]    (9.6 GB  -> ~1.2 GB, ~8x)
// =============================================================================

#pragma once

#include "common.h"
#include "factorization.h" // flat-table magics, read by the converters

#include <gmp.h>
#include <stdint.h>

//
// A sorted prime list compresses ~4-8x by storing gaps, not values: every
// prime gap below 2^35 is <= 456, so (gap/2) <= 228 < 256 fits in ONE byte.
// The layout keeps O(1) random *block* access, so the schedule(static) trial-
// division loop and the per-rank / per-GPU range split are all preserved.
//
// File format (.dlt):
//   [u64 magic][u64 count][u32 block_size B][u32 width]      (width = 4 or 8)
//   then ceil(count/B) blocks; block g holds len = min(B, count - g*B) primes:
//     [anchor : width bytes = p[g*B]] [u8 gaphalf[len-1]]    p[i]=p[i-1]+2*gaphalf
//   Full blocks are fixed size (width + B-1 bytes), so block g starts at byte
//   HDR(24) + g*(width+B-1): O(1) seek. Each block re-anchors its first prime,
//   so blocks decode independently (embarrassingly parallel; one per warp).
//
//   width 4  <- k32 flat table  [65537, 2^32]   (813 MB  -> ~204 MB, ~4x)
//   width 8  <- k35 upper table (2^32, 2^35]    (9.6 GB  -> ~1.2 GB, ~8x)
// ---------------------------------------------------------------------------

#define KF_DELTA_MAGIC_U32 0x4B3332444C544131ULL // "K32DLTA1"
#define KF_DELTA_MAGIC_U64 0x4B3335444C544131ULL // "K35DLTA1"
#define KF_DELTA_BLOCK 1024                      // primes per block (0.4% overhead)
#define KF_DELTA_HDR 24                          // header bytes

// A loaded (sub-)range of a delta table, ready for trial division.
typedef struct
{
    uint8_t *bytes;              // owned: concatenated blocks for the loaded slice
    uint64_t total_count;        // primes in the whole file (for last-block length)
    uint64_t first_block_index;  // global index of the first block in `bytes`
    uint32_t n_blocks;           // number of blocks held in `bytes`
    uint32_t block_size;         // primes per full block (B)
    int width;                   // anchor width in bytes: 4 (k32) or 8 (k35 upper)
    uint64_t range_lo, range_hi; // inclusive prime-value bounds this slice serves
} kf_delta_slice_t;

// One-time conversion: flat prime table -> delta table. Returns false (and
// removes the output) on I/O error or if any gap/2 exceeds 255 (which would
// break the 1-byte scheme — a hard correctness guard, never expected to fire).
bool kf_convert_prime_table_to_delta(const char *flat_path, const char *delta_path,
                                     const char *log_prefix); // u32 (k32)
bool kf_convert_prime_table_to_delta_u64(const char *flat_path, const char *delta_path,
                                         const char *log_prefix); // u64 (k35 upper)

// Load the delta blocks covering [range_lo, range_hi] (inclusive) into *out.
// Mirrors kf_load_prime_slice: exact prime-value boundaries (trial division
// filters the over-read at each end), so adjacent ranks never overlap.
// Returns false on error; on success free with kf_delta_slice_free.
bool kf_load_delta_slice(kf_delta_slice_t *out, const char *path,
                         uint64_t range_lo, uint64_t range_hi);

void kf_delta_slice_free(kf_delta_slice_t *s);

// Trial-divide n by a delta slice (OMP over blocks; each block is decoded into
// a stack buffer, then the same inner mpz loop as kf_trial_divide_table runs).
// Handles both widths. Output is identical to the flat table method.
void kf_trial_divide_delta(const mpz_t n, const kf_delta_slice_t *slice,
                           kfactor_list_t *found,
                           int64_t *thread_tested = nullptr,
                           int *thread_found = nullptr,
                           int nthreads_out = 0);
