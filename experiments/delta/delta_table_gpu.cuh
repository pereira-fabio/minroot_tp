// =============================================================================
// delta_table_gpu.cuh — GPU decode kernels for the delta-encoded prime table
//
// EXPERIMENT — not part of the attack. See experiments/delta/README.md.
//
// The flat prime table stores 4 bytes per prime. This one stores 1: an anchor
// per 1024-prime block, then a byte per gap-half. The question it was built to
// answer is whether shrinking the table below a GPU's L2 makes trial division
// faster. The answer is essentially no: the 3.99x smaller table buys 1.10x,
// because trial division is bound by big-integer arithmetic, not by memory
// traffic. Kept because the measurement is reported in the thesis and
// scripts/iris/all_gpu.slurm still runs it.
//
// One CUDA thread block per 1024-prime delta block, grid-stride over blocks.
// Thread 0 decodes the block (anchor + running gap-half sum) into shared
// memory; then all threads trial-divide their primes with the SAME
// divides_big_u32/u64 as the flat kernels. Decode is ~1024 adds against a
// ~384-limb mod per prime (<0.3% overhead), so no cooperative block-scan is
// needed.
//
// Block layout (width=4): [uint32 anchor][uint8 gaphalf[len-1]];
//   p[0]=anchor, p[k]=p[k-1]+2*gaphalf[k]. Width 8 is the same with a 64-bit
// anchor. The final block of the whole file may be partial (len<B). d_delta
// points at THIS device's contiguous block range; local block j is at byte
// j*FULL. first_block_index is the GLOBAL index of local block 0, needed only
// to size the last, possibly-short block.
//
// Requires gpu_common.h and gpu_k32_kernels.cuh (d_y_limbs, d_nlimbs) and
// delta_table.h (KF_DELTA_BLOCK).
// =============================================================================

#pragma once

#include "gpu_common.h"
#include "gpu_k32_kernels.cuh"
#include "delta_table.h"

#include <cstdint>

// Bytes per full block: anchor + one gap-half byte per subsequent prime.
#define K32_DELTA_FULL  ((size_t)4 + (KF_DELTA_BLOCK - 1)) // 1027 B (width 4)
#define K35U_DELTA_FULL ((size_t)8 + (KF_DELTA_BLOCK - 1)) // 1031 B (width 8)

// Lower range (65536, 2^32] — 4-byte anchors, uint32 primes.
__global__ void k32_trial_divide_delta_kernel(
    const uint8_t *__restrict__ d_delta, uint32_t n_blocks,
    uint64_t first_block_index, uint64_t total_count,
    uint64_t *out_factors, int *out_count)
{
    __shared__ uint32_t s_primes[KF_DELTA_BLOCK]; // 4 KB
    const int nlimbs = d_nlimbs;

    for (uint32_t j = blockIdx.x; j < n_blocks; j += gridDim.x)
    {
        uint64_t g = first_block_index + j;
        uint64_t base = g * (uint64_t)KF_DELTA_BLOCK;
        uint32_t len = (base + KF_DELTA_BLOCK <= total_count)
                           ? (uint32_t)KF_DELTA_BLOCK
                           : (uint32_t)(total_count - base);
        const uint8_t *r = d_delta + (size_t)j * K32_DELTA_FULL;

        // Decode: thread 0 walks the running gap-half sum into shared memory.
        // Anchor is little-endian; read byte-wise (a block start is not 4-aligned).
        if (threadIdx.x == 0)
        {
            uint32_t cur = (uint32_t)r[0] | ((uint32_t)r[1] << 8) | ((uint32_t)r[2] << 16) | ((uint32_t)r[3] << 24);
            s_primes[0] = cur;
            const uint8_t *gh = r + 4;
            for (uint32_t k = 1; k < len; k++)
            {
                cur += (uint32_t)gh[k - 1] << 1;
                s_primes[k] = cur;
            }
        }
        __syncthreads();

        // Trial division (the expensive part — identical to the flat kernel).
        for (uint32_t k = threadIdx.x; k < len; k += blockDim.x)
        {
            uint64_t p = s_primes[k];
            if (divides_big_u32(d_y_limbs, nlimbs, (uint32_t)p))
            {
                int slot = atomicAdd(out_count, 1);
                if (slot < MAX_GPU_FACTORS)
                    out_factors[slot] = p;
            }
        }
        __syncthreads(); // reuse s_primes next iteration
    }
}

// k35 upper range (2^32, 2^35] — 8-byte anchors, uint64 primes.
__global__ void k35_upper_trial_divide_delta_kernel(
    const uint8_t *__restrict__ d_delta, uint32_t n_blocks,
    uint64_t first_block_index, uint64_t total_count,
    uint64_t *out_factors, int *out_count)
{
    __shared__ uint64_t s_primes[KF_DELTA_BLOCK]; // 8 KB
    const int nlimbs = d_nlimbs;
    for (uint32_t j = blockIdx.x; j < n_blocks; j += gridDim.x)
    {
        uint64_t g = first_block_index + j;
        uint64_t base = g * (uint64_t)KF_DELTA_BLOCK;
        uint32_t len = (base + KF_DELTA_BLOCK <= total_count)
                           ? (uint32_t)KF_DELTA_BLOCK
                           : (uint32_t)(total_count - base);
        const uint8_t *r = d_delta + (size_t)j * K35U_DELTA_FULL;
        if (threadIdx.x == 0)
        {
            // 8-byte little-endian anchor, read byte-wise (blocks are not 8-aligned).
            uint64_t cur = 0;
            for (int b = 0; b < 8; b++)
                cur |= ((uint64_t)r[b]) << (8 * b);
            s_primes[0] = cur;
            const uint8_t *gh = r + 8;
            for (uint32_t k = 1; k < len; k++)
            {
                cur += (uint64_t)gh[k - 1] << 1;
                s_primes[k] = cur;
            }
        }
        __syncthreads();
        for (uint32_t k = threadIdx.x; k < len; k += blockDim.x)
        {
            uint64_t q = s_primes[k];
            if (divides_big_u64(d_y_limbs, nlimbs, q))
            {
                int slot = atomicAdd(out_count, 1);
                if (slot < MAX_GPU_FACTORS)
                    out_factors[slot] = q;
            }
        }
        __syncthreads();
    }
}
