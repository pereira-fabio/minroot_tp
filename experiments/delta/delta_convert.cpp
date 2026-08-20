// =============================================================================
// delta_convert.cpp — build a delta-encoded prime table from a flat one
//
// EXPERIMENT — not part of the attack. See experiments/delta/README.md.
//
// This used to be a --convert-delta mode of alg2_k32, alg2_k35, alg3 and alg4,
// which meant running a single-threaded file conversion through the MPI job
// launcher (srun -N1 --ntasks=2 ./alg2_k32 --convert-delta) and carrying the
// same block of code in four solvers. It is a file format conversion; it does
// not need MPI, a cluster allocation, or a solver.
//
// Usage:
//   delta_convert <flat.bin> [out.dlt]        4-byte anchors  (k32 lower range)
//   delta_convert --u64 <flat.bin> [out.dlt]  8-byte anchors  (k35 upper range)
//
// The output path defaults to the input with .bin replaced by .dlt. Both
// solvers and CUDA kernels detect the format by magic, so nothing needs to be
// told which table it was handed.
//
// Build:  make delta_convert
// =============================================================================

#include "delta_table.h"

#include <stdio.h>
#include <string.h>

// k32_primes.bin -> k32_primes.dlt
static void default_out_path(char *dst, size_t n, const char *flat)
{
    snprintf(dst, n, "%s", flat);
    size_t L = strlen(dst);
    if (L > 4 && strcmp(dst + L - 4, ".bin") == 0)
        snprintf(dst + L - 4, 5, ".dlt");
    else if (L + 5 <= n)
        snprintf(dst + L, 5, ".dlt");
}

static int usage(void)
{
    fprintf(stderr,
            "Usage: delta_convert [--u64] <flat.bin> [out.dlt]\n"
            "\n"
            "  --u64   8-byte anchors, for the k35 upper table over (2^32, 2^35].\n"
            "          Omit for the k32 lower table over [65537, 2^32].\n"
            "\n"
            "  out.dlt defaults to <flat.bin> with the extension replaced.\n");
    return 2;
}

int main(int argc, char **argv)
{
    bool u64 = false;
    const char *flat = nullptr, *out = nullptr;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--u64") == 0)
            u64 = true;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
            return usage();
        else if (!flat)
            flat = argv[i];
        else if (!out)
            out = argv[i];
        else
            return usage();
    }
    if (!flat)
        return usage();

    char derived[512];
    if (!out)
    {
        default_out_path(derived, sizeof(derived), flat);
        out = derived;
    }

    bool ok = u64 ? kf_convert_prime_table_to_delta_u64(flat, out, "")
                  : kf_convert_prime_table_to_delta(flat, out, "");
    if (!ok)
    {
        fprintf(stderr, "delta_convert: failed to convert %s\n", flat);
        return 1;
    }
    printf("delta_convert: %s -> %s\n", flat, out);
    return 0;
}
