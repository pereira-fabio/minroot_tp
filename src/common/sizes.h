#pragma once

// Canonical challenge sizes (bit-lengths) shared by all MinRoot solvers.
// Used in --all mode to iterate every benchmark size in one MPI execution.
// Update this list whenever challenge_gen.slurm changes its SIZES variable.
#define KSC_CANONICAL_SIZES_LIST \
    "256", "384", "512", "768", "1024", "2048", "4096", "8192", "16384", "24576"

#define KSC_N_CANONICAL_SIZES 10
