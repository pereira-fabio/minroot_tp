#pragma once

// Canonical challenge sizes (bit-lengths) shared by all MinRoot solvers.
// Used in --all mode to iterate every benchmark size in one MPI execution.
// Update this list whenever challenge_gen.slurm changes its SIZES variable.
//
// 2026-06-11: switched to the paper's small-modulus regime {256, 384, 512,
// 768, 1024} to demonstrate the attack at (and near) the paper's actual
// 256-bit MinRoot prime, per Gap 1.  Replaces the earlier large-modulus
// sweep (256..24576) used to chase wall-clock speedup; see
// wiki/planning/gap-analysis.md Gap 1 and wiki/DASHBOARD.md.
//
// 2026-06-25: added 2048 -- matches RFC 3526 Group 14 / RFC 7919 ffdhe2048
// (real-world DH prime size) and is needed for fresh, current-codebase data
// at the size the timelock demo plan uses; see
// wiki/planning/timelock-demo-realism-plan-2026-06-25.md.
#define KSC_CANONICAL_SIZES_LIST \
    "256", "384", "512", "768", "1024", "2048"

#define KSC_N_CANONICAL_SIZES 6

// Edit: 2026-06-25
