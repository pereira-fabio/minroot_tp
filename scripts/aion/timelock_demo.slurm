#!/bin/bash -l
#SBATCH --job-name=minroot_timelock
#SBATCH --output=minroot_timelock_%j.out
#SBATCH --error=minroot_timelock_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# ---------------------------------------------------------------------------
# MinRoot Time-Lock Puzzle — build, generate, solve, attack
#
# Single-node, single-core job. No MPI needed — the tool is serial.
#
# Runs three configurations to show the attack speedup:
#
#   Config A — 64-bit prime, T=5000
#     u = |p|/log2(B) = 64/16 = 4, rho(4) ~ 1/200
#     Attack needs ~200 blinding attempts per step — practical demo.
#
#   Config B — 128-bit prime, T=500
#     u = 128/16 = 8, rho(8) ~ 1/31M
#     Attack is slow (many retries); demonstrates sequential solver only.
#     Shown for comparison — do not wait for attack to finish.
#
# Submit with:
#   sbatch timelock_demo.slurm
#
# Output lands in:
#   data/timelock/<jobid>/
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0
# OpenSSL is available system-wide on AION nodes (no module needed).

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
echo "=== Compilation ==="
make -C tools/timelock

if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi
echo "Compilation successful"
echo ""

TOOL=$SLURM_SUBMIT_DIR/tools/timelock/minroot-timelock

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT=$SLURM_SUBMIT_DIR/data/timelock/${SLURM_JOB_ID}
mkdir -p $OUT

# ---------------------------------------------------------------------------
# Config A: 64-bit prime, T=5000, B=65536
#   u = 4  ->  rho(4) ~ 0.00491  ->  ~155 blinding attempts per step
#   Expected solve  : well under a second (64-bit modexps are ~microseconds)
#   Expected attack : ~75 s — SLOWER than solve at 64-bit (Gap 1; see summary below)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Config A: 64-bit prime, T=5000, e=5, B=65536"
echo "============================================================"
echo ""

PUZZLE_A=$OUT/puzzle_64b.mrt
MESSAGE_A="MinRoot time-lock: 64-bit demo"

echo "--- Generate ---"
$TOOL generate \
    -b 64 \
    -T 5000 \
    -e 5 \
    -m "$MESSAGE_A" \
    $PUZZLE_A
echo ""

echo "--- Sequential solve ---"
time $TOOL solve $PUZZLE_A
echo ""

echo "--- Attack solve (B=65536) ---"
time $TOOL attack -B 65536 $PUZZLE_A
echo ""

# ---------------------------------------------------------------------------
# Config B: 128-bit prime, T=200, sequential solve only
#   u = 8  ->  rho(8) ~ 2^{-24.9}  ->  ~31M blinding attempts per step
#   Attack not run here (would take too long without higher B).
#   Demonstrates sequential cost scaling with prime size.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Config B: 128-bit prime, T=200, e=5 — sequential solve only"
echo "  (Attack with B=65536 not practical at u=8; needs B >= 2^32)"
echo "============================================================"
echo ""

PUZZLE_B=$OUT/puzzle_128b.mrt
MESSAGE_B="MinRoot time-lock: 128-bit"

echo "--- Generate ---"
$TOOL generate \
    -b 128 \
    -T 200 \
    -e 5 \
    -m "$MESSAGE_B" \
    $PUZZLE_B
echo ""

echo "--- Sequential solve ---"
time $TOOL solve $PUZZLE_B
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Job ${SLURM_JOB_ID} complete."
echo "Puzzle files in: $OUT/"
ls $OUT/*.mrt 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Config A reports solve_time vs attack_time. NOTE: at a 64-bit prime one"
echo "sequential modexp is ~microseconds, so the smoothness attack (~150 blinding"
echo "attempts/step) is SLOWER per step — the attack only wins when the modulus is"
echo "large enough that a single modexp is expensive (see the Gap-1 / t-step pages)."
echo "Config B shows sequential cost — attack requires B >= 2^32 for 128-bit (u=8)."
echo "============================================================"

# Edit: 2026-06-13
