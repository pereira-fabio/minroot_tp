#!/bin/bash -l
#SBATCH --job-name=minroot_gen_challenges
#SBATCH --output=minroot_gen_challenges_%j.out
#SBATCH --error=minroot_gen_challenges_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

# No MPI or OpenMP needed — challenge generation is single-threaded GMP arithmetic.

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Number of challenges to generate per (bitbound, bittarget) combination.
# 2 gives you challenge_0 and challenge_1 — one for benchmarking, one spare.
NUM_CHALLENGES=2

# Target modulo sizes to test — lower and upper bounds of the parallelisation
# benefit range.
# Target sizes per bitbound — minimum is 2x bitbound (need at least 2 primes)
SIZES_K16="32 64 128 256 512 1024 2048 4096 8192"   # bitbound=16
SIZES_K32="1024 2048 4096 5120 6144 7168 8192"       # bitbound=32
SIZES_K35="8192 12288 16384 20480 24576"       # bitbound=35
 
# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
gcc -O2 -o challenges challenges.c -lgmp

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation of challenges.c failed"
    exit 1
fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Helper: generate all sizes for one bitbound
#
# Usage: generate_all <bitbound>
#
# Writes to: data/challenges/k<bitbound>/
#   modulo_<bitbound>_<bittarget>.txt      — one modulo per bittarget
#   challenge_<i>_<bitbound>_<bittarget>.txt
#   factors_<i>_<bitbound>_<bittarget>.txt
#
# Note: when bittarget < bitbound the generator prints a "weird inputs" warning
# but still produces valid output — this is expected for small target sizes
# (e.g. bittarget=16 with bitbound=32).
# ---------------------------------------------------------------------------
generate_all() {
    local BITBOUND=$1
    local SIZES=$2

    echo "============================================================"
    echo "  Generating k${BITBOUND} challenges  (bitbound=${BITBOUND})"
    echo "  Output: data/challenges/k${BITBOUND}/"
    echo "============================================================"

    for SIZE in $SIZES; do
        echo -n "  bittarget=${SIZE} ... "
        ./challenges $NUM_CHALLENGES $SIZE $BITBOUND
        if [ $? -eq 0 ]; then
            echo "  OK → data/challenges/k${BITBOUND}/"
        else
            echo "  FAILED (bittarget=${SIZE}, bitbound=${BITBOUND})"
        fi
        echo ""
    done
}

# ---------------------------------------------------------------------------
# Generate k16 challenges  (16-bit prime factors)
# ---------------------------------------------------------------------------
generate_all 16 "$SIZES_K16"

# ---------------------------------------------------------------------------
# Generate k32 challenges  (32-bit prime factors)
# ---------------------------------------------------------------------------
generate_all 32 "$SIZES_K32"

# ---------------------------------------------------------------------------
# Generate k64 challenges  (64-bit prime factors)
# ---------------------------------------------------------------------------
generate_all 35 "$SIZES_K35"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Challenge generation complete"
echo ""
echo "Files written:"
for KB in 16 32 35; do
    echo "  data/challenges/k${KB}/"
    ls data/challenges/k${KB}/ 2>/dev/null | sed 's/^/    /'
done
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"