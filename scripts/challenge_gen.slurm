#!/bin/bash -l
#SBATCH --job-name=minroot_gen_challenges
#SBATCH --output=minroot_gen_challenges_%j.out
#SBATCH --error=minroot_gen_challenges_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32  # OMP threads per ./challenges invocation
#SBATCH --time=01:00:00  # parallel: wall time = slowest single modulo generation

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

# Use all allocated CPUs for OpenMP parallel challenge generation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
SIZES_K16="32 64 128 256 512 1024 2048 4096 8192 12288 16384 20480 24576"   # bitbound=16
SIZES_K32="32 64 128 256 512 1024 2048 4096 8192 12288 16384 20480 24576"       # bitbound=32
SIZES_K35="32 64 128 256 512 1024 2048 4096 8192 12288 16384 20480 24576"       # bitbound=35
 
# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
gcc -O2 -fopenmp -o challenges tools/challenges.c -lgmp

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
# Challenges within each call are generated in parallel with OMP_NUM_THREADS.
# Note: when bittarget < bitbound the generator prints a size warning
# but still produces valid output (expected for small bittarget values).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# generate_size <bitbound> <bittarget>
#   Runs ./challenges in the background.  All sizes within a bitbound are
#   launched simultaneously so their modulo generations overlap.
#   The first call for a given bittarget generates (and caches) the modulo;
#   subsequent calls load it instantly.
# ---------------------------------------------------------------------------
generate_size() {
    local BITBOUND=$1
    local SIZE=$2
    echo "  [k${BITBOUND}] bittarget=${SIZE} starting..."
    ./challenges $NUM_CHALLENGES $SIZE $BITBOUND
    if [ $? -eq 0 ]; then
        echo "  [k${BITBOUND}] bittarget=${SIZE} DONE"
    else
        echo "  [k${BITBOUND}] bittarget=${SIZE} FAILED"
    fi
}

# ---------------------------------------------------------------------------
# Launch ALL sizes for ALL bitbounds in parallel.
#
# Each ./challenges call is independent (different modulo file path), so
# they never conflict even when running simultaneously.
#
# The expensive part is mpz_nextprime for the modulo:
#   k16/k32 sizes: <1s each  — no problem
#   k35 large sizes (20480, 24576 bits): several minutes each
# Running them in parallel means total wall time = slowest single call,
# not the sum of all calls.
# ---------------------------------------------------------------------------
echo "Launching all challenge generators in parallel..."
echo "(Each size generates its modulo independently — no conflicts)"
echo ""

for SIZE in $SIZES_K16; do
    generate_size 16 $SIZE &
done

for SIZE in $SIZES_K32; do
    generate_size 32 $SIZE &
done

for SIZE in $SIZES_K35; do
    generate_size 35 $SIZE &
done

echo "Waiting for all generators to finish..."
wait
echo "All generators done."

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