#!/bin/bash -l
#SBATCH --job-name=minroot_gen_challenges
#SBATCH --output=minroot_gen_challenges_%j.out
#SBATCH --error=minroot_gen_challenges_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# Challenge generator — produces B-smooth challenge files for k16/k32/k35.
#
# Structure: three k-variant groups run simultaneously; within each group,
# sizes are generated sequentially so modulo generation is not starved of
# threads by competing background processes.
#
#   Group k16 ──┐
#   Group k32 ──┼── run in parallel (background), each group is sequential
#   Group k35 ──┘
#
# Each group is allocated (SLURM_CPUS_PER_TASK / 3) OMP threads.  The new
# find_prime_parallel() in challenges.c uses those threads to parallelise the
# modulo search — previously the single-threaded mpz_nextprime call dominated
# wall time for large bittargets (e.g. 24576 bits ≈ several minutes on one core).
#
# Wall time = max wall time across the three groups = k35 group time,
# which is dominated by the largest sizes.  With 10 OMP threads per group,
# modulo generation is ~10× faster than the old single-threaded approach.
#
# Sizes are shared across k16/k32/k35 so all three algorithms can be directly
# compared at every bit size (same challenge files, same modulos per algorithm).
#
# Output files per (bitbound, bittarget):
#   data/challenges/k<B>/modulo_<B>_<bits>.txt          (one per bittarget)
#   data/challenges/k<B>/challenge_<i>_<B>_<bits>.txt   (i = 0 .. NUM_CHALLENGES-1)
#   data/challenges/k<B>/factors_<i>_<B>_<bits>.txt
#
# Note: modulo files are cached — if one already exists the generator loads it
# instantly and skips the expensive mpz_nextprime step.
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# 5 challenges per (bitbound, bittarget): challenge_0 through challenge_4.
# Solver SLURMs use challenge_0 by default; extras are spares for re-runs.
NUM_CHALLENGES=5

# Shared set for all three algorithms: same sizes enable direct comparison.
# Spacing is approximately log-uniform with finer resolution near the k35
# breakeven (~17000 bits sieve, ~12000 bits table) to capture the crossover.
SIZES="256 512 1024 2048 4096 6144 8192 12288 16384 20480 24576"

# Each of the three bitbound groups gets an equal share of available cores.
# With SLURM_CPUS_PER_TASK=32: 10 threads/group × 3 groups = 30 threads.
THREADS_PER_GROUP=$(( SLURM_CPUS_PER_TASK / 3 ))

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
gcc -O2 -fopenmp -o challenges tools/challenges.c -lgmp

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation of challenges.c failed"
    exit 1
fi
echo "Compilation successful (challenges)"
echo "THREADS_PER_GROUP: ${THREADS_PER_GROUP}  (${SLURM_CPUS_PER_TASK} CPUs / 3 groups)"
echo ""

# ---------------------------------------------------------------------------
# generate_size <bitbound> <bittarget>
#   Runs ./challenges with THREADS_PER_GROUP OMP threads.
#   The modulo is loaded from disk if cached, generated in parallel otherwise.
# ---------------------------------------------------------------------------
generate_size() {
    local BITBOUND=$1
    local SIZE=$2
    echo "  [k${BITBOUND}] bittarget=${SIZE}..."
    OMP_NUM_THREADS=${THREADS_PER_GROUP} ./challenges $NUM_CHALLENGES $SIZE $BITBOUND
    local RET=$?
    if [ $RET -eq 0 ]; then
        echo "  [k${BITBOUND}] ${SIZE} bits DONE"
    else
        echo "  [k${BITBOUND}] ${SIZE} bits FAILED (exit ${RET})"
    fi
}

# ---------------------------------------------------------------------------
# generate_group <bitbound>
#   Runs all sizes sequentially for one bitbound.
#   Called in the background so k16/k32/k35 groups overlap.
# ---------------------------------------------------------------------------
generate_group() {
    local BITBOUND=$1
    local N_SIZES=$(echo $SIZES | wc -w)
    echo "=== k${BITBOUND} group: ${N_SIZES} sizes × ${NUM_CHALLENGES} challenges each ==="
    for SIZE in $SIZES; do
        generate_size $BITBOUND $SIZE
    done
    echo "=== k${BITBOUND} group complete ==="
}

# ---------------------------------------------------------------------------
# Launch all three groups simultaneously.
# Sizes within each group run sequentially so each modulo generation
# has THREADS_PER_GROUP OMP threads available without interference.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Launching k16, k32, k35 groups in parallel"
echo "  Sizes:   $SIZES"
echo "  Threads: ${THREADS_PER_GROUP} per group"
echo "  Challenges per size: ${NUM_CHALLENGES}"
echo "============================================================"
echo ""

generate_group 16 &
generate_group 32 &
generate_group 35 &

wait
echo ""
echo "k16/k32/k35 groups done."

# ---------------------------------------------------------------------------
# ALG3 group — runs after the three existing groups to avoid CPU contention.
# Uses all available threads since nothing else is running.
#
# Sizes chosen to match the alg3_mpi --all set plus a few extras.
# Modulo files are written to data/challenges/alg3/ (isolated namespace).
# ---------------------------------------------------------------------------
SIZES_ALG3="256 512 1024 2048 4096 6144 8192 12288 16384 20480 24576"
N_SIZES_ALG3=$(echo $SIZES_ALG3 | wc -w)

echo "============================================================"
echo "Launching alg3 group: ${N_SIZES_ALG3} sizes × ${NUM_CHALLENGES} challenges"
echo "  Sizes:   $SIZES_ALG3"
echo "  Threads: ${SLURM_CPUS_PER_TASK} (all available)"
echo "============================================================"
echo ""

for SIZE in $SIZES_ALG3; do
    echo "  [alg3] bittarget=${SIZE}..."
    OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} ./challenges $NUM_CHALLENGES $SIZE 32 alg3
    RET=$?
    if [ $RET -eq 0 ]; then
        echo "  [alg3] ${SIZE} bits DONE"
    else
        echo "  [alg3] ${SIZE} bits FAILED (exit ${RET})"
    fi
done

echo ""
echo "alg3 group done."

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
echo "  data/challenges/alg3/"
ls data/challenges/alg3/ 2>/dev/null | sed 's/^/    /'
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"
