#!/bin/bash -l
#SBATCH --job-name=minroot_gen_challenges
#SBATCH --output=minroot_gen_challenges_%j.out
#SBATCH --error=minroot_gen_challenges_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00

# ---------------------------------------------------------------------------
# Challenge generator — 2-node split for faster generation of 50 challenges.
#
# Node 0 (SLURM_PROCID=0):  k16 | k32 | k35 groups in parallel (10 th each)
#                            then alg3 with all 32 threads
# Node 1 (SLURM_PROCID=1):  alg4 | alg4_multi in parallel (16 th each)
#
# Wall time ≈ max(node-0 time, node-1 time).
# Node 1 typically finishes first and simply idles until node 0 is done.
#
# Alternative (single node): change header to
#   --nodes=1 --ntasks=2 --cpus-per-task=32
# The same SLURM_PROCID split works unchanged.
#
# Output files per (bitbound, bittarget):
#   data/challenges/k<B>/modulo_<B>_<bits>.txt
#   data/challenges/k<B>/challenge_<i>_<B>_<bits>.txt   (i = 0..49)
#   data/challenges/k<B>/factors_<i>_<B>_<bits>.txt
#
# Modulo files are cached — regenerated only when missing.
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Binary cleanup on any exit (success, failure, signal, timeout)
# ---------------------------------------------------------------------------
trap 'rm -f challenges' EXIT

# ---------------------------------------------------------------------------
# Configuration — exported so srun subshells inherit them
# ---------------------------------------------------------------------------
export NUM_CHALLENGES=50
# Paper-scale moduli (Gap 1): 256-bit MinRoot prime and near sizes.
# 2048 added 2026-06-25: matches RFC 3526 Group 14 / RFC 7919 ffdhe2048.
export SIZES="256 384 512 768 1024 2048 4096 8192 16384 24576"

# ---------------------------------------------------------------------------
# Compilation (runs once on the batch node; binary is on shared NFS)
# ---------------------------------------------------------------------------
gcc -O2 -fopenmp -o challenges tools/challenges.c -lgmp

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation of challenges.c failed"
    exit 1
fi
echo "Compilation successful (challenges)"
echo "NUM_CHALLENGES: ${NUM_CHALLENGES}"
echo "Sizes: ${SIZES}"
echo ""

# ---------------------------------------------------------------------------
# Distribute work across both nodes via srun
# ---------------------------------------------------------------------------
srun --ntasks=2 bash -c '
CPUS=$SLURM_CPUS_PER_TASK
THREADS_PER_GROUP=$(( CPUS / 3 ))   # 10 threads per k-group (3 run in parallel)
HALF_CPUS=$(( CPUS / 2 ))           # 16 threads per alg-group (2 run in parallel)

if [ "$SLURM_PROCID" = "0" ]; then
    echo "=== Node 0 (PROCID=0): k16/k32/k35 groups + alg3 ==="

    # k16/k32/k35 run in background simultaneously; sizes sequential within each group
    for SIZE in $SIZES; do
        echo "  [k16] ${SIZE}..."
        OMP_NUM_THREADS=$THREADS_PER_GROUP ./challenges $NUM_CHALLENGES $SIZE 16
    done &
    for SIZE in $SIZES; do
        echo "  [k32] ${SIZE}..."
        OMP_NUM_THREADS=$THREADS_PER_GROUP ./challenges $NUM_CHALLENGES $SIZE 32
    done &
    for SIZE in $SIZES; do
        echo "  [k35] ${SIZE}..."
        OMP_NUM_THREADS=$THREADS_PER_GROUP ./challenges $NUM_CHALLENGES $SIZE 35
    done &
    wait
    echo "k16/k32/k35 done."

    # alg3 runs alone with all CPUs after k-groups finish
    for SIZE in $SIZES; do
        echo "  [alg3] ${SIZE}..."
        OMP_NUM_THREADS=$CPUS ./challenges $NUM_CHALLENGES $SIZE 32 alg3
    done
    echo "alg3 done."

else
    echo "=== Node 1 (PROCID=1): alg4 + alg4_multi ==="

    # alg4 and alg4_multi run in background simultaneously
    for SIZE in $SIZES; do
        echo "  [alg4] ${SIZE}..."
        OMP_NUM_THREADS=$HALF_CPUS ./challenges $NUM_CHALLENGES $SIZE 32 alg4
    done &
    for SIZE in $SIZES; do
        echo "  [alg4_multi] ${SIZE}..."
        OMP_NUM_THREADS=$HALF_CPUS ./challenges $NUM_CHALLENGES $SIZE 32 alg4_multi
    done &
    wait
    echo "alg4 + alg4_multi done."
fi
'

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Challenge generation complete"
echo ""
echo "Files written:"
for KB in 16 32 35; do
    COUNT=$(ls data/challenges/k${KB}/challenge_* 2>/dev/null | wc -l)
    echo "  data/challenges/k${KB}/  (${COUNT} challenge files)"
done
for ALG in alg3 alg4 alg4_multi; do
    COUNT=$(ls data/challenges/${ALG}/challenge_* 2>/dev/null | wc -l)
    echo "  data/challenges/${ALG}/  (${COUNT} challenge files)"
done
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

# Edit: 2026-06-25
