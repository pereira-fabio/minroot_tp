#!/bin/bash -l
#SBATCH --job-name=minroot_gen_challenges_e5
#SBATCH --output=minroot_gen_challenges_e5_%j.out
#SBATCH --error=minroot_gen_challenges_e5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# Challenge generator — exponent 5 (5th root / actual MinRoot VDF operation)
#
# Generates modulo files where gcd(5, M-1) = 1, required for the exponent-5
# solvers (--exponent 5 flag in all solver binaries after Phase 5).
#
# Key difference from challenge_gen.slurm:
#   All ./challenges calls include --exponent 5.
#   Modulo files get an _e5 suffix: modulo_32_1024_e5.txt
#   Challenge files (smooth numbers) are exponent-agnostic — the generator
#   writes them to the same directories with the same names as exponent-3.
#   Since seeds are deterministic the content is identical; effectively only
#   the _e5 modulo files are new output.
#
# Structure: k16/k32/k35 groups run in parallel; alg3/alg4/alg4_multi run
# sequentially after, using all available threads each.
#
# Output modulo files per variant:
#   data/challenges/k<B>/modulo_<B>_<bits>_e5.txt
#   data/challenges/alg3/modulo_alg3_<bits>_e5.txt
#   data/challenges/alg4/modulo_alg4_<bits>_e5.txt
#   data/challenges/alg4_multi/modulo_alg4_multi_<bits>_e5.txt
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_CHALLENGES=5
# Full sweep (256..24576) — matches challenge_gen.slurm (e3) and src/common/sizes.h
# so AION --all can benchmark e5 at large moduli too (2026-07-11).
# Paper-scale {256..2048} are the Gap-1 sizes; 2048 also = RFC 3526 Group 14.
SIZES="256 384 512 768 1024 2048 4096 8192 16384 24576"
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
# ---------------------------------------------------------------------------
generate_size() {
    local BITBOUND=$1
    local SIZE=$2
    echo "  [k${BITBOUND} e5] bittarget=${SIZE}..."
    OMP_NUM_THREADS=${THREADS_PER_GROUP} ./challenges $NUM_CHALLENGES $SIZE $BITBOUND --exponent 5
    local RET=$?
    if [ $RET -eq 0 ]; then
        echo "  [k${BITBOUND} e5] ${SIZE} bits DONE"
    else
        echo "  [k${BITBOUND} e5] ${SIZE} bits FAILED (exit ${RET})"
    fi
}

generate_group() {
    local BITBOUND=$1
    local N_SIZES=$(echo $SIZES | wc -w)
    echo "=== k${BITBOUND} e5 group: ${N_SIZES} sizes × ${NUM_CHALLENGES} challenges ==="
    for SIZE in $SIZES; do
        generate_size $BITBOUND $SIZE
    done
    echo "=== k${BITBOUND} e5 group complete ==="
}

# ---------------------------------------------------------------------------
# Launch k16/k32/k35 groups in parallel
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Launching k16, k32, k35 groups in parallel (exponent=5)"
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
echo "k16/k32/k35 e5 groups done."

# ---------------------------------------------------------------------------
# ALG3 e5 group
# ---------------------------------------------------------------------------
N_SIZES_ALG3=$(echo $SIZES | wc -w)

echo "============================================================"
echo "Launching alg3 e5 group: ${N_SIZES_ALG3} sizes × ${NUM_CHALLENGES} challenges"
echo "  Sizes:   $SIZES"
echo "  Threads: ${SLURM_CPUS_PER_TASK} (all available)"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    echo "  [alg3 e5] bittarget=${SIZE}..."
    OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} ./challenges $NUM_CHALLENGES $SIZE 32 alg3 --exponent 5
    RET=$?
    if [ $RET -eq 0 ]; then
        echo "  [alg3 e5] ${SIZE} bits DONE"
    else
        echo "  [alg3 e5] ${SIZE} bits FAILED (exit ${RET})"
    fi
done

echo ""
echo "alg3 e5 group done."

# ---------------------------------------------------------------------------
# ALG4 e5 group
# ---------------------------------------------------------------------------
N_SIZES_ALG4=$(echo $SIZES | wc -w)

echo "============================================================"
echo "Launching alg4 e5 group: ${N_SIZES_ALG4} sizes × ${NUM_CHALLENGES} challenges"
echo "  Sizes:   $SIZES"
echo "  Threads: ${SLURM_CPUS_PER_TASK} (all available)"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    echo "  [alg4 e5] bittarget=${SIZE}..."
    OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} ./challenges $NUM_CHALLENGES $SIZE 32 alg4 --exponent 5
    RET=$?
    if [ $RET -eq 0 ]; then
        echo "  [alg4 e5] ${SIZE} bits DONE"
    else
        echo "  [alg4 e5] ${SIZE} bits FAILED (exit ${RET})"
    fi
done

echo ""
echo "alg4 e5 group done."

# ---------------------------------------------------------------------------
# ALG4_MULTI e5 group
# ---------------------------------------------------------------------------
N_SIZES_ALG4_MULTI=$(echo $SIZES | wc -w)

echo "============================================================"
echo "Launching alg4_multi e5 group: ${N_SIZES_ALG4_MULTI} sizes × ${NUM_CHALLENGES} challenges"
echo "  Sizes:   $SIZES"
echo "  Threads: ${SLURM_CPUS_PER_TASK} (all available)"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    echo "  [alg4_multi e5] bittarget=${SIZE}..."
    OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} ./challenges $NUM_CHALLENGES $SIZE 32 alg4_multi --exponent 5
    RET=$?
    if [ $RET -eq 0 ]; then
        echo "  [alg4_multi e5] ${SIZE} bits DONE"
    else
        echo "  [alg4_multi e5] ${SIZE} bits FAILED (exit ${RET})"
    fi
done

echo ""
echo "alg4_multi e5 group done."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Exponent-5 challenge generation complete"
echo ""
echo "New modulo files (gcd(5,M-1)=1):"
for KB in 16 32 35; do
    ls data/challenges/k${KB}/modulo_*_e5.txt 2>/dev/null | sed 's/^/  /'
done
ls data/challenges/alg3/modulo_*_e5.txt    2>/dev/null | sed 's/^/  /'
ls data/challenges/alg4/modulo_*_e5.txt    2>/dev/null | sed 's/^/  /'
ls data/challenges/alg4_multi/modulo_*_e5.txt 2>/dev/null | sed 's/^/  /'
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "============================================================"

# Edit: 2026-07-11
