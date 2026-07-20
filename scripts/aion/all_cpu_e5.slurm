#!/bin/bash -l
#SBATCH --job-name=minroot_all_cpu_e5
#SBATCH --output=minroot_all_cpu_e5_%A_%a.out
#SBATCH --error=minroot_all_cpu_e5_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=24:00:00

# ===========================================================================
# Universal AION CPU sweep — EXPONENT 5 (the actual MinRoot 5th root).
# The e=5 companion to all_cpu.slurm: same build-all-then-run-all structure and
# descending node sweep, but every solver runs "--exponent 5 --modulo-suffix _e5"
# against the e5 modulo files (gcd(5, M-1)=1, from challenge_gen_e5.slurm).
# Challenge files are exponent-agnostic and shared with the e3 run.
#
# Solvers: k16, k32, k35, alg3, alg4 — the five that support --modulo-suffix.
# alg7 is NOT included: the CPU alg7 solver has no --modulo-suffix flag (it was
# never an e5 target and cannot select the _e5 modulus). Run alg7 at e3 only
# (all_cpu.slurm). This matches the retired per-algorithm e5 slurm set, which
# also had no alg7_e5.
#
# Node sweep DESCENDING (task 0 = 64 nodes; 128 ranks):
#   task 0 -> 64 nodes    task 3 ->  8 nodes    task 6 -> 1 node
#   task 1 -> 32 nodes    task 4 ->  4 nodes
#   task 2 -> 16 nodes    task 5 ->  2 nodes
#
# Each: powm baseline once, then table + sieve (--skip-powm-baseline). Results:
#   data/results/aion/alg2/{k16,k32,k35}_e5_<jid>_N<n>/
#   data/results/aion/{alg3,alg4}/<alg>_e5_<jid>_N<n>/
#
# PREREQUISITES:
#   1. e5 modulo files:  sbatch challenge_gen_e5.slurm   (modulo_*_<bits>_e5.txt)
#   2. (optional) an e5 cbrt merge. If data/cbrt/<variant>_e5/cbrt_<size>.cbrt is
#      absent, each rank builds its cbrts locally (slower but correct) — the e5
#      merge is a separate follow-up (merge_cbrt.slurm currently does e3 only).
#
# Submit:  sbatch all_cpu_e5.slurm
# ===========================================================================

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

# ---------------------------------------------------------------------------
# Map array index -> node count (DESCENDING) and challenge count.
# ---------------------------------------------------------------------------
NODE_COUNTS=(64 32 16 8 4 2 1)
N_NODES=${NODE_COUNTS[$SLURM_ARRAY_TASK_ID]}
N_CHALLENGE_COUNTS=(50 50 40 30 20 15 10)
N_CHALLENGES=${N_CHALLENGE_COUNTS[$SLURM_ARRAY_TASK_ID]}
NRANKS=$(( N_NODES * 2 ))

cd $SLURM_SUBMIT_DIR

echo "=== Universal AION CPU sweep — exponent 5 ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES  (ranks: $NRANKS, workers: $(( NRANKS * SLURM_CPUS_PER_TASK )))"
echo "Challenges:    $N_CHALLENGES per method  (all timed warm runs — Plan A, no cold run)"
echo "Exponent:      5 (5th root — actual MinRoot VDF)  |  modulo suffix: _e5"
echo "Sizes:         256 384 512 768 1024 2048 4096 8192 16384 24576 (canonical --all)"
echo ""

# ---------------------------------------------------------------------------
# run_variant NAME SRC DATA_DIR CBRT_DIR RESULT_SUBPATH JSON_PREFIX
#   builds src/aion/<SRC>, then runs powm + table + sieve at the current node
#   count with --exponent 5 --modulo-suffix _e5. A compile failure skips just
#   this variant; the sweep continues.
# ---------------------------------------------------------------------------
run_variant() {
    local NAME=$1 SRC=$2 DATA=$3 CBRT=$4 RSUB=$5 JPREFIX=$6
    local BIN=./${NAME}_e5_N${N_NODES}
    local JDIR=$SLURM_SUBMIT_DIR/data/results/aion/${RSUB}_e5_${SLURM_JOB_ID}_N${N_NODES}
    mkdir -p "$JDIR" "$CBRT"

    echo "============================================================"
    echo "[$NAME e5]  N=${N_NODES}  data=$DATA  cbrt=$CBRT"
    echo "  JSON -> $JDIR/"
    echo "============================================================"

    mpicxx -O3 -std=c++17 -fopenmp -Wall \
        -I src/common \
        src/aion/${SRC} src/common/solver_common.cpp \
        src/common/common.cpp src/common/factorization.cpp \
        -lgmp -o "$BIN"
    if [ $? -ne 0 ]; then
        echo "  !! compilation failed for $NAME — skipping this variant"
        return 1
    fi

    local M SKIP
    for M in powm table sieve; do
        SKIP=""
        [ "$M" != "powm" ] && SKIP="--skip-powm-baseline"
        echo "  [$NAME e5 $M]"
        srun --mpi=pmix -N ${N_NODES} --ntasks=${NRANKS} "$BIN" \
            --all --data "$DATA" \
            --exponent 5 --modulo-suffix _e5 \
            --method $M $SKIP \
            --n-challenges $N_CHALLENGES \
            --cbrt-dir "$CBRT" \
            --json $JDIR/${JPREFIX}_${M}_${SLURM_JOB_ID}
        echo ""
    done

    rm -f "$BIN"
}

# ---------------------------------------------------------------------------
# Five solvers at e=5 (no alg7 — it lacks --modulo-suffix).
#             NAME       SRC             DATA_DIR               CBRT_DIR                                 RESULT_SUBPATH  JSON_PREFIX
# ---------------------------------------------------------------------------
run_variant  alg2_k16   alg2_k16.cpp    data/challenges/k16    $SLURM_SUBMIT_DIR/data/cbrt/alg2_k16_e5  alg2/k16        alg2_k16
run_variant  alg2_k32   alg2_k32.cpp    data/challenges/k32    $SLURM_SUBMIT_DIR/data/cbrt/alg2_k32_e5  alg2/k32        alg2_k32
run_variant  alg2_k35   alg2_k35.cpp    data/challenges/k35    $SLURM_SUBMIT_DIR/data/cbrt/alg2_k35_e5  alg2/k35        alg2_k35
run_variant  alg3       alg3.cpp        data/challenges/alg3   $SLURM_SUBMIT_DIR/data/cbrt/alg3_e5      alg3/alg3       alg3
run_variant  alg4       alg4.cpp        data/challenges/alg4   $SLURM_SUBMIT_DIR/data/cbrt/alg4_e5      alg4/alg4       alg4

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}, e5) complete."
echo "Results under data/results/aion/ (job $SLURM_JOB_ID):"
find $SLURM_SUBMIT_DIR/data/results/aion -type d -name "*_e5_${SLURM_JOB_ID}_N${N_NODES}" 2>/dev/null | sed 's|.*/aion/|  |'
echo ""
echo "Use warm_avg.* (all timed runs averaged; no cold run) for analysis."
echo "Join each timing JSON with its _powm_ sibling for the baseline (results-join-methodology)."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"

# Edit: 2026-07-17
