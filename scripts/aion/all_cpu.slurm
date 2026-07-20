#!/bin/bash -l
#SBATCH --job-name=minroot_all_cpu
#SBATCH --output=minroot_all_cpu_%A_%a.out
#SBATCH --error=minroot_all_cpu_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=24:00:00

# ===========================================================================
# Universal AION CPU sweep — the single build-all-then-run-all script for the
# whole study (replaces the per-algorithm alg2_k16/k32/k35 / alg3 / alg4 / alg7
# slurms, which are retired). Mirrors scripts/iris/all_gpu.slurm for the CPU.
#
# Each array task runs the FULL benchmark — all six solvers, all three methods,
# all ten canonical sizes — at one node count, exponent e=3. Node counts are
# swept DESCENDING so the fast large-node tasks land first:
#
#   task 0 -> 64 nodes (128 ranks)     task 4 ->  4 nodes ( 8 ranks)
#   task 1 -> 32 nodes ( 64 ranks)     task 5 ->  2 nodes ( 4 ranks)
#   task 2 -> 16 nodes ( 32 ranks)     task 6 ->  1 node  ( 2 ranks)
#   task 3 ->  8 nodes ( 16 ranks)
#
# Solvers (single-prime Algorithm 4 only — the composite alg4_multi path is not
# run): k16, k32, k35, alg3, alg4, alg7. Each: powm baseline once, then table +
# sieve (--skip-powm-baseline). Results keep the existing per-variant layout so
# analysis/plot_results.py works unchanged:
#   data/results/aion/alg2/{k16,k32,k35}_<jid>_N<n>/
#   data/results/aion/{alg3,alg4,alg7}/<alg>_<jid>_N<n>/
#
# PREREQUISITES (run once, before this sweep):
#   1. Challenges generated for all 10 sizes at e=3:  sbatch challenge_gen.slurm
#   2. cbrt caches merged per variant:                sbatch merge_cbrt.slurm
#      (a brand-new modulus with no merged cache falls back to a per-rank build,
#       which merge_cbrt.slurm later consolidates — see M1, 2026-07-16.)
#
# Submit:  sbatch all_cpu.slurm      (all 7 tasks; SLURM schedules them)
#          sbatch --array=0 all_cpu.slurm   (just the 64-node task)
#
# NOTE on walltime: the 1-node task (task 6) running k35 at the largest moduli
# is the long pole. 24h is generous for the descending large-node tasks; if the
# N1 task times out, resubmit it alone with a trimmed size set or fewer
# --n-challenges. This is e=3 only; an e=5 sweep is a separate follow-up.
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
# More challenges at the fast large-node tasks; fewer at the slow small-node
# tasks (keeps the N1 wall-clock bounded).
# ---------------------------------------------------------------------------
NODE_COUNTS=(64 32 16 8 4 2 1)
N_NODES=${NODE_COUNTS[$SLURM_ARRAY_TASK_ID]}
N_CHALLENGE_COUNTS=(50 50 40 30 20 15 10)
N_CHALLENGES=${N_CHALLENGE_COUNTS[$SLURM_ARRAY_TASK_ID]}
NRANKS=$(( N_NODES * 2 ))

cd $SLURM_SUBMIT_DIR

echo "=== Universal AION CPU sweep ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES  (ranks: $NRANKS, workers: $(( NRANKS * SLURM_CPUS_PER_TASK )))"
echo "Challenges:    $N_CHALLENGES per method  (all timed warm runs — Plan A, no cold run)"
echo "Exponent:      3"
echo "Sizes:         256 384 512 768 1024 2048 4096 8192 16384 24576 (canonical --all)"
echo ""

# ---------------------------------------------------------------------------
# run_variant NAME SRC DATA_DIR CBRT_DIR RESULT_SUBPATH JSON_PREFIX [EXTRA_FLAGS]
#   builds src/aion/<SRC>, then runs powm + table + sieve at the current node
#   count. CBRT_DIR is the per-variant merged-cache dir (never wiped — M1).
#   A compile failure skips just this variant; the sweep continues.
# ---------------------------------------------------------------------------
run_variant() {
    local NAME=$1 SRC=$2 DATA=$3 CBRT=$4 RSUB=$5 JPREFIX=$6 EXTRA=$7
    local BIN=./${NAME}_N${N_NODES}
    local JDIR=$SLURM_SUBMIT_DIR/data/results/aion/${RSUB}_${SLURM_JOB_ID}_N${N_NODES}
    mkdir -p "$JDIR" "$CBRT"

    echo "============================================================"
    echo "[$NAME]  N=${N_NODES}  data=$DATA  cbrt=$CBRT"
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
        echo "  [$NAME $M]"
        srun --mpi=pmix -N ${N_NODES} --ntasks=${NRANKS} "$BIN" \
            --all --data "$DATA" \
            --method $M $SKIP $EXTRA \
            --n-challenges $N_CHALLENGES \
            --cbrt-dir "$CBRT" \
            --json $JDIR/${JPREFIX}_${M}_${SLURM_JOB_ID}
        echo ""
    done

    rm -f "$BIN"
}

# ---------------------------------------------------------------------------
# All six solvers, e=3.  (Algorithm 4 = single-prime path only; no --multi.)
#             NAME       SRC             DATA_DIR               CBRT_DIR                                 RESULT_SUBPATH  JSON_PREFIX   EXTRA
# ---------------------------------------------------------------------------
run_variant  alg2_k16   alg2_k16.cpp    data/challenges/k16    $SLURM_SUBMIT_DIR/data/cbrt/alg2_k16_e3  alg2/k16        alg2_k16      ""
run_variant  alg2_k32   alg2_k32.cpp    data/challenges/k32    $SLURM_SUBMIT_DIR/data/cbrt/alg2_k32_e3  alg2/k32        alg2_k32      ""
run_variant  alg2_k35   alg2_k35.cpp    data/challenges/k35    $SLURM_SUBMIT_DIR/data/cbrt/alg2_k35_e3  alg2/k35        alg2_k35      ""
run_variant  alg3       alg3.cpp        data/challenges/alg3   $SLURM_SUBMIT_DIR/data/cbrt/alg3_e3      alg3/alg3       alg3          ""
run_variant  alg4       alg4.cpp        data/challenges/alg4   $SLURM_SUBMIT_DIR/data/cbrt/alg4_e3      alg4/alg4       alg4          ""
run_variant  alg7       alg7.cpp        data/challenges/alg3   $SLURM_SUBMIT_DIR/data/cbrt/alg7_e3      alg7/alg7       alg7_mpi      "--R 1048576"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}) complete."
echo "Results under data/results/aion/ (job $SLURM_JOB_ID):"
find $SLURM_SUBMIT_DIR/data/results/aion -type d -name "*_${SLURM_JOB_ID}_N${N_NODES}" 2>/dev/null | sed 's|.*/aion/|  |'
echo ""
echo "Use warm_avg.* (all timed runs averaged; no cold run) for analysis."
echo "Join each timing JSON with its _powm_ sibling for the baseline (results-join-methodology)."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"

# Edit: 2026-07-17
