#!/bin/bash -l
#SBATCH --job-name=minroot_alg2_k32
#SBATCH --output=minroot_alg2_k32_%A_%a.out
#SBATCH --error=minroot_alg2_k32_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# Job array — scaling study across node counts
#
# Each array task runs the full benchmark at a different node count:
#   task 0  ->  1 node   (2  ranks,  64 workers)
#   task 1  ->  2 nodes  (4  ranks, 128 workers)
#   task 2  ->  4 nodes  (8  ranks, 256 workers)
#   task 3  ->  8 nodes  (16 ranks, 512 workers)
#   task 4  -> 16 nodes  (32 ranks, 1024 workers)
#   task 5  -> 32 nodes  (64 ranks, 2048 workers)
#   task 6  -> 64 nodes  (128 ranks, 4096 workers)
#
# Submit with:
#   sbatch k32.slurm
#
# All seven tasks are submitted at once. SLURM schedules them as resources
# become available — you do not need to resubmit or change anything.
#
# Results land in separate directories per node count:
#   data/results/aion/alg2/k32_<jobid>_N1/   ...   data/results/aion/alg2/k32_<jobid>_N64/
#
# To run only specific tasks, e.g. tasks 2 and 3:
#   sbatch --array=2-3 k32.slurm
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

# ---------------------------------------------------------------------------
# Map array index -> node count
# ---------------------------------------------------------------------------
NODE_COUNTS=(1 2 4 8 16 32 64)
N_NODES=${NODE_COUNTS[$SLURM_ARRAY_TASK_ID]}
N_CHALLENGE_COUNTS=(10 15 20 30 40 50 50)
N_CHALLENGES=${N_CHALLENGE_COUNTS[$SLURM_ARRAY_TASK_ID]}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/k32

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/aion/alg2/k32_${SLURM_JOB_ID}_N${N_NODES}
mkdir -p $JSON_DIR

CBRT_DIR=$SLURM_SUBMIT_DIR/data/cbrt/alg2_k32_e3/N${N_NODES}
rm -rf $CBRT_DIR
mkdir -p $CBRT_DIR

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES"
echo "Tasks (ranks): $(( N_NODES * 2 ))"
echo "Total workers: $(( N_NODES * 2 * SLURM_CPUS_PER_TASK ))"
echo "Challenges:    $N_CHALLENGES"
echo "Sizes:         256 384 512 768 1024 2048"
echo "Results:       $JSON_DIR"
echo ""

cd $SLURM_SUBMIT_DIR
BINARY=./alg2_k32_N${N_NODES}
trap 'rm -f $BINARY' EXIT

# ---------------------------------------------------------------------------
# Compilation — each task compiles independently (safe, idempotent)
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common \
    src/aion/alg2_k32.cpp src/common/solver_common.cpp \
    src/common/common.cpp src/common/factorization.cpp \
    -lgmp -o alg2_k32_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (alg2_k32_N${N_NODES})"
echo ""

# ---------------------------------------------------------------------------
# Run all sizes via --all (one srun per method, avoids multiple-srun hangs)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "alg2_k32 Scaling Study  (N=${N_NODES} nodes, $N_CHALLENGES challenges/method)"
echo "  methods: powm (baseline, once) | table | sieve (--skip-powm-baseline)"
echo "  JSON  -> $JSON_DIR/"
echo "============================================================"
echo ""

echo "  [powm — baseline ($N_CHALLENGES challenges)]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg2_k32_N${N_NODES} \
    --all --data $DATA_DIR \
    --method powm \
    --n-challenges $N_CHALLENGES \
    --cbrt-dir $CBRT_DIR \
    --json $JSON_DIR/alg2_k32_powm_${SLURM_JOB_ID}
echo ""

echo "  [table]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg2_k32_N${N_NODES} \
    --all --data $DATA_DIR \
    --n-challenges $N_CHALLENGES \
    --method table \
    --skip-powm-baseline \
    --cbrt-dir $CBRT_DIR \
    --json $JSON_DIR/alg2_k32_table_${SLURM_JOB_ID}
echo ""

echo "  [sieve]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg2_k32_N${N_NODES} \
    --all --data $DATA_DIR \
    --n-challenges $N_CHALLENGES \
    --method sieve \
    --skip-powm-baseline \
    --cbrt-dir $CBRT_DIR \
    --json $JSON_DIR/alg2_k32_sieve_${SLURM_JOB_ID}
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}) complete."
echo "JSON files in: $JSON_DIR/"
ls $JSON_DIR/*_${SLURM_JOB_ID}_*.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Use warm_avg.* (runs_valid runs) for all analysis."
echo "runs_failed counts blinding-exhaustion outliers excluded from averages."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"

# Edit: 2026-07-06
