#!/bin/bash -l
#SBATCH --job-name=minroot_k16_e5_mpi
#SBATCH --output=minroot_k16_e5_mpi_%A_%a.out
#SBATCH --error=minroot_k16_e5_mpi_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# Job array — k16_mpi exponent-5 (5th root) scaling study
#
# Identical structure to k16.slurm but uses --exponent 5 and modulo files
# with the _e5 suffix (gcd(5, M-1) = 1 guaranteed by challenge_gen_e5.slurm).
# Challenge files are exponent-agnostic and reused from the e3 run.
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
# Prerequisites:
#   sbatch challenge_gen_e5.slurm   (generates modulo_16_<bits>_e5.txt)
#
# Submit with:
#   sbatch k16_e5.slurm
#
# Results land in:
#   data/results/alg2/k16_e5_<jobid>_N<nodes>/
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
DATA_DIR=data/challenges/k16

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/alg2/k16_e5_${SLURM_JOB_ID}_N${N_NODES}
mkdir -p $JSON_DIR

CBRT_DIR=$SLURM_SUBMIT_DIR/data/cbrt/alg2_k16_e5/N${N_NODES}
rm -rf $CBRT_DIR
mkdir -p $CBRT_DIR

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES"
echo "Exponent:      5 (5th root)"
echo "Primes/rank:   $(( 6542 / (N_NODES * 2) ))  (6542 total / $(( N_NODES * 2 )) ranks)"
echo "Tasks (ranks): $(( N_NODES * 2 ))"
echo "Total workers: $(( N_NODES * 2 * SLURM_CPUS_PER_TASK ))"
echo "Challenges:    $N_CHALLENGES"
echo "Sizes:         256 384 512 768 1024 2048"
echo "Results:       $JSON_DIR"
echo ""

cd $SLURM_SUBMIT_DIR
BINARY=./k16_e5_mpi_N${N_NODES}
trap 'rm -f $BINARY' EXIT

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common \
    src/aion/alg2_k16.cpp src/common/solver_common.cpp \
    src/common/common.cpp src/common/factorization.cpp \
    -lgmp -o k16_e5_mpi_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (k16_e5_mpi_N${N_NODES})"
echo ""

# ---------------------------------------------------------------------------
# Run all sizes via --all (one srun per method, avoids multiple-srun hangs)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "alg2_k16 e5 Scaling Study  (N=${N_NODES} nodes, $N_CHALLENGES challenges/method)"
echo "  exponent: 5 (5th root / actual MinRoot VDF)"
echo "  methods:  powm (baseline, once) | sieve | table (--skip-powm-baseline)"
echo "  JSON  ->  $JSON_DIR/"
echo "============================================================"
echo ""

echo "  [powm — baseline ($N_CHALLENGES challenges)]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k16_e5_mpi_N${N_NODES} \
    --all --data $DATA_DIR \
    --modulo-suffix _e5 \
    --exponent 5 \
    --method powm \
    --n-challenges $N_CHALLENGES \
    --cbrt-dir $CBRT_DIR \
    --json $JSON_DIR/alg2_k16_powm_${SLURM_JOB_ID}
echo ""

echo "  [sieve]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k16_e5_mpi_N${N_NODES} \
    --all --data $DATA_DIR \
    --modulo-suffix _e5 \
    --exponent 5 \
    --n-challenges $N_CHALLENGES \
    --method sieve \
    --skip-powm-baseline \
    --cbrt-dir $CBRT_DIR \
    --json $JSON_DIR/alg2_k16_sieve_${SLURM_JOB_ID}
echo ""

echo "  [table]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k16_e5_mpi_N${N_NODES} \
    --all --data $DATA_DIR \
    --modulo-suffix _e5 \
    --exponent 5 \
    --n-challenges $N_CHALLENGES \
    --method table \
    --skip-powm-baseline \
    --cbrt-dir $CBRT_DIR \
    --json $JSON_DIR/alg2_k16_table_${SLURM_JOB_ID}
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}) complete."
echo "JSON files in: $JSON_DIR/"
ls $JSON_DIR/*_${SLURM_JOB_ID}_*.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Use warm_avg.* for all analysis."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"

# Edit: 2026-06-25
