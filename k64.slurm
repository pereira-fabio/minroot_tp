#!/bin/bash -l
#SBATCH --job-name=minroot_k64_scaling
#SBATCH --output=minroot_k64_scaling_%A_%a.out
#SBATCH --error=minroot_k64_scaling_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=8          # allocate max; each task uses only N_NODES via srun -N
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# Job array — k64_mpi scaling study across node counts
#
# Each task maps to a different node count:
#   task 0  ->  1 node   (2  ranks,  64 workers)
#   task 1  ->  2 nodes  (4  ranks, 128 workers)
#   task 2  ->  4 nodes  (8  ranks, 256 workers)
#   task 3  ->  8 nodes  (16 ranks, 512 workers)
#
# k64 scaling behaviour differs from k32:
#   - Trial division (Phase A) scales well with rank count (same as k32 sieve)
#   - Pollard's rho (Phase C) runs on rank 0 only with OMP threads — does NOT
#     scale with rank count, only with threads per rank.
#   - So increasing nodes helps Phase A but not Phase C.
#   - For challenges with many small factors and few large factors (common),
#     Phase C is negligible and scaling mirrors k32.
#   - For challenges with large 64-bit factors, Phase C may dominate and
#     multi-node scaling will show diminishing returns.
#
# Submit with:
#   sbatch k64_scaling.slurm
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
NODE_COUNTS=(1 2 4 8)
N_NODES=${NODE_COUNTS[$SLURM_ARRAY_TASK_ID]}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/k64
BENCH=10
SIZES="1024 2048 4096 5120 6144 7168 8192"

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k64_mpi_N${N_NODES}
mkdir -p $JSON_DIR

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES"
echo "Tasks (ranks): $(( N_NODES * 2 ))"
echo "Total workers: $(( N_NODES * 2 * SLURM_CPUS_PER_TASK ))"
echo "Bench runs:    $BENCH"
echo "Sizes:         $SIZES"
echo "Results:       $JSON_DIR"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    src/k64_mpi.cpp src/k_solver_common.cpp \
    src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k64_mpi_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (k64_mpi_N${N_NODES})"
echo ""

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_64_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_64_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found"
        return
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits  |  Nodes: ${N_NODES}"
    echo "------------------------------------------------------------"

    echo "  [powm]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k64_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k64_mpi_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [rho]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k64_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method rho \
        --bench $BENCH \
        --json $JSON_DIR/k64_mpi_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""
}

# ---------------------------------------------------------------------------
# Run all sizes
# ---------------------------------------------------------------------------
echo "============================================================"
echo "k64_mpi Scaling Study  (N=${N_NODES} nodes, $BENCH runs/method)"
echo "  JSON  -> $JSON_DIR/"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    run_size $SIZE
done

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}) complete."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"