#!/bin/bash -l
#SBATCH --job-name=minroot_k32_mpi
#SBATCH --output=minroot_k32_mpi_%A_%a.out
#SBATCH --error=minroot_k32_mpi_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# Job array — scaling study across node counts
#
# Each array task runs the full benchmark at a different node count:
#   task 0  ->  1 node   (2  ranks,  64 workers)
#   task 1  ->  2 nodes  (4  ranks, 128 workers)
#   task 2  ->  4 nodes  (8  ranks, 256 workers)
#   task 3  ->  8 nodes  (16 ranks, 512 workers)
#
# Submit with:
#   sbatch k32.slurm
#
# All four tasks are submitted at once. SLURM schedules them as resources
# become available — you do not need to resubmit or change anything.
#
# Results land in separate directories per node count:
#   data/results/k32_mpi_N1/   data/results/k32_mpi_N2/
#   data/results/k32_mpi_N4/   data/results/k32_mpi_N8/
#
# To run only specific sizes, e.g. tasks 2 and 3:
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
NODE_COUNTS=(1 2 4 8)
N_NODES=${NODE_COUNTS[$SLURM_ARRAY_TASK_ID]}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/k32
BENCH=10
SIZES="64 128 256 512 1024 2048 4096 8192"

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k32_mpi_N${N_NODES}
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
# Compilation — each task compiles independently (safe, idempotent)
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    src/k32_mpi.cpp src/k_solver_common.cpp \
    src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k32_mpi_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (k32_mpi_N${N_NODES})"
echo ""

# ---------------------------------------------------------------------------
# Helper: benchmark one challenge size across all three methods
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_32_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_32_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found -- run gen_challenges.slurm first"
        return
    fi

    CACHE=$(echo "$MODULO" | sed "s|data/challenges/k32/|data/|" | sed 's/\.txt$/.cbrt.0/')
    if [ ! -f "$CACHE" ]; then
        echo "  NOTE: cbrt cache missing -- first run will do an untimed precompute pass"
    else
        echo "  cbrt cache: found ($(wc -c < "$CACHE") bytes)"
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits  |  Nodes: ${N_NODES}"
    echo "------------------------------------------------------------"

    echo "  [powm]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k32_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k32_mpi_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [table]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k32_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method table \
        --bench $BENCH \
        --json $JSON_DIR/k32_mpi_table_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [sieve]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k32_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method sieve \
        --bench $BENCH \
        --json $JSON_DIR/k32_mpi_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""
}

# ---------------------------------------------------------------------------
# Run all sizes
# ---------------------------------------------------------------------------
echo "============================================================"
echo "k32_mpi Scaling Study  (N=${N_NODES} nodes, $BENCH runs/method)"
echo "  methods: powm | table | sieve"
echo "  JSON  -> $JSON_DIR/"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    run_size $SIZE
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}) complete."
echo "JSON files in: $JSON_DIR/"
ls $JSON_DIR/*_${SLURM_JOB_ID}.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Use warm_avg.* (runs_valid runs) for all analysis."
echo "runs_failed counts blinding-exhaustion outliers excluded from averages."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"