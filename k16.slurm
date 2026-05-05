#!/bin/bash -l
#SBATCH --job-name=minroot_k16_mpi
#SBATCH --output=minroot_k16_mpi_%A_%a.out
#SBATCH --error=minroot_k16_mpi_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=8          
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=02:00:00

# ---------------------------------------------------------------------------
# Job array — k16_mpi scaling study across node counts
#
# Each array task runs the full benchmark at a different node count:
#   task 0  ->  1 node   (2  ranks,  64 workers)
#   task 1  ->  2 nodes  (4  ranks, 128 workers)
#   task 2  ->  4 nodes  (8  ranks, 256 workers)
#   task 3  ->  8 nodes  (16 ranks, 512 workers)
#
# With only 6542 primes total, adding more ranks beyond a certain point will
# show diminishing returns as MPI communication overhead exceeds compute time.
# This scaling study lets you find that crossover point empirically.
#
# Submit with:
#   sbatch k16.slurm
#
# Results land in:
#   data/results/k16_mpi_N1/   data/results/k16_mpi_N2/
#   data/results/k16_mpi_N4/   data/results/k16_mpi_N8/
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
DATA_DIR=data/challenges/k16
BENCH=10
SIZES="32 64 128 256 512 1024 2048 4096 8192"

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k16_mpi_N${N_NODES}
mkdir -p $JSON_DIR

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES"
echo "Primes/rank:   $(( 6542 / (N_NODES * 2) ))  (6542 total / $(( N_NODES * 2 )) ranks)"
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
    src/k16_mpi.cpp src/k_solver_common.cpp \
    src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k16_mpi_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (k16_mpi_N${N_NODES})"
echo ""

# ---------------------------------------------------------------------------
# Helper: benchmark one challenge size across all three methods
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_16_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_16_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found -- run gen_challenges.slurm first"
        return
    fi

    CACHE=$(echo "$MODULO" | sed "s|data/challenges/k16/|data/|" | sed 's/\.txt$/.cbrt/')
    if [ ! -f "$CACHE" ]; then
        echo "  NOTE: cbrt cache missing -- first run will precompute all 6542 entries"
    else
        echo "  cbrt cache: found ($(wc -c < "$CACHE") bytes)"
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits  |  Nodes: ${N_NODES}  |  Primes/rank: $(( 6542 / (N_NODES * 2) ))"
    echo "------------------------------------------------------------"

    echo "  [powm]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k16_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k16_mpi_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [sieve]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k16_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method sieve \
        --bench $BENCH \
        --json $JSON_DIR/k16_mpi_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [table]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k16_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method table \
        --bench $BENCH \
        --json $JSON_DIR/k16_mpi_table_${BITS}_${SLURM_JOB_ID}.json
    echo ""
}

# ---------------------------------------------------------------------------
# Run all sizes
# ---------------------------------------------------------------------------
echo "============================================================"
echo "k16_mpi Scaling Study  (N=${N_NODES} nodes, $BENCH runs/method)"
echo "  methods: powm | sieve | table"
echo "  JSON  -> $JSON_DIR/"
echo "  Key question: at what node count does MPI overhead exceed compute?"
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
echo "Use warm_avg.* for all analysis."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"