#!/bin/bash -l
#SBATCH --job-name=minroot_k35_mpi
#SBATCH --output=minroot_k35_mpi_%A_%a.out
#SBATCH --error=minroot_k35_mpi_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=64          
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# Job array — k35_mpi scaling study across node counts
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
# k35 covers primes up to 2^35 — paper's Algorithm 2 parameters.
# Breakeven vs sequential powm: ~17000 bits (sieve), ~12000 bits (table).
#
# Submit with:
#   sbatch k35.slurm
#
# All four tasks are submitted at once. SLURM schedules them as resources
# become available — you do not need to resubmit or change anything.
#
# Results land in separate directories per node count:
#   data/results/k35_mpi_N1/   data/results/k35_mpi_N2/
#   data/results/k35_mpi_N4/   data/results/k35_mpi_N8/
#
# Note: --method table generates data/k35_upper_primes.bin (~10 GB) on first run.
# To run only specific tasks, e.g. tasks 2 and 3:
#   sbatch --array=2-3 k35.slurm
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/k35
BENCH=50

# k35 breakeven vs powm is ~17000 bits; smaller sizes still useful for comparison
SIZES="256 512 1024 2048 4096 6144 8192 12288 16384 20480 24576"

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k35_mpi_N${N_NODES}
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
echo "Algorithm:     B = 2^35 (paper Algorithm 2)"
echo "Breakeven:     ~17000 bits (sieve), ~12000 bits (table)"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation — each task compiles independently (safe, idempotent)
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common \
    src/solvers/k35_mpi.cpp src/common/k_solver_common.cpp \
    src/common/k_common.cpp src/common/k_factorization.cpp \
    -lgmp -o k35_mpi_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (k35_mpi_N${N_NODES})"
echo ""

# ---------------------------------------------------------------------------
# Helper: benchmark one challenge size across all three methods
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_35_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_35_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found -- run challenge_gen.slurm first"
        return
    fi

    CACHE="data/cbrt/$(basename ${MODULO%.txt}).cbrt.0"
    if [ ! -f "$CACHE" ]; then
        echo "  NOTE: cbrt cache missing -- first run will do an untimed precompute pass"
    else
        echo "  cbrt cache: found ($(wc -c < "$CACHE") bytes)"
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits  |  Nodes: ${N_NODES}  (k35: B=2^35)"
    echo "------------------------------------------------------------"

    echo "  [powm]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k35_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k35_mpi_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [sieve]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k35_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method sieve \
        --bench $BENCH \
        --json $JSON_DIR/k35_mpi_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [table]  (generates data/k35_upper_primes.bin ~10 GB on first run)"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./k35_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method table \
        --bench $BENCH \
        --json $JSON_DIR/k35_mpi_table_${BITS}_${SLURM_JOB_ID}.json
    echo ""
}

# ---------------------------------------------------------------------------
# Run all sizes
# ---------------------------------------------------------------------------
echo "============================================================"
echo "k35_mpi Scaling Study  (N=${N_NODES} nodes, $BENCH runs/method)"
echo "  methods: powm | sieve | table"
echo "  JSON  -> $JSON_DIR/"
echo "  Breakeven: sieve ~17000 bits, table ~12000 bits (vs powm)"
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
