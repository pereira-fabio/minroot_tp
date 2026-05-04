#!/bin/bash -l
#SBATCH --job-name=minroot_k16_mpi
#SBATCH --output=minroot_k16_mpi_%j.out
#SBATCH --error=minroot_k16_mpi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=02:00:00

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

# ---------------------------------------------------------------------------
# Configuration — only BENCH needs editing between runs
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/k16
BENCH=10

# k16_mpi: minimum useful size is 32 bits (need at least 2 x 16-bit primes)
SIZES="32 64 128 256 512 1024 2048 4096 8192"

# JSON output directory — automatically includes the node count so scaling
# runs never overwrite each other:
#   --nodes=1  ->  data/results/k16_mpi_N1/
#   --nodes=4  ->  data/results/k16_mpi_N4/
#   --nodes=8  ->  data/results/k16_mpi_N8/
JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k16_mpi_N${SLURM_JOB_NUM_NODES}
mkdir -p $JSON_DIR

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "Tasks (ranks): $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total workers: $(( SLURM_NTASKS * SLURM_CPUS_PER_TASK ))"
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
    -lgmp -o k16_mpi

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Helper: benchmark one challenge size across all three methods
#
# Usage: run_size <bits>
#
# Methods:
#   powm  -- sequential mpz_powm baseline (rank 0 only, no parallelism)
#   sieve -- generate 6542 primes inline per rank, no file I/O
#   table -- identical to sieve at k16 scale; kept for interface uniformity
#
# Precomputation (handled automatically inside k16_mpi):
#   data/k16_primes.bin         -- generated once, reused across all runs
#   data/modulo_16_<bits>.cbrt  -- cbrt(p) mod M for all 6542 primes,
#                                  precomputed and broadcast before any timed run
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_16_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_16_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found -- run gen_challenges.slurm first"
        echo "         (expected: $CHALLENGE)"
        return
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits"
    echo "------------------------------------------------------------"

    echo "  [powm]"
    srun --mpi=pmix ./k16_mpi \
        $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k16_mpi_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [sieve]"
    srun --mpi=pmix ./k16_mpi \
        $CHALLENGE $MODULO \
        --method sieve \
        --bench $BENCH \
        --json $JSON_DIR/k16_mpi_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [table]"
    srun --mpi=pmix ./k16_mpi \
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
echo "k16_mpi Benchmark  (N=${SLURM_JOB_NUM_NODES} nodes, $BENCH runs/method)"
echo "  methods: powm | sieve | table"
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
echo "All sizes complete."
echo "JSON files in: $JSON_DIR/"
ls $JSON_DIR/*_${SLURM_JOB_ID}.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "To compare scaling across node counts, resubmit with"
echo "#SBATCH --nodes=<N> -- results will land in k16_mpi_N<N>/."
echo ""
echo "Use warm_avg fields for all analysis."
echo "Job finished  (Job ID: $SLURM_JOB_ID)"
echo "============================================================"