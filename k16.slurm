#!/bin/bash -l
#SBATCH --job-name=minroot_k16
#SBATCH --output=minroot_k16_%j.out
#SBATCH --error=minroot_k16_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=02:00:00

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/k16
BENCH=10

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k16
mkdir -p $JSON_DIR

# All target sizes to benchmark — lower and upper bounds of parallelisation benefit
# k16: minimum useful size is 32 bits (need at least 2 × 16-bit primes to factor)
SIZES="32 64 128 256 512 1024 2048 4096 8192"

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Bench runs:    $BENCH"
echo "Sizes:         $SIZES"
echo "Results:       $JSON_DIR"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
g++ -O3 -std=c++17 -fopenmp -Wall \
    src/k16.cpp src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k16

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Helper: benchmark one challenge size across all three methods
#
# Usage: run_size <bits>
#
# Runs: powm  — sequential mpz_powm baseline (single core, no parallelism)
#       table — precomputed prime array + cbrt table (the attack)
#       sieve — inline Eratosthenes + cbrt table (same attack, no file I/O)
#
# JSON layout (cold_run / warm_runs split):
#   cold_run   — run 0, cold CPU cache, labelled — do NOT use for comparison
#   warm_runs  — runs 1..N, steady state — use warm_avg for all analysis
#   warmup_runs — 2 silent warmup iterations before the bench loop
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_16_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_16_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found — run gen_challenges.slurm first"
        echo "         (expected: $CHALLENGE)"
        return
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits"
    echo "------------------------------------------------------------"

    echo "  [powm]"
    ./k16 $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k16_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [table]"
    ./k16 $CHALLENGE $MODULO \
        --method table \
        --bench $BENCH \
        --json $JSON_DIR/k16_table_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [sieve]"
    ./k16 $CHALLENGE $MODULO \
        --method sieve \
        --bench $BENCH \
        --json $JSON_DIR/k16_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""
}

# ---------------------------------------------------------------------------
# Run all sizes
# ---------------------------------------------------------------------------
echo "============================================================"
echo "k16 Benchmark — ${BENCH} runs per method per size"
echo "  methods: powm | table | sieve"
echo "  JSON  → $JSON_DIR/"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    run_size $SIZE
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "All sizes complete.  JSON files:"
ls $JSON_DIR/*_${SLURM_JOB_ID}.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Use warm_avg fields for all analysis."
echo "cold_run and warmup_runs are labelled separately."
echo "Job finished  (Job ID: $SLURM_JOB_ID)"
echo "============================================================"