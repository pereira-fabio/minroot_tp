#!/bin/bash -l
#SBATCH --job-name=minroot_k16_test
#SBATCH --output=minroot_k16_test_%j.out
#SBATCH --error=minroot_k16_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=00:30:00

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# ---------------------------------------------------------------------------
# Configuration — edit these variables before submitting
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/k16
BENCH=50

# JSON output directory
JSON_DIR=$SLURM_SUBMIT_DIR/results
mkdir -p $JSON_DIR

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Bench runs:    $BENCH  (run 0 = cold, runs 1-9 = warm, reported separately)"
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
# Helper: run one challenge size, both methods, with JSON output
#
# Usage: run_size <bits>  e.g. run_size 2048
#
# Note on cold-start:
#   Run 0 hits a cold CPU cache (prime table not yet in L1/L2, branch
#   predictor unpopulated). It is 3-4x slower than runs 1-9 for small
#   moduli. The JSON separates it into "cold_run" vs "warm_runs" so the
#   warm average is uncontaminated. The printed stdout average includes
#   run 0 and is therefore slightly pessimistic — use the JSON warm_avg
#   numbers for all analysis and thesis tables.
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_2_16_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_16_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] $BITS-bit challenge not found ($CHALLENGE)"
        return
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits"
    echo "------------------------------------------------------------"

    echo "  [factor]"
    ./k16 $CHALLENGE $MODULO \
        --method factor \
        --bench $BENCH \
        --json $JSON_DIR/k16_factor_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [powm baseline]"
    ./k16 $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k16_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

}

# ---------------------------------------------------------------------------
# Run all standard modulus sizes
# ---------------------------------------------------------------------------
echo "============================================================"
echo "k16 Benchmark — all sizes, $BENCH runs each"
echo "  JSON → $JSON_DIR/k16_*_${SLURM_JOB_ID}.json"
echo "============================================================"
echo ""

run_size 1024
run_size 2048
run_size 5000
run_size 7000

echo "============================================================"
echo "JSON results written to: $JSON_DIR/"
echo "  k16_factor_{size}_${SLURM_JOB_ID}.json"
echo "  k16_powm_{size}_${SLURM_JOB_ID}.json"
echo ""
echo "Use warm_avg.* fields for all analysis — cold_run is labelled"
echo "separately and should not be used for performance comparison."
echo "Job finished"
echo "============================================================"
