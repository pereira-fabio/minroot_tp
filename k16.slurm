#!/bin/bash -l
#SBATCH --job-name=minroot_k16
#SBATCH --output=minroot_k16_%j.out
#SBATCH --error=minroot_k16_%j.err
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
export OMP_WAIT_POLICY=active

BENCH=5
DATA=data/challenges/k16

# Challenge sizes to run — mirrors the k32/k64 suite for direct comparison
SIZES="1024 2048 5000 7000"

echo "=== Job info ==="
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total cores:   $SLURM_CPUS_PER_TASK"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation
# k16 is single-node OpenMP only — no MPI needed
# ---------------------------------------------------------------------------
g++ -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k16.cpp src/k_common.cpp \
    -lgmp -o k16

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Section 1 — METHOD = factor  (default, production path)
#
# Precomputes cbrt(q) mod M for all 6542 primes q <= 2^16 once per modulo,
# persists the table to disk (<modulo>.cbrt), then reconstructs the cube
# root via table lookups + multiplications — no powm on large numbers.
# The cbrt table is built on the first run for each modulo and cached.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 1: METHOD = factor  ($BENCH runs, all challenge sizes)"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    CHALLENGE=$DATA/challenge_2_16_${SIZE}.txt
    MODULO=$DATA/modulo_16_${SIZE}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "Skipping size $SIZE — files not found ($CHALLENGE)"
        continue
    fi

    echo "--- challenge size: ${SIZE} bits ---"
    ./k16 $CHALLENGE $MODULO \
        --method factor \
        --bench $BENCH
    echo ""
done

# ---------------------------------------------------------------------------
# Section 2 — METHOD = powm  (naive Fermat baseline)
#
# Computes challenge^e mod M directly with a single large modular
# exponentiation (e = 3^-1 mod M-1). This is the sequential O(log M)
# squaring baseline — no parallelism, no factorization.
# Used to quantify the speedup from the factorization approach.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 2: METHOD = powm  ($BENCH runs, all challenge sizes)"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    CHALLENGE=$DATA/challenge_2_16_${SIZE}.txt
    MODULO=$DATA/modulo_16_${SIZE}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "Skipping size $SIZE — files not found"
        continue
    fi

    echo "--- challenge size: ${SIZE} bits ---"
    ./k16 $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH
    echo ""
done

# ---------------------------------------------------------------------------
# Section 3 — METHOD = powm-recon  (isolate reconstruction cost)
#
# Factors the challenge first (like METHOD_FACTOR), then reconstructs
# using on-the-fly powm per factor instead of precomputed table lookups.
# Purpose: isolates the cost of each powm call so the supervisor can
# compare the table-lookup speedup against raw powm-per-factor.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 3: METHOD = powm-recon  ($BENCH runs, all challenge sizes)"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    CHALLENGE=$DATA/challenge_2_16_${SIZE}.txt
    MODULO=$DATA/modulo_16_${SIZE}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "Skipping size $SIZE — files not found"
        continue
    fi

    echo "--- challenge size: ${SIZE} bits ---"
    ./k16 $CHALLENGE $MODULO \
        --method powm-recon \
        --bench $BENCH
    echo ""
done

echo "============================================================"
echo "Job finished"
echo "============================================================"