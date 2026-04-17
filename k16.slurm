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

# Challenge sizes — mirrors the k32/k64 suite for cross-solver comparison
SIZES="1024 2048 5000 7000"

echo "=== Job info ==="
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total cores:   $SLURM_CPUS_PER_TASK"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation — single-node OpenMP only, no MPI needed
# ---------------------------------------------------------------------------
g++ -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k16.cpp src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k16

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Section 1 — Compare: all three methods in one pass per challenge size
#
# --method compare runs the following in a single bench loop per iteration:
#
#   [1] powm (sequential baseline)
#       challenge^e mod M with a single mpz_powm — the unparallelisable
#       O(log M) squaring chain. This is what the ASIC executes each round.
#
#   [2] factor (parallel table-lookup reconstruction)
#       Factor challenge into 16-bit primes, reconstruct using precomputed
#       cbrt(q) mod M table. Reconstruction is OpenMP-parallel.
#       The cbrt table is built once per modulo and cached to disk.
#
#   [3] powm-recon (sequential powm per factor)
#       Same factoring as [2], but reconstruction calls mpz_powm(p, e, M)
#       for each factor instead of a table lookup. Isolates raw per-factor
#       powm cost without any table benefit.
#
# Output per size includes a speedup table:
#   [1] vs [2] recon only  <-- Algorithm 2 speedup (key number for report)
#   [1] vs [2] total       <-- end-to-end speedup including factoring
#   [1] vs [3] recon only  <-- how much of the gain is parallelism alone
#   [2] vs [3] recon only  <-- additional gain from the precomputed table
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 1: METHOD = COMPARE  ($BENCH runs, all challenge sizes)"
echo "           [1] powm  [2] factor  [3] powm-recon"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    CHALLENGE=$DATA/challenge_16_${SIZE}.txt
    MODULO=$DATA/modulo_16_${SIZE}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "Skipping size $SIZE — files not found ($CHALLENGE)"
        continue
    fi

    echo "--- challenge size: ${SIZE} bits ---"
    ./k16 $CHALLENGE $MODULO \
        --method compare \
        --bench $BENCH
    echo ""
done

# ---------------------------------------------------------------------------
# Section 2 — Individual methods (for clean isolated timing data)
#
# Running the methods separately avoids the small overhead of the compare
# loop running all three back-to-back in the same process. The numbers
# should be essentially identical, but having clean single-method runs
# is useful for tables in the report.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 2: METHOD = factor (isolated, $BENCH runs)"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    CHALLENGE=$DATA/challenge_16_${SIZE}.txt
    MODULO=$DATA/modulo_16_${SIZE}.txt
    [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ] && continue

    echo "--- ${SIZE} bits ---"
    ./k16 $CHALLENGE $MODULO --method factor --bench $BENCH
    echo ""
done

echo "============================================================"
echo "SECTION 3: METHOD = powm (isolated, $BENCH runs)"
echo "           Sequential baseline — no factoring, no parallelism"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    CHALLENGE=$DATA/challenge_16_${SIZE}.txt
    MODULO=$DATA/modulo_16_${SIZE}.txt
    [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ] && continue

    echo "--- ${SIZE} bits ---"
    ./k16 $CHALLENGE $MODULO --method powm --bench $BENCH
    echo ""
done

echo "============================================================"
echo "SECTION 4: METHOD = powm-recon (isolated, $BENCH runs)"
echo "           Factor + sequential powm per factor, no table"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    CHALLENGE=$DATA/challenge_16_${SIZE}.txt
    MODULO=$DATA/modulo_16_${SIZE}.txt
    [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ] && continue

    echo "--- ${SIZE} bits ---"
    ./k16 $CHALLENGE $MODULO --method powm-recon --bench $BENCH
    echo ""
done

echo "============================================================"
echo "Job finished"
echo "============================================================"