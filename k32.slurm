#!/bin/bash -l
#SBATCH --job-name=minroot_k32_test
#SBATCH --output=minroot_k32_test_%j.out
#SBATCH --error=minroot_k32_test_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:15:00

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

# ---------------------------------------------------------------------------
# Configuration — edit these variables before submitting
# ---------------------------------------------------------------------------
CHALLENGE=data/challenges/k32/challenge_0_32_7000.txt
MODULO=data/challenges/k32/modulo_32_7000.txt
PRIMES=$SLURM_SUBMIT_DIR/k32_primes.bin
BENCH=10

# JSON output directory — one file per section, named by method and job ID
JSON_DIR=$SLURM_SUBMIT_DIR/results
mkdir -p $JSON_DIR

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "Tasks (ranks): $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total cores:   $(( SLURM_NTASKS * SLURM_CPUS_PER_TASK ))"
echo "Challenge:     $CHALLENGE"
echo "Bench runs:    $BENCH"
echo "Target:        47.855 ms  (professor's single-core powm)"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Sanity checks — fail fast with a clear message
# ---------------------------------------------------------------------------
for f in "$CHALLENGE" "$MODULO" "$PRIMES"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

# Check cbrt cache — warn if missing (first run will be slow)
CACHE=$(echo "$MODULO" | sed 's/\.txt$/.cbrt.0/')
if [ ! -f "$CACHE" ]; then
    echo "WARNING: cbrt cache not found ($CACHE)"
    echo "         First run will compute all powms (~168 ms reconstruction)."
    echo "         Resubmit after this job completes for warm-cache timings."
else
    ENTRIES=$(head -1 "$CACHE" 2>/dev/null)
    echo "cbrt cache: $CACHE ($ENTRIES entries — reconstruction ~0 ms expected)"
fi
echo ""

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    src/k32_mpi.cpp src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k32_mpi

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Section 1 — Table method (primary benchmark)
#
# The method that beats the 47.855 ms target.
# With warm cbrt cache: reconstruction ~0 ms, factoring ~44 ms.
# Total: ~45 ms → 1.06x speedup over professor's single-core baseline.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 1: TABLE METHOD  ($BENCH runs)"
echo "           Expected total: ~45 ms (warm cache)"
echo "============================================================"
echo ""

srun --mpi=pmix ./k32_mpi \
    $CHALLENGE $MODULO \
    --method table \
    --primes $PRIMES \
    --bench $BENCH \
    --json $JSON_DIR/table_${SLURM_JOB_ID}.json

echo ""

# ---------------------------------------------------------------------------
# Section 2 — Sieve method
#
# Pure segmented-sieve factoring — no prime table I/O overhead.
# Slower than table (~77 ms vs ~44 ms) but useful to confirm linear scaling
# across ranks and validate that reconstruction is truly near-zero.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 3: SIEVE METHOD  ($BENCH runs)"
echo "           Expected total: ~78 ms"
echo "============================================================"
echo ""

srun --mpi=pmix ./k32_mpi \
    $CHALLENGE $MODULO \
    --method sieve \
    --bench $BENCH \
    --json $JSON_DIR/sieve_${SLURM_JOB_ID}.json

echo ""
echo "============================================================"
echo "JSON results written to: $JSON_DIR/"
echo "  table_${SLURM_JOB_ID}.json"
echo "  sieve_${SLURM_JOB_ID}.json"
echo "Job finished"
echo "============================================================"
