#!/bin/bash -l
#SBATCH --job-name=minroot_k32
#SBATCH --output=minroot_k32_%j.out
#SBATCH --error=minroot_k32_%j.err
#SBATCH --nodes=8
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
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/k32
PRIMES=$SLURM_SUBMIT_DIR/k32_primes.bin
BENCH=10

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k32
mkdir -p $JSON_DIR

# All target sizes to benchmark
# k32: minimum useful size is 64 bits (need at least 2 × 32-bit primes to factor)
SIZES="64 128 256 512 1024 2048 4096 8192"

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "Tasks (ranks): $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total cores:   $(( SLURM_NTASKS * SLURM_CPUS_PER_TASK ))"
echo "Bench runs:    $BENCH"
echo "Sizes:         $SIZES"
echo "Results:       $JSON_DIR"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ ! -f "$PRIMES" ]; then
    echo "ERROR: prime table not found: $PRIMES"
    echo "       Run: srun ./k32_mpi --precompute --primes $PRIMES"
    exit 1
fi

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
# Helper: benchmark one challenge size across all three methods
#
# Usage: run_size <bits>
#
# Runs: powm  — sequential mpz_powm baseline (rank 0 only, no parallelism)
#       table — precomputed 813 MB prime file, one slice per rank
#       sieve — on-the-fly segmented sieve, no file I/O
#
# All three methods run inside a single srun call to avoid MPI reinit hangs
# from multiple srun calls in one job.
#
# cbrt cache note:
#   The first run for a new modulo computes cbrt(p) via powm for each found
#   prime and saves it to modulo_32_<bits>.cbrt.<rank>. Subsequent runs load
#   the cache and reconstruction costs ~0 ms. If the cache is missing the
#   first bench iteration will be slower — this is captured in cold_run.
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_32_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_32_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found — run gen_challenges.slurm first"
        echo "         (expected: $CHALLENGE)"
        return
    fi

    # Check cbrt cache for this modulo (rank 0 cache file)
    CACHE=$(echo "$MODULO" | sed 's/\.txt$/.cbrt.0/')
    if [ ! -f "$CACHE" ]; then
        echo "  NOTE: cbrt cache missing for ${BITS}-bit modulo."
        echo "        First table/sieve run will build it (~slow reconstruction)."
        echo "        Resubmit this job after first run for fully warm timings."
    else
        echo "  cbrt cache: found ($CACHE)"
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits"
    echo "------------------------------------------------------------"

    echo "  [powm]"
    srun --mpi=pmix ./k32_mpi \
        $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k32_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [table]"
    srun --mpi=pmix ./k32_mpi \
        $CHALLENGE $MODULO \
        --method table \
        --primes $PRIMES \
        --bench $BENCH \
        --json $JSON_DIR/k32_table_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [sieve]"
    srun --mpi=pmix ./k32_mpi \
        $CHALLENGE $MODULO \
        --method sieve \
        --bench $BENCH \
        --json $JSON_DIR/k32_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""
}

# ---------------------------------------------------------------------------
# Run all sizes
# ---------------------------------------------------------------------------
echo "============================================================"
echo "k32 Benchmark — ${BENCH} runs per method per size"
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