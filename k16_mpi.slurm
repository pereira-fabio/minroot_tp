#!/bin/bash -l
#SBATCH --job-name=minroot_k16_mpi
#SBATCH --output=minroot_k16_mpi_%j.out
#SBATCH --error=minroot_k16_mpi_%j.err
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
DATA_DIR=data/challenges/k16
BENCH=10

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k16_mpi
mkdir -p $JSON_DIR

# k16_mpi: minimum useful size is 32 bits (need at least 2 × 16-bit primes)
# With 16 MPI ranks × 32 OMP threads = 512 workers splitting 6542 primes,
# the communication overhead may dominate for small sizes — this is intentional
# and scientifically interesting to measure.
SIZES="32 64 128 256 512 1024 2048 4096 8192"

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
echo "Note: k16_mpi partitions 6542 primes across $SLURM_NTASKS ranks"
echo "      (~$(( 6542 / SLURM_NTASKS )) primes/rank). For small challenge sizes"
echo "      MPI communication overhead will likely exceed compute time — this"
echo "      is the expected crossover point and the key result to observe."
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    src/k16_mpi.cpp src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k16_mpi

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Helper: benchmark one challenge size across all three methods
#
# Usage: run_size <bits>
#
# Runs: powm  — sequential mpz_powm baseline (rank 0 only, no parallelism)
#       sieve — generate 6542 primes inline, partition across ranks
#       table — identical to sieve at k16 scale (both fit in L1 cache);
#               kept for interface uniformity with k32_mpi
#
# cbrt cache:
#   Built on first run and saved to modulo_16_<bits>.cbrt (single file,
#   not per-rank like k32, since all ranks share the same dense table).
#   First run computes all 6542 cbrt entries via powm — slow reconstruction.
#   Resubmit after first run for fully warm cbrt timings.
#
# JSON layout:
#   cold_run    — run 0, one-time synchronisation cost at first MPI barrier
#   warm_runs   — runs 1..N, steady state — use warm_avg for all analysis
#   warmup_runs — 2 silent warmup iterations before the bench loop
#   prime_stats — primes tested/found per rank (from first warm run)
#   thread_balance — primes tested/found per OMP thread on rank 0
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

    # Check cbrt cache
    CACHE=$(echo "$MODULO" | sed 's/\.txt$/.cbrt/')
    if [ ! -f "$CACHE" ]; then
        echo "  NOTE: cbrt cache missing for ${BITS}-bit modulo ($CACHE)"
        echo "        First table/sieve run will compute all 6542 cbrt entries."
        echo "        Resubmit for fully warm reconstruction timings."
    else
        ENTRIES=$(head -1 "$CACHE" 2>/dev/null)
        echo "  cbrt cache: found ($ENTRIES entries)"
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
echo "k16_mpi Benchmark — ${BENCH} runs per method per size"
echo "  methods: powm | sieve | table"
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
echo "Key comparisons to make with Python script:"
echo "  k16 (single node) vs k16_mpi (8 nodes) — same sizes, table method"
echo "  k16_mpi sieve vs table — should be ~identical (both fit in L1 cache)"
echo "  prime_stats.found_per_rank — load balance across MPI ranks"
echo "  thread_balance — load balance across OMP threads within rank 0"
echo ""
echo "Use warm_avg fields for all analysis."
echo "cold_run and warmup_runs are labelled separately."
echo "Job finished  (Job ID: $SLURM_JOB_ID)"