#!/bin/bash -l
#SBATCH --job-name=minroot_k32
#SBATCH --output=minroot_k32_%j.out
#SBATCH --error=minroot_k32_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=01:00:00

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

BENCH=5
DATA=data/challenges/k32
PRIMES=$SLURM_SUBMIT_DIR/k32_primes.bin

echo "=== Job info ==="
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "Tasks (ranks): $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total cores:   $(( SLURM_NTASKS * SLURM_CPUS_PER_TASK ))"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k32_mpi.cpp src/k_common.cpp \
    -lgmp -o k32_mpi

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Section 1 — Sieve method
#
# The segmented sieve generates primes on-the-fly during factoring.
# Each rank sieves its own [my_lo, my_hi] sub-range with a 512 KiB
# working set that stays in L2 cache. No pre-generated file needed.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 1: METHOD = SIEVE  ($BENCH runs, all challenge sizes)"
echo "============================================================"
echo ""

srun --mpi=pmix ./k32_mpi \
    --all \
    --data $DATA \
    --method sieve \
    --bench $BENCH

echo ""

# ---------------------------------------------------------------------------
# Section 2 — Prime table generation (one-time, rank 0 only)
#
# Generates all ~203 M primes in [65537, 2^32] and writes them to a
# sorted binary file (~813 MB). Cost: ~10 s. Subsequent runs reuse the
# file, amortising the I/O over all bench iterations.
# Each rank will later load only its own slice via binary search,
# so no rank reads the full 813 MB at query time.
# ---------------------------------------------------------------------------
if [ ! -f "$PRIMES" ]; then
    echo "============================================================"
    echo "SECTION 2: PRECOMPUTE PRIME TABLE"
    echo "============================================================"
    echo ""

    srun --mpi=pmix ./k32_mpi \
        --precompute \
        --primes $PRIMES

    echo ""
else
    echo "Prime table already exists at $PRIMES — skipping precompute."
    echo ""
fi

# ---------------------------------------------------------------------------
# Section 3 — Table method
#
# Each rank loads its own prime slice from the pre-built binary file.
# Trial division iterates a preloaded uint32_t array instead of running
# the sieve on-the-fly. The working set is larger (~range/nranks × 4 B)
# but the per-prime cost is lower because sieve marking is skipped.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 3: METHOD = TABLE  ($BENCH runs, all challenge sizes)"
echo "============================================================"
echo ""

srun --mpi=pmix ./k32_mpi \
    --all \
    --data $DATA \
    --method table \
    --primes $PRIMES \
    --bench $BENCH

echo ""
echo "============================================================"
echo "Job finished"
echo "============================================================"