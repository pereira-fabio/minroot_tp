#!/bin/bash -l
#SBATCH --job-name=minroot_k32
#SBATCH --output=minroot_k32_%j.out
#SBATCH --error=minroot_k32_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=01:30:00

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
    src/k32_mpi.cpp src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k32_mpi

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# Section 1 — Sieve method
#
# Segmented sieve generates primes on-the-fly. Each rank sieves its own
# [my_lo, my_hi] sub-range with a 512 KiB L2-resident working set.
# No pre-generated file needed.
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
# ~203 M primes written to a sorted binary file (~813 MB).
# Each rank later loads only its own slice via binary search.
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
# Each rank reads its own prime slice from the pre-built file.
# Larger working set than sieve but no per-segment marking overhead.
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

# ---------------------------------------------------------------------------
# Section 4 — Compare: distributed sieve + sequential powm baseline
#
# --method compare runs the full distributed sieve across all ranks (same
# as Section 1), then rank 0 runs a plain sequential mpz_powm(challenge, e, M)
# for the same bench_count iterations with all other ranks idle.
#
# This produces the key comparison your supervisor asked for:
#
#   "powm vs reconstruction only" — parallel independent powm calls on
#   small factors vs one sequential powm on the full challenge.
#   This directly measures the Algorithm 2 speedup (Biryukov et al.).
#
# Look for this line in the output:
#   powm vs reconstruction only:  X.XXx  (parallel cbrt, no factoring cost)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "SECTION 4: METHOD = COMPARE  ($BENCH runs, all challenge sizes)"
echo "           (distributed sieve + sequential powm baseline on rank 0)"
echo "============================================================"
echo ""

srun --mpi=pmix ./k32_mpi \
    --all \
    --data $DATA \
    --method compare \
    --bench $BENCH

echo ""
echo "============================================================"
echo "Job finished"
echo "============================================================"