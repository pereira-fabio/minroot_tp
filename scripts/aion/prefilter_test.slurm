#!/bin/bash -l
#SBATCH --job-name=minroot_prefilter_test
#SBATCH --output=minroot_prefilter_test_%j.out
#SBATCH --error=minroot_prefilter_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00

# ===========================================================================
# Phase-0 correctness gate for the SIMPLE pre-filter (paper Fig. 6 / Algorithm 5).
#
# Builds tests/prefilter_test.cpp DIRECTLY with mpicxx (no Makefile — the basic
# slurm pattern used by the other AION scripts) and runs the gate at np = 1, 2, 4.
#
# The test is self-contained: it generates a 256-bit prime M and a random
# challenge from a fixed seed, builds the blinding table (|Re| = 4096), then
# checks that the parallel argmin (prefilter_simple_select) equals the single-rank
# brute-force argmin AND that the winning y_j is below the table mean. It needs no
# challenge files and no cbrt cache.
#
# PASS criteria: every np in {1, 2, 4} prints "PASS" with the SAME winning index j
# and returns 0.
#
# Submit:  sbatch prefilter_test.slurm
# ===========================================================================

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR

echo "=== building prefilter_test (direct mpicxx, no make) ==="
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common \
    tests/prefilter_test.cpp \
    src/common/prefilter.cpp src/common/factorization.cpp src/common/common.cpp \
    -lgmp -o prefilter_test
if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

BINARY=./prefilter_test
trap 'rm -f $BINARY' EXIT

FAIL=0
for np in 1 2 4; do
    echo "------------------------------------------------------------"
    echo "  prefilter_test  np=$np"
    echo "------------------------------------------------------------"
    srun --mpi=pmix -N 1 --ntasks=$np $BINARY
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  !! np=$np returned $rc (gate FAILED)"
        FAIL=1
    fi
    echo ""
done

echo "============================================================"
if [ $FAIL -eq 0 ]; then
    echo "prefilter_test: ALL np in {1,2,4} PASSED"
else
    echo "prefilter_test: FAILED — see output above"
fi
echo "The winning index j must be IDENTICAL across np=1/2/4 (parallel == brute force)."
echo "============================================================"
exit $FAIL

# Edit: 2026-07-17
