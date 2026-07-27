#!/bin/bash -l
#SBATCH --job-name=minroot_prefilter_bench
#SBATCH --output=results/prefilter/aion/prefilter_bench_%j.out
#SBATCH --error=results/prefilter/aion/prefilter_bench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

# SLURM does not create --output directories; make it before the job writes anything.
mkdir -p results/prefilter/aion

# ===========================================================================
# Is the simple pre-filter PROFITABLE?  (thesis sub4:prefilter)
#
# prefilter_test.slurm answers "is the parallel selection CORRECT?".  This
# answers the separate question the thesis actually needs: does one selection
# round cost less than the sieving it saves?
#
#   pre-filtering pays  iff  C_pf < (2^b - 1) * C_sieve
#
# where C_pf is one selection round (measured here), C_sieve is one full
# B-smooth sieve, and b is the bits of SEARCH saved.  For N=512, B=2^35 the
# standard estimate is b ~ 2, so the threshold is C_pf < 3 * C_sieve.
#
# It also turns the single 13-bit data point from prefilter_test into a curve:
# the saving is measured over --trials independent draws at each |Re|, so the
# log2|Re| prediction is tested across three orders of magnitude rather than at
# one point.
#
# SAFETY: this builds a STANDALONE binary linked only against prefilter.cpp /
# factorization.cpp / common.cpp.  No solver is rebuilt, so no benchmark result
# under results/{aion,iris} can change.
#
# C_sieve default (790 ms) is the measured k32 single-rank factor_ms at 24576
# bits.  Override for other regimes, e.g. --sieve-ms 8 for the 2048-bit N64 case.
# ===========================================================================

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

set -e
cd "$SLURM_SUBMIT_DIR"

echo "=== building prefilter_bench (direct mpicxx, no make) ==="
mpicxx -O3 -std=c++17 -fopenmp -Wall -I src/common \
       tests/prefilter_bench.cpp \
       src/common/prefilter.cpp \
       src/common/factorization.cpp \
       src/common/common.cpp \
       -lgmp -o prefilter_bench
echo "Compilation successful"
echo ""

# Sweep rank counts as well: the selection is an MPI argmin, so C_pf should fall
# with ranks while the reported bit saving stays put (it is a property of |Re|,
# not of the partitioning).  If the saving MOVES with np, the partitioning is wrong.
for np in 1 2 4; do
  echo "------------------------------------------------------------"
  echo "  prefilter_bench  np=${np}"
  echo "------------------------------------------------------------"
  srun -n $np ./prefilter_bench --bits 256 --trials 20 --sieve-ms 790
  echo ""
done

echo "============================================================"
echo "Read the C_pf/C_s column against the break-even threshold."
echo "The bit-saving columns must NOT change with np."
echo "============================================================"

# Edit: 2026-07-27
