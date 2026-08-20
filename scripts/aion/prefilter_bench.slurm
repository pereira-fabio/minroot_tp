#!/bin/bash -l
#SBATCH --job-name=minroot_prefilter_bench
#SBATCH --output=results/prefilter/aion/prefilter_bench_%j.out
#SBATCH --error=results/prefilter/aion/prefilter_bench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# SLURM does not create --output directories; make it before the job writes anything.
mkdir -p results/prefilter/aion

# ===========================================================================
# Is the simple pre-filter PROFITABLE?  (thesis sub4:prefilter)
#
# prefilter_test answers "is the parallel selection CORRECT?".  This answers
# the question the thesis needs: does one selection round cost less than the
# sieving it saves?
#
#   pre-filtering pays  iff  C_pf < (2^b - 1) * C_sieve
#
# where C_pf is one selection round (measured here), C_sieve is one full
# B-smooth sieve, and b is the bits of SEARCH saved (~2 for N=512, B=2^35).
#
# ---------------------------------------------------------------------------
# TWO PHASES, because the two quantities have different sampling needs.
#
#   Phase 1 (statistics).  The bit saving is a property of |Re| ALONE -- it does
#   not depend on the modulus size or on the rank partitioning.  Run 1 proved the
#   np-invariance exactly (identical savings at np=1/2/4).  So measure it ONCE,
#   at a cheap modulus, with MANY trials to get tight error bars.  100 trials at
#   256 bits costs seconds.
#
#   Phase 2 (profitability).  C_pf DOES grow with modulus size, and so does
#   C_sieve, so the ratio must be taken at MATCHED sizes -- run 1 compared a
#   256-bit C_pf against a 24576-bit C_sieve, which is not a fair ratio.  Here
#   each size is paired with the measured k32 sieve cost at that same size.
#   Only a few trials are needed: C_pf is a timing, not a distribution.
#
# C_sieve references below are k32 *table*-method factor_ms at N1 (= 2 MPI
# ranks), from results/aion/alg2/k32_*_N1/.  The table method is the FASTER of
# the two, so using it is the conservative choice: it makes the pre-filter's
# margin look smaller than the sieve method would.  Phase 2 therefore runs at
# np=2, to match the 2-rank reference.
#
# SAFETY: builds a STANDALONE binary linked only against prefilter.cpp /
# factorization.cpp / common.cpp.  No solver is rebuilt, so nothing under
# results/{aion,iris} can change.
# ===========================================================================

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

set -e
cd "$SLURM_SUBMIT_DIR"

echo "=== building prefilter_bench (direct mpicxx, no make) ==="
mpicxx -O3 -std=c++17 -fopenmp -Wall -I src/common -I experiments/delta -I experiments/delta \
       tests/prefilter_bench.cpp \
       src/common/prefilter.cpp \
       src/common/factorization.cpp \
       src/common/common.cpp \
       -lgmp -o prefilter_bench
echo "Compilation successful"
echo ""

# ---------------------------------------------------------------------------
# PHASE 1 — bit-saving curve + np-invariance check (cheap modulus, many trials)
# ---------------------------------------------------------------------------
echo "############################################################"
echo "# PHASE 1: bit saving vs |Re|   (256-bit modulus, 100 trials)"
echo "#   Expect saving ~ log2|Re| + 1.33:  +0.83 from E[-log2 U],"
echo "#   U ~ Beta(1,|Re|), plus ~0.5 from mpz_sizeinbase rounding."
echo "#   The saving columns MUST be identical across np."
echo "############################################################"
echo ""
for np in 1 2 4; do
  echo "------------------------------------------------------------"
  echo "  PHASE 1  np=${np}"
  echo "------------------------------------------------------------"
  srun -n $np ./prefilter_bench --bits 256 --trials 100 --sieve-ms 74.838
  echo ""
done

# ---------------------------------------------------------------------------
# PHASE 2 — C_pf at MATCHED sizes, against the k32 sieve cost at that size
# ---------------------------------------------------------------------------
echo "############################################################"
echo "# PHASE 2: matched-size profitability   (np=2, 5 trials)"
echo "#   C_sieve = k32 table-method factor_ms at N1 (2 ranks),"
echo "#   same modulus size. Read the C_pf/C_s column: pre-filtering"
echo "#   pays while it stays below (2^b - 1) ~ 3."
echo "############################################################"
echo ""

# size:C_sieve(ms) — k32 table factor_ms at N1, measured.
# ALL TEN canonical sizes, so the paper-scale set {256,384,512,768,1024,2048} is
# complete: that is the regime where pre-filtering is cheapest relative to the
# sieve, so it is the half of the curve that actually matters for this technique.
PAIRS="256:74.838 384:88.441 512:99.629 768:123.871 1024:131.454 2048:180.008 \
       4096:235.546 8192:364.115 16384:585.180 24576:789.647"

# k35 pairing (see thesis tab:prefilter-coupling): C_pf does NOT depend on which
# algorithm you would attach the pre-filter to -- only the divisor changes. So the
# k35 ratio is obtainable from the same run as C_pf / C_sieve(k35), no re-run
# needed. k35 table factor_ms at N1, same sizes:
#   256:337.5  384:417.9  512:507.1   768:697.3   1024:961.2
#   2048:1310.3 4096:1723.4 8192:2712.9 16384:4663.0 24576:6748.1

for pair in $PAIRS; do
  BITS="${pair%%:*}"
  CSIEVE="${pair##*:}"
  echo "------------------------------------------------------------"
  echo "  PHASE 2  |M|=${BITS} bits   C_sieve=${CSIEVE} ms   np=2"
  echo "------------------------------------------------------------"
  srun -n 2 ./prefilter_bench --bits "$BITS" --trials 5 --sieve-ms "$CSIEVE"
  echo ""
done

echo "============================================================"
echo "PHASE 1: saving must track log2|Re| + ~1.33 and must NOT"
echo "         change with np."
echo "PHASE 2: C_pf/C_s is now a MATCHED ratio. Pre-filtering is"
echo "         profitable wherever that column stays below ~3."
echo "============================================================"
