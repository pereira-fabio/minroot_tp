#!/bin/bash -l
#SBATCH --job-name=minroot_cpu_scale_sim
#SBATCH --output=minroot_cpu_scale_sim_%j.out
#SBATCH --error=minroot_cpu_scale_sim_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# AION — CPU scaling simulation.
#
# ONE node answers what 512 would do. A rank of an N-rank job owns 1/N of the
# prime range and runs the same kernel on it, so timing that slice here IS the
# per-rank cost at scale; rank 0's combine is (N-1) full-width modmuls, which
# is arithmetic on one core and needs no cluster either. Only `gather` stays
# modelled, and the JSON reports both with and without it.
#
# --exclusive and 2 ranks/node reproduce the production layout, so the memory
# contention a real rank sees is present rather than assumed.
#
# This is a MEASUREMENT job, not a solve: no root is produced and no
# correctness gate applies (see the .cpp header). Run the real solver for that.
#
# Usage:
#   sbatch scripts/aion/cpu_scale_sim.slurm
#   sbatch --export=ALL,VARIANTS="k35",SIZES="24576" scripts/aion/cpu_scale_sim.slurm
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

cd $SLURM_SUBMIT_DIR

# k35 spans two prime tables (u32 up to 2^32, u64 above it) and a rank's slice
# can straddle the boundary; cpu_scale_sim reproduces alg2_k35's split and loads
# both, so --k 35 measures all of k35's primes rather than the lower seventh.
VARIANTS=${VARIANTS:-"k32 k35"}
SIZES=${SIZES:-"256 384 512 768 1024 2048 4096 8192 16384 24576"}
METHODS=${METHODS:-"sieve table"}
SPLITS=${SPLITS:-"1,2,4,8,16,32,64,128,192,256,384,512,768,1024,2048"}
EXPONENT=${EXPONENT:-3}
REPS=${REPS:-3}

# Each variant reads its OWN challenge set, as the solvers do: a k35 challenge
# is crafted 2^35-smooth and lives in data/challenges/k35. Using k32's set for a
# k35 run (the first version of this script) asks for challenge_0_35_* inside
# the k32 directory, which does not exist.
# The repo is inconsistent about where prime tables live -- alg2_k32.cpp defaults
# to "k32_primes.bin" (submit dir) while scripts/iris/ksweep.slurm uses
# "data/k32_primes.bin" -- so resolve it rather than assume, and say so if it is
# missing instead of launching sixteen jobs that each discover it separately.
find_table () {
    for p in "$@"; do [ -f "$p" ] && { echo "$p"; return 0; }; done
    return 1
}
PRIMES=$(find_table k32_primes.bin data/k32_primes.bin data/tables/k32_primes.bin) || {
    echo "FATAL: no k32 prime table found (looked in ., data/, data/tables/)"; exit 1; }
PRIMES_UPPER=$(find_table data/k35_upper_primes.bin k35_upper_primes.bin \
                          data/tables/k35_upper_primes.bin) || PRIMES_UPPER=""
OUTDIR=$SLURM_SUBMIT_DIR/data/results/aion/cpu_scale_sim/cpu_scale_sim_${SLURM_JOB_ID}
mkdir -p $OUTDIR

BINARY=./cpu_scale_sim_${SLURM_JOB_ID}
trap 'rm -f $BINARY' EXIT

mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common -I experiments/delta \
    experiments/scale_sim/cpu_scale_sim.cpp src/common/solver_common.cpp \
    src/common/common.cpp src/common/factorization.cpp \
    experiments/delta/delta_table.cpp \
    src/common/prefilter.cpp experiments/crt/crt_common.cpp \
    -lgmp -o $BINARY

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful ($BINARY)"

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Layout:    1 node, $SLURM_NTASKS_PER_NODE ranks x $SLURM_CPUS_PER_TASK threads"
echo "Variants:  $VARIANTS"
echo "Sizes:     $SIZES"
echo "Methods:   $METHODS"
echo "Splits:    $SPLITS"
echo "Primes:    $PRIMES${PRIMES_UPPER:+ + $PRIMES_UPPER}"
echo "Results:   $OUTDIR"
echo ""

for V in $VARIANTS; do
    KBITS=${V#k}
    CHAL=data/challenges/${V}
    if [ ! -d "$CHAL" ]; then
        echo "SKIP $V: no challenge set at $CHAL"; continue
    fi
    # k35's table method needs the upper (2^32,2^35] table as well as k32's.
    if [ "$KBITS" -gt 32 ] && [ -z "$PRIMES_UPPER" ]; then
        echo "SKIP $V: --method table needs k35_upper_primes.bin, not found"; continue
    fi
    UPPER_ARG=""
    [ -n "$PRIMES_UPPER" ] && UPPER_ARG="--primes-upper $PRIMES_UPPER"
    for SIZE in $SIZES; do
        for M in $METHODS; do
            echo "---- $V $M ${SIZE}b ----"
            srun --mpi=pmix --ntasks=$SLURM_NTASKS $BINARY \
                 --data $CHAL --primes $PRIMES $UPPER_ARG \
                 --size $SIZE --k $KBITS --chal-k $KBITS --method $M \
                 --splits $SPLITS --reps $REPS --exponent $EXPONENT \
                 --json $OUTDIR/cpu_scale_sim_${SLURM_JOB_ID}_${SIZE}_${V}_${M}_e${EXPONENT}.json \
                 || echo "  ($V $M ${SIZE}b failed -- continuing)"
            echo ""
        done
    done
done

echo "=== done ==="
echo "Analyse with:  python3 analysis/thesis_figures.py content/figures/"

