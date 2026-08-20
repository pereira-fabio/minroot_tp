#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --qos=besteffort
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --output=ksweep_%j.out
#SBATCH --error=ksweep_%j.err

# ---------------------------------------------------------------------------
#
# THE QUESTION
# ------------
# The thesis explains every GPU result with one principle: residency decides
# everything. k16's 26 KB table sits in L1 and a single V100 runs its sieve step
# 800x faster than sequential; k32's 813 MB misses every cache and loses to the
# CPU cluster. But those are two samples of a continuum and the boundary between
# them has never been measured -- only predicted.
#
# analysis/cache_residency.py predicts it from prime counts alone: on an L40 at
# 4 GPUs the largest L2-resident bound is k30 (54.4 MB/GPU of 96 MB); k31 is the
# first that does not fit (105 MB/GPU). This job tests whether that predicted
# boundary is visible in the TIMINGS, which is the part arithmetic cannot settle.
#
# WHAT TO LOOK FOR
# ----------------
# The per-prime cost, printed as "ns per prime per pass". If residency matters,
# that figure should be roughly flat while the slice fits in L2 and then jump
# between k30 and k31 as the slice spills to GDDR6. If it rises smoothly instead,
# residency is NOT the step-change the thesis's k16-vs-k32 contrast implies, and
# the delta-encoding result (sub6:delta -- residency achieved, only 1.10x gained)
# was the first hint of exactly that.
#
# Either outcome is worth having. A visible step confirms the model; a smooth
# curve says the k16 win comes from having little WORK, not from residency, which
# is what sub6:delta already argues.
#
# WHY A SEPARATE BINARY
# ---------------------
# alg2_ksweep_cuda writes algorithm "ksweep" into its own results directory and
# never uses the alg2_k32_ filename prefix. A --k flag on the k32 solver was
# tried on 2026-08-07 and reverted: thesis_figures.py matches that prefix, so a
# k20 run would have been silently loaded as k32 data. Keep them apart.
# ---------------------------------------------------------------------------

set -u
module purge
module use /opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/modules/all
module load system/CUDA

cd "$SLURM_SUBMIT_DIR" || exit 1

# Paths follow all_gpu_l40.slurm:
#   results ->  data/results/iris/<alg>/G<n>/l40
#   builds  ->  build/gpu_<jobid>
#   inputs  ->  data/challenges/k<K>/
#
# The k32 challenge set is used for EVERY table bound. That is deliberate: a
# challenge is crafted smooth for one bound, and no k20..k31 sets exist, so
# sweeping the table against a fixed set is the only option. It is also the
# right one -- the kernel scans its shard whatever the challenge is, so
# factor_ms (the residency signal) is valid throughout. End-to-end speedup is
# not, and the binary prints "n/a" for it rather than a wrong number whenever
# the table bound is below the challenge's.
N_GPUS=${SLURM_GPUS_ON_NODE:-4}
CHAL="data/challenges/k32"
OUTDIR="data/results/iris/ksweep/G${N_GPUS}/l40"
BINDIR="build/gpu_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR" "$BINDIR"

echo "Job:          $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "GPUs:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x${N_GPUS}"
echo "In:           $CHAL  (k32 set, used for every table bound — see note above)"
echo "Out:          $OUTDIR"
echo "PURPOSE:      does the predicted L2-residency boundary show up in timings?"
echo ""

CHAL_SIZES="24576"

# ---- Build ----------------------------------------------------------------
NVCC_FLAGS="-arch=sm_89 -std=c++17 -O3 -I src/common -I src/common/gpu -I experiments/delta -Xcompiler -fopenmp"
COMMON="src/common/common.cpp src/common/factorization.cpp experiments/delta/delta_table.cpp"
LIBS="-lgmp -lgomp"

# ---- Challenge set ---------------------------------------------------------
# Generated here if absent so the job does not depend on what happens to be on
# disk. Naming is fixed by tools/challenges.c:
#     data/challenges/k<K>/{modulo_<K>_<bits>.txt, challenge_<n>_<K>_<bits>.txt}
#
# Cost warning: the FIRST call at a given size has to find a random prime of that
# width, which at 24576 bits takes a few minutes. It is cached afterwards -- the
# tool reloads an existing modulo rather than regenerating -- so later sizes and
# later jobs are quick.
echo -n "Building challenges tool ... "
gcc -O2 -fopenmp tools/challenges.c -lgmp -o "$BINDIR/challenges" \
    && echo "OK" || { echo "FAILED"; exit 1; }

for SZ in $CHAL_SIZES; do
    if [ -f "$CHAL/challenge_0_32_${SZ}.txt" ] && [ -f "$CHAL/modulo_32_${SZ}.txt" ]; then
        echo "challenges ${SZ}b: present"
    else
        echo "challenges ${SZ}b: generating (first run at this size is slow) ..."
        "$BINDIR/challenges" 10 "$SZ" 32 || { echo "challenge generation FAILED"; exit 1; }
    fi
done
echo ""

echo -n "Building gen_prime_table ... "
gcc -O2 tools/gen_prime_table.c -o "$BINDIR/gen_prime_table" \
    && echo "OK" || { echo "FAILED"; exit 1; }

echo -n "Building alg2_ksweep_cuda ... "
nvcc experiments/ksweep/alg2_ksweep_cuda.cu $COMMON $NVCC_FLAGS $LIBS -o "$BINDIR/alg2_ksweep_cuda" \
    && echo "OK" || { echo "FAILED"; exit 1; }
echo ""

# ---- Prime tables ---------------------------------------------------------
# Sizes rise steeply: k31 is 420 MB and takes ~10 s, everything below k28 is
# under a second. Generated once here, reused by every run below; k17..k31 come
# to ~870 MB on disk, with k32's 813 MB already shipped.
#
# EVERY k from 17 to 32 (was "20 24 26 28 29 30 31 32", coarse at the bottom on
# the assumption that nothing happens there). It does: k20 turned out to sit on
# an OCCUPANCY cliff -- 17.48 ns/prime against a 2.7 ns steady state -- and with
# the old spacing that cliff was a single point with a 4x gap to its neighbour,
# so where it starts and where it recovers were both unmeasured. k17..k23 map it.
# The extra eight values are cheap: 10.7M primes against the old list's 411M,
# so the sweep runs under 3% longer.
KS="17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32"

echo "========================================"
echo "PRIME TABLES"
echo "========================================"
for K in $KS; do
    if [ "$K" = "32" ] && [ -f data/k32_primes.bin ]; then
        echo "k32: using the shipped data/k32_primes.bin"
        continue
    fi
    [ -f "data/k${K}_primes.bin" ] && { echo "k${K}: already present"; continue; }
    "$BINDIR/gen_prime_table" "$K" || exit 1
done
echo ""

# ---- Sweep ----------------------------------------------------------------
# 24576 bits only. The residency question is about the TABLE, which does not
# change with modulus size; running every size would multiply the job time
# without adding a data point to the question being asked. 24576b is also where
# the sieve dominates most, so the per-prime cost is measured most cleanly.
SIZE=24576
NRUNS=10

# ---------------------------------------------------------------------------
# EXPERIMENT 1 (the clean one) — vary the SPLIT at fixed k.
#
# Total work is constant; only the shard size moves. Per-prime cost therefore
# isolates residency: flat while the shard streams from GDDR6, dropping once it
# fits in L2. k32's 813 MB table crosses the L40's 96 MB L2 at a 9-way split, so
# the sweep brackets that with room on both sides.
#
# Every run uses ONE physical GPU and times ONE shard, which is exactly what one
# of N GPUs would cost. That is what lets a 4-GPU allocation speak about 256.
# ---------------------------------------------------------------------------
SPLITS="1 2 4 8 9 12 16 32 64 128 256"

for K in 32 30; do
    echo "========================================"
    echo "EXPERIMENT 1 — k=$K, split sweep, ${SIZE}b, 1 GPU"
    echo "========================================"
    for SP in $SPLITS; do
        JSON="$OUTDIR/ksweep_table_${SLURM_JOB_ID}_${SIZE}_k${K}_s${SP}_e3.json"
        echo "--- k=$K, 1/$SP of the table ---"
        "$BINDIR/alg2_ksweep_cuda" \
            --k "$K" --chal-k 32 --data "$CHAL" --size "$SIZE" \
            --n-challenges "$NRUNS" --n-gpus 1 --split "$SP" \
            --exponent 3 --json "$JSON" || echo "  (k=$K split=$SP failed — continuing)"
        echo ""
    done
done

# ---------------------------------------------------------------------------
# EXPERIMENT 2 — vary k at a fixed split, for the hit-rate/table-size trade.
#
# Confounded by design (k changes work AND size together), so it does NOT settle
# the residency question on its own. It is here because it answers a different
# one: which k gives the best speedup at a realistic device count, which is what
# a real attacker would pick.
# ---------------------------------------------------------------------------
for NG in 4 1; do
    echo "========================================"
    echo "EXPERIMENT 2 — k sweep, $NG GPU(s), ${SIZE}b"
    echo "========================================"
    for K in $KS; do
        JSON="$OUTDIR/ksweep_table_${SLURM_JOB_ID}_${SIZE}_k${K}_g${NG}_e3.json"
        echo "--- k=$K, $NG GPU(s) ---"
        "$BINDIR/alg2_ksweep_cuda" \
            --k "$K" --chal-k 32 --data "$CHAL" --size "$SIZE" \
            --n-challenges "$NRUNS" --n-gpus "$NG" \
            --exponent 3 --json "$JSON" || echo "  (k=$K failed — continuing)"
        echo ""
    done
done

echo "========================================"
echo "DONE — analyse with:"
echo "  python3 analysis/cache_residency.py --measured $OUTDIR"
echo "  (N-GPU scaling is a separate job: scripts/iris/gpu_scale_sim.slurm)"
echo "========================================"
