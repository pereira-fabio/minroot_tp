#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --qos=besteffort
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --output=gpu_scale_sim_%j.out
#SBATCH --error=gpu_scale_sim_%j.err

# ---------------------------------------------------------------------------
# How many GPUs would this attack actually need? (advisor meeting 2026-08-05)
#
# IRIS gives 4 GPUs. At 4 GPUs k32 reaches ~9x on an L40 against the CPU
# cluster's 52x, so the question is how many it would take to matter -- and
# whether more keep helping.
#
# This job answers it by MEASUREMENT rather than projection. For each simulated
# GPU count N it splits the prime table N ways, runs one node's worth of shards
# on the real cards, and times what that node actually costs:
#
#     factor       measured   the kernel on a 1/N shard, real cache behaviour
#     local_recon  measured   this host reconstructing its shards' factors
#     combine      measured   (nodes-1) full-width modmuls, timed here
#     gather       modelled   AION MPI_Gather; 0.2% of the solve, cannot be
#                             measured on one node and does not change the answer
#
# An earlier Python version extrapolated the factor term as W/N below the
# smallest measured slice. That is optimistic exactly where it matters -- a shard
# small enough for L2 is no longer memory-bound, so further splitting should stop
# paying -- and the gap between extrapolating and measuring reached 3x at 64 GPUs
# on synthetic data. Hence this.
#
# WHAT TO LOOK FOR
#   1. Where the speedup PEAKS. Past that point the serial combine grows faster
#      than the parallel phase shrinks and more GPUs make it slower. That is the
#      same ceiling the CPU cluster hits at 128 ranks, and finding it on the GPU
#      side too would make it a property of the ALGORITHM, not of either machine.
#   2. Whether the factor column keeps falling as 1/N past N=9, where k32's shard
#      first fits the L40's 96 MB L2. If it flattens there, residency has stopped
#      buying anything and the "shrink the table" reasoning of sub6:delta fails
#      for a second, independent reason.
# ---------------------------------------------------------------------------

set -u
module purge
module use /opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/modules/all
module load system/CUDA

cd "$SLURM_SUBMIT_DIR" || exit 1

# Paths follow the convention every other IRIS script uses (all_gpu_l40.slurm):
#   results ->  data/results/iris/<alg>/G<n>/l40
#   builds  ->  src/iris/build_<jobid>   (job-private, so concurrent jobs cannot
#                                         overwrite each other's binaries)
#   inputs  ->  data/challenges/k<K>/
N_GPUS=${SLURM_GPUS_ON_NODE:-4}
CHAL="data/challenges/k32"
OUTDIR="data/results/iris/gpu_scale_sim/G${N_GPUS}/l40"
BINDIR="src/iris/build_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR" "$BINDIR"

echo "Job:   $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x${N_GPUS}"
echo "In:    $CHAL"
echo "Out:   $OUTDIR"
echo ""

CHAL_SIZES="24576 8192"

NVCC_FLAGS="-arch=sm_89 -std=c++17 -O3 -I src/common -Xcompiler -fopenmp"
COMMON="src/common/common.cpp src/common/factorization.cpp"
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

echo -n "Building gpu_scale_sim_cuda ... "
nvcc src/iris/gpu_scale_sim_cuda.cu $COMMON $NVCC_FLAGS $LIBS \
     -o "$BINDIR/gpu_scale_sim_cuda" && echo "OK" || { echo "FAILED"; exit 1; }
echo ""

if [ ! -f data/k32_primes.bin ]; then
    echo "ERROR: data/k32_primes.bin missing"; exit 1
fi

# 24576 bits: the design point, where the sieve dominates and the projection is
# least polluted by fixed overhead. The split list brackets the L40's residency
# threshold (N=9 for k32) with several points on each side, and runs out to 1024
# so the combine-bound turnover is unambiguous rather than inferred from a tail.
SPLITS="1 2 4 8 9 12 16 24 32 48 64 96 128 192 256 384 512 1024"

for SIZE in 24576 8192; do
    echo "========================================"
    echo "N-GPU SCALING — k32, ${SIZE} bits"
    echo "========================================"
    "$BINDIR/gpu_scale_sim_cuda" \
        --data "$CHAL" --size "$SIZE" --k 32 \
        --splits "$SPLITS" --n-runs 5 --exponent 3 \
        --json "$OUTDIR/gpu_scale_sim_${SLURM_JOB_ID}_${SIZE}_k32.json"
    echo ""
done

# k30 is the largest L2-resident bound on an L40 at 4 GPUs
# (analysis/cache_residency.py). Running it here shows whether starting from a
# resident table changes the SHAPE of the scaling curve or only its offset.
if [ -f data/k30_primes.bin ]; then
    echo "========================================"
    echo "N-GPU SCALING — k30 (largest L2-resident bound), 24576 bits"
    echo "========================================"
    # k30 table against the k32 challenge set: --chal-k 32 names the set, --k 30
    # the table. Only factor_ms is meaningful here (a 2^30 table cannot factor a
    # 2^32-smooth challenge), which is the quantity this job is about.
    "$BINDIR/gpu_scale_sim_cuda" \
        --data "$CHAL" --size 24576 --k 30 --chal-k 32 \
        --splits "$SPLITS" --n-runs 5 --exponent 3 \
        --json "$OUTDIR/gpu_scale_sim_${SLURM_JOB_ID}_24576_k30.json"
else
    echo "k30 table absent — run tools/gen_prime_table 30 to include it."
fi

echo ""
echo "========================================"
echo "DONE — results in $OUTDIR"
echo "The peak row is the answer: past it, more GPUs make the attack slower."
echo "========================================"
