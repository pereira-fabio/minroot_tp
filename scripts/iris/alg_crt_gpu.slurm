#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --output=alg_crt_gpu_%j.out
#SBATCH --error=alg_crt_gpu_%j.err

# ---------------------------------------------------------------------------
# IRIS GPU — CRT solver (Option A: GPU residues + host reconstruction).
# See wiki/planning/gpu-crt-implementation-plan-2026-07-14.md
#
# Single GPU for now (Option A). Sweeps ALPHAS x canonical sizes; one JSON per
# (alpha, size). alpha is encoded in the filename (a<ALPHA>) so
# analysis/crt_best_alpha.py can group by it. The CRT JSON is self-contained
# (correct inverse c^b baseline recorded in it).
#
# Reuses the alg3 challenge/modulo files (CRT works for any challenge). Copy them
# from AION first (same as all_gpu.slurm prerequisites).
#
#   Override the alpha list:  sbatch --export=ALL,ALPHAS="4 16" scripts/iris/alg_crt_gpu.slurm
#   WALL TIME: runtime ~x(#alphas); large moduli at small alpha are the slow
#   corner (~|M|/log2(alpha) steps, each a host reconstruction). Trim ALPHAS /
#   N_CHALLENGES / SIZES or raise --time if it times out.
# ---------------------------------------------------------------------------

SIZES=${SIZES:-"256 384 512 768 1024 2048 4096 8192 16384 24576"}
ALPHAS=${ALPHAS:-"2 4 8 16 32"}
RECONS=${RECONS:-"host gpu"}     # host = Option A (host reconstruct); gpu = Option B (GPU MAC)
EXPONENT=${EXPONENT:-3}
N_CHALLENGES=${N_CHALLENGES:-10}
DATA_DIR=data/challenges/alg3

cd $SLURM_SUBMIT_DIR

module load env/release/2023b
module load system/CUDA/12.6.0
module load math/GMP/6.3.0-GCCcore-13.2.0   # adjust name if the link step fails (module spider GMP)

BASE_DIR=$SLURM_SUBMIT_DIR/data/results/iris/alg_crt

echo "=== alg_crt_cuda sweep (1 GPU) ==="
echo "Job:      $SLURM_JOB_ID"
echo "Recons:   $RECONS   (host = Option A, gpu = Option B)"
echo "Sizes:    $SIZES"
echo "Alphas:   $ALPHAS"
echo "Exponent: $EXPONENT   Challenges: $N_CHALLENGES"
echo ""

# ---- Build (direct nvcc — same flags as all_gpu.slurm; no make) ----
# Self-contained CRT context; links common + factorization only (MPI-free).
NVCC_FLAGS="-arch=sm_70 -std=c++17 -O3 -I src/common -Xcompiler -fopenmp"
COMMON="src/common/common.cpp src/common/factorization.cpp"
LIBS="-lgmp -lgomp"
BIN=src/iris/alg_crt_cuda
echo -n "Building alg_crt_cuda ... "
nvcc src/iris/alg_crt_cuda.cu $COMMON $NVCC_FLAGS $LIBS -o "$BIN"
if [ $? -ne 0 ]; then echo "FAILED"; exit 1; fi
echo "OK"

# recon modes in separate dirs so analysis/crt_best_alpha.py can group each cleanly.
for RECON in $RECONS; do
    JSON_DIR=$BASE_DIR/G1/$RECON
    mkdir -p $JSON_DIR
    for SIZE in $SIZES; do
        for A in $ALPHAS; do
            echo "  ---- recon=$RECON  size=$SIZE  alpha=$A ----"
            $BIN --data $DATA_DIR --size $SIZE \
                 --alpha $A --exponent $EXPONENT --recon $RECON \
                 --n-challenges $N_CHALLENGES \
                 --json $JSON_DIR/alg_crt_a${A}_crt_${SLURM_JOB_ID}_${SIZE}.json
        done
    done
done

echo ""
echo "=== done. Compare A vs B with: ==="
echo "  python3 analysis/crt_best_alpha.py $BASE_DIR/G1/host   # Option A"
echo "  python3 analysis/crt_best_alpha.py $BASE_DIR/G1/gpu    # Option B"

# Edit: 2026-07-16
