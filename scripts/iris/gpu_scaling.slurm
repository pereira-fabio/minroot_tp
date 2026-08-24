#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --qos=besteffort
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --job-name=gpu_scaling
#SBATCH --output=gpu_scaling_%j.out
#SBATCH --error=gpu_scaling_%j.err

# ---------------------------------------------------------------------------
# IRIS GPU strong-scaling sweep — G1/G2/G3/G4 in ONE job, on ONE node.
#
# WHY THIS EXISTS
# ---------------
# The GPU scaling data on the NAS cannot be used for a scaling claim:
#
#   1. Cold cbrt cache. Every file in the old G1-G4 batch shows recon_ms of
#      1118-1567 ms at 24576b against 53 ms warm. That is the bug fixed by the
#      two-pass warm loop (386b62a). Those runs predate the fix.
#
#   2. Mixed hardware. G1/G2/G3 came from V100 jobs (55244xx), G4 from L40 jobs
#      (5543207+). A curve measured across different GPUs is not a curve.
#
#   3. No k32 at all. Every k32 GPU run was skipped because data/k32_primes.bin
#      is absent on IRIS (data/ left git in 924f436). Note that
#      `gen_prime_table --all` deliberately SKIPS k32 with a stale message
#      ("already ships with the solvers" -- it no longer does). The preflight
#      below calls `gen_prime_table 32` explicitly, which is the working form.
#
# HOW IT FIXES 2
# --------------
# Rather than submitting four jobs that may land on different nodes, this
# allocates 4 GPUs once and varies the *number the solver uses* via --n-gpus.
# Every point in the curve therefore runs on the same node, the same four
# devices, the same host CPU and the same binary. Only the device count moves.
#
# USAGE
#   sbatch scripts/iris/gpu_scaling.slurm
#
#   # subset, if the queue will not give you 24h in one block:
#   sbatch --export=ALL,GPU_LEVELS="1 2" scripts/iris/gpu_scaling.slurm
#   sbatch --export=ALL,GPU_LEVELS="3 4" scripts/iris/gpu_scaling.slurm
#
# Results: data/results/iris/<alg>/G<n>/<alg>_<method>_<jobid>_<size>_e3.json
# Same layout as all_gpu.slurm, so existing analysis scripts read it unchanged.
# ---------------------------------------------------------------------------

cd $SLURM_SUBMIT_DIR

SIZES="256 384 512 768 1024 2048 4096 8192 16384 24576"
N_CHALLENGES=10
EXPONENT=3
GPU_LEVELS="${GPU_LEVELS:-1 2 3 4}"

# ---- Modules ---------------------------------------------------------------
module load env/release/2023b
module load system/CUDA/12.6.0

if module load math/GMP/6.2.1-GCCcore-13.2.0 2>/dev/null; then
    echo "GMP loaded: math/GMP/6.2.1-GCCcore-13.2.0"
elif module load math/GMP/6.3.0-GCCcore-13.2.0 2>/dev/null; then
    echo "GMP loaded: math/GMP/6.3.0-GCCcore-13.2.0"
else
    echo "GMP module not found — available versions:"
    module spider GMP 2>&1 | grep -i "math/GMP"
    exit 1
fi

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# ---- Preflight 1: the warm-cache fix must be present -----------------------
# Without it the whole sweep reproduces the cold-cache numbers and is wasted
# allocation. Cheaper to check now than to discover it in the JSON later.
if ! grep -q "gpu_reset_warm_accumulators" src/common/gpu/gpu_common.h 2>/dev/null; then
    echo "FATAL: gpu_reset_warm_accumulators() not found in src/common/gpu/gpu_common.h"
    echo "       This checkout predates the warm-cache fix (386b62a)."
    echo "       Run: git pull   (the fix is on main)"
    exit 1
fi
echo "PREFLIGHT: warm-cache fix present."

# ---- Preflight 2: k32 prime table ------------------------------------------
# ~813 MB. Without it every k32 run is silently skipped -- which is exactly why
# k32 is missing from all existing GPU results.
if [ ! -f data/k32_primes.bin ]; then
    echo "PREFLIGHT: data/k32_primes.bin missing — generating."
    if [ ! -x ./gen_prime_table ]; then
        make gen_prime_table || { echo "FATAL: gen_prime_table build failed"; exit 1; }
    fi
    mkdir -p data
    # NOTE: `--all` skips k32 on purpose; the explicit form is required.
    ./gen_prime_table 32 data/k32_primes.bin || { echo "FATAL: generation failed"; exit 1; }
    echo "PREFLIGHT: generated $(du -h data/k32_primes.bin | cut -f1)."
else
    echo "PREFLIGHT: data/k32_primes.bin present ($(du -h data/k32_primes.bin | cut -f1))."
fi

# ---- Preflight 3: challenges ------------------------------------------------
for d in k16 k32 k35 alg3 alg4; do
    [ -d "data/challenges/$d" ] || echo "PREFLIGHT WARNING: data/challenges/$d missing — those runs will be skipped."
done
echo ""

# ---- Build once, reuse for every G level -----------------------------------
BINDIR="build/gpu_scaling_${SLURM_JOB_ID}"
mkdir -p "$BINDIR"

NVCC_FLAGS="-arch=sm_70 -std=c++17 -O3 -I src/common -I src/common/gpu -I experiments/delta -Xcompiler -fopenmp"
COMMON="src/common/common.cpp src/common/factorization.cpp experiments/delta/delta_table.cpp"
LIBS="-lgmp -lgomp"

echo "======================================== BUILD"
build_one() {
    echo -n "  $2 ... "
    if nvcc src/iris/$1 $COMMON $NVCC_FLAGS $LIBS -o "$BINDIR/$2" 2>&1; then echo OK; else echo FAILED; return 1; fi
}
build_one alg2_k16_cuda.cu alg2_k16_cuda
build_one alg2_k32_cuda.cu alg2_k32_cuda
build_one alg2_k35_cuda.cu alg2_k35_cuda
build_one alg3_cuda.cu     alg3_cuda
build_one alg4_cuda.cu     alg4_cuda
build_one alg7_cuda.cu     alg7_cuda
echo ""

# ---- One G level -----------------------------------------------------------
# $1 = number of GPUs the solver may use (allocation is always 4).
run_level() {
    local NG=$1
    local A2="data/results/iris/alg2/G${NG}"
    local A3="data/results/iris/alg3/G${NG}"
    local A4="data/results/iris/alg4/G${NG}"
    local A7="data/results/iris/alg7/G${NG}"
    mkdir -p "$A2" "$A3" "$A4" "$A7"

    echo ""
    echo "############################################################"
    echo "#  G${NG} — solver uses ${NG} of 4 allocated L40S"
    echo "############################################################"

    for SIZE in $SIZES; do
        # ---- alg2 k16 (batch split; table fits constant memory) ----
        if [ -x "$BINDIR/alg2_k16_cuda" ] && [ -d data/challenges/k16 ]; then
            "$BINDIR/alg2_k16_cuda" --data data/challenges/k16 --size $SIZE \
                --method powm --exponent $EXPONENT --n-challenges 1 \
                --json "$A2"/alg2_k16_powm_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            for METH in sieve table; do
                "$BINDIR/alg2_k16_cuda" --data data/challenges/k16 --size $SIZE \
                    --method $METH --exponent $EXPONENT \
                    --n-challenges $N_CHALLENGES --n-gpus $NG \
                    --json "$A2"/alg2_k16_${METH}_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            done
        fi

        # ---- alg2 k32 (prime-range split; the one missing from all old data) ----
        if [ -x "$BINDIR/alg2_k32_cuda" ] && [ -f data/k32_primes.bin ] && [ -d data/challenges/k32 ]; then
            "$BINDIR/alg2_k32_cuda" --data data/challenges/k32 --size $SIZE \
                --method powm --exponent $EXPONENT --primes data/k32_primes.bin \
                --n-challenges 1 --n-gpus $NG \
                --json "$A2"/alg2_k32_powm_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            for METH in sieve table; do
                "$BINDIR/alg2_k32_cuda" --data data/challenges/k32 --size $SIZE \
                    --method $METH --exponent $EXPONENT --primes data/k32_primes.bin \
                    --n-challenges $N_CHALLENGES --n-gpus $NG \
                    --json "$A2"/alg2_k32_${METH}_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            done
            if [ -f data/k32_primes.dlt ]; then
                "$BINDIR/alg2_k32_cuda" --data data/challenges/k32 --size $SIZE \
                    --method table --exponent $EXPONENT \
                    --primes data/k32_primes.dlt --table-format delta \
                    --n-challenges $N_CHALLENGES --n-gpus $NG \
                    --json "$A2"/alg2_k32_tabledelta_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            fi
        fi

        # ---- alg2 k35 ----
        if [ -x "$BINDIR/alg2_k35_cuda" ] && [ -d data/challenges/k35 ]; then
            "$BINDIR/alg2_k35_cuda" --data data/challenges/k35 --size $SIZE \
                --method powm --exponent $EXPONENT --n-challenges 1 --n-gpus $NG \
                --json "$A2"/alg2_k35_powm_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            "$BINDIR/alg2_k35_cuda" --data data/challenges/k35 --size $SIZE \
                --method sieve --exponent $EXPONENT \
                --n-challenges $N_CHALLENGES --n-gpus $NG \
                --json "$A2"/alg2_k35_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            if [ -f data/k32_primes.bin ] && [ -f data/k35_upper_primes.bin ]; then
                "$BINDIR/alg2_k35_cuda" --data data/challenges/k35 --size $SIZE \
                    --method table --exponent $EXPONENT \
                    --primes32 data/k32_primes.bin --primes35 data/k35_upper_primes.bin \
                    --n-challenges $N_CHALLENGES --n-gpus $NG \
                    --json "$A2"/alg2_k35_table_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            fi
        fi

        # ---- alg3 / alg4 (identical shape, shared k32 table) ----
        for A in 3 4; do
            BIN="$BINDIR/alg${A}_cuda"; DIR=$([ $A -eq 3 ] && echo "$A3" || echo "$A4")
            [ -x "$BIN" ] && [ -d data/challenges/alg${A} ] || continue
            "$BIN" --data data/challenges/alg${A} --size $SIZE \
                --method powm --exponent $EXPONENT --n-challenges 1 --n-gpus $NG \
                --json "$DIR"/alg${A}_powm_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            "$BIN" --data data/challenges/alg${A} --size $SIZE \
                --method sieve --exponent $EXPONENT \
                --n-challenges $N_CHALLENGES --n-gpus $NG \
                --json "$DIR"/alg${A}_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            if [ -f data/k32_primes.bin ]; then
                "$BIN" --data data/challenges/alg${A} --size $SIZE \
                    --method table --exponent $EXPONENT --primes data/k32_primes.bin \
                    --n-challenges $N_CHALLENGES --n-gpus $NG \
                    --json "$DIR"/alg${A}_table_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            fi
        done

        # ---- alg7 (inverted sieve; reuses alg3 challenges) ----
        if [ -x "$BINDIR/alg7_cuda" ] && [ -d data/challenges/alg3 ]; then
            "$BINDIR/alg7_cuda" --data data/challenges/alg3 --size $SIZE \
                --method powm --exponent $EXPONENT --n-challenges 1 \
                --json "$A7"/alg7_powm_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
            "$BINDIR/alg7_cuda" --data data/challenges/alg3 --size $SIZE \
                --method sieve --exponent $EXPONENT --R 20 --bitbound 32 \
                --n-challenges $N_CHALLENGES --n-gpus $NG \
                --json "$A7"/alg7_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXPONENT}.json
        fi

        echo "  G${NG} size=${SIZE} done"
    done
}

for NG in $GPU_LEVELS; do
    run_level $NG
done

# ---- Cleanup + a warm-cache sanity check -----------------------------------
rm -rf "$BINDIR"
echo ""
echo "======================================== SUMMARY"
echo "JSON files written: $(find data/results/iris -name "*_${SLURM_JOB_ID}_*.json" | wc -l)"
echo ""
echo "Warm-cache check — alg3 sieve recon_ms at 24576b should be ~50-60 ms."
echo "Anything above 1000 ms means the cache went cold and the run is unusable:"
for NG in $GPU_LEVELS; do
    f="data/results/iris/alg3/G${NG}/alg3_sieve_${SLURM_JOB_ID}_24576_e${EXPONENT}.json"
    [ -f "$f" ] && echo "  G${NG}: $(grep -o '"recon_ms"[: ]*[0-9.]*' "$f" | head -1)"
done
echo ""
echo "Done — full log in gpu_scaling_${SLURM_JOB_ID}.out"

# Edit: 2026-08-24
