#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --qos=besteffort
# --gres=gpu:4 takes all GPUs on the node, so request all its CPU cores too:
# the parallel cbrt reconstruction (kf_compute_cbrt_from_factors_omp) sizes
# itself to whatever cores this process is granted (sched_getaffinity). 28 =
# the IRIS V100 node core count; the l40s node may differ — adjust if so.
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --output=all_gpu_l40_%j.out
#SBATCH --error=all_gpu_l40_%j.err

# ---------------------------------------------------------------------------
# IRIS GPU (L40 / Ada Lovelace partition) — all CUDA prototypes, all canonical sizes, multi-challenge
#
# Each binary is called ONCE per (algorithm, method, size, exponent).
# It loads challenge_0 (cold run) then challenges 1..N-1 (warm runs)
# internally and writes ONE JSON — matching the CPU benchmark structure.
#
# Multi-GPU: N_GPUS is read from $SLURM_GPUS_ON_NODE (set by --gres=gpu:N),
# so the same script works for G1/G2/G4 jobs without edits. Each device scans
# a disjoint prime-range slice; factor_ms = max(device times), ~1/N_GPUS scaling.
# k16 and alg7 are unaffected (k16 doesn't benefit; alg7 has its own kernel).
#
# JSON naming: data/results/iris/<alg>/G{N_GPUS}/l40/<alg>_<method>_JOBID_SIZE_eE.json
#
# Prerequisites (copy from AION once):
#   rsync -av <aion>:<proj>/data/challenges/  <iris>:<proj>/data/challenges/
#   rsync <aion>:<proj>/data/k32_primes.bin   <iris>:<proj>/data/k32_primes.bin
#   rsync <aion>:<proj>/data/k35_upper_primes.bin <iris>:<proj>/data/k35_upper_primes.bin
#
# NOTE — the delta-encoded prime table (--table-format delta on the CPU solvers)
# is CPU-ONLY. The CUDA solvers here always use the flat table; there is no
# --table-format flag on the GPU side. The GPU delta decode kernel (each thread
# block decodes its delta block into shared memory, so a 4-way k32 slice ~51 MB
# lives in the L40's ~96 MB L2) is future work — the step that would actually
# test the predicted L40 L2-residency win for k32 documented in THESIS §7.12.
# See wiki/implementation/delta-prime-table.md ("Still to do — GPU").
# ---------------------------------------------------------------------------

# Canonical sizes matching src/common/sizes.h KSC_CANONICAL_SIZES_LIST
SIZES="256 384 512 768 1024 2048 4096 8192 16384 24576"

# Number of challenge files: challenge_0 = cold, challenge_1..N-1 = warm
N_CHALLENGES=10

cd $SLURM_SUBMIT_DIR

# ---- Modules (L40 partition) -----------------------------------------------
module use /opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/modules/all
module load system/CUDA

if module load math/GMP/6.2.1-GCCcore-13.2.0 2>/dev/null; then
    echo "GMP loaded: math/GMP/6.2.1-GCCcore-13.2.0"
elif module load math/GMP/6.3.0-GCCcore-13.2.0 2>/dev/null; then
    echo "GMP loaded: math/GMP/6.3.0-GCCcore-13.2.0"
else
    echo "GMP module not found — available GMP versions:"
    module spider GMP 2>&1 | grep -i "math/GMP"
    echo "Edit this script with the correct module name and resubmit."
    exit 1
fi

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# SLURM sets SLURM_GPUS_ON_NODE from --gres=gpu:N (header or sbatch command-line override).
# Using it directly is simpler and faster than spawning nvidia-smi.
N_GPUS=${SLURM_GPUS_ON_NODE:-1}

# Results go into per-algorithm subdirs, under an l40/ leaf so L40 and V100 data
# coexist (data/results/iris/alg2/G4/l40, alg3/G4/l40 ...).
IRIS_BASE="data/results/iris"
ALG2_DIR="$IRIS_BASE/alg2/G${N_GPUS}/l40"
ALG3_DIR="$IRIS_BASE/alg3/G${N_GPUS}/l40"
ALG4_DIR="$IRIS_BASE/alg4/G${N_GPUS}/l40"
ALG7_DIR="$IRIS_BASE/alg7/G${N_GPUS}/l40"
mkdir -p "$ALG2_DIR" "$ALG3_DIR" "$ALG4_DIR" "$ALG7_DIR"

# Job-private build directory: prevents concurrent G1/G2/G4 jobs from overwriting
# each other's binaries when running in the same working directory simultaneously.
BINDIR="src/iris/build_${SLURM_JOB_ID}"
mkdir -p "$BINDIR"

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "N_GPUS: $N_GPUS"
echo "ALG2_DIR: $ALG2_DIR"
echo "ALG3_DIR: $ALG3_DIR"
echo "ALG4_DIR: $ALG4_DIR"
echo "ALG7_DIR: $ALG7_DIR"
echo ""

# ---- Build all binaries ----------------------------------------------------
echo "========================================"
echo "BUILD PHASE"
echo "========================================"

# -arch=sm_70 embeds compute_70 PTX, which JITs fine on the L40 (sm_89) — this is
# how the existing L40 results were produced. For native Ada codegen switch to sm_89.
NVCC_FLAGS="-arch=sm_70 -std=c++17 -O3 -I src/common -Xcompiler -fopenmp"
COMMON="src/common/common.cpp src/common/factorization.cpp"
LIBS="-lgmp -lgomp"

build_one() {
    local src=$1 out=$2
    echo -n "Building $out ... "
    nvcc src/iris/${src} $COMMON $NVCC_FLAGS $LIBS -o "$BINDIR/${out}" 2>&1
    if [ $? -eq 0 ]; then echo "OK"; else echo "FAILED"; return 1; fi
}

build_one alg7_cuda.cu      alg7_cuda      || BUILD_FAIL=1
build_one alg2_k16_cuda.cu  alg2_k16_cuda  || BUILD_FAIL=1
build_one alg2_k32_cuda.cu  alg2_k32_cuda  || BUILD_FAIL=1
build_one alg2_k35_cuda.cu  alg2_k35_cuda  || BUILD_FAIL=1
build_one alg3_cuda.cu      alg3_cuda      || BUILD_FAIL=1
build_one alg4_cuda.cu      alg4_cuda      || BUILD_FAIL=1

if [ -n "$BUILD_FAIL" ]; then
    echo "One or more builds failed — continuing with what compiled"
fi
echo ""

section() { echo ""; echo "========================================"; echo "$1"; echo "========================================"; echo ""; }

# ============================================================================
# Algorithm 7 — inverted sieve (reuses alg3 challenge files)
# Multi-GPU: --n-gpus splits the prime range across devices for BOTH the Phase-1
# inverted-sieve kernel (partial log_sums summed on host) and the Phase-2
# candidate-factoring kernel (dividing-prime lists merged on host).
# ============================================================================
section "ALG7 CUDA — all sizes, $N_CHALLENGES challenges, $N_GPUS GPU(s)"

if [ -x "$BINDIR/alg7_cuda" ]; then
  for EXP in 3 5; do
    for SIZE in $SIZES; do
      echo "--- alg7 e=${EXP} size=${SIZE}: powm ---"
      "$BINDIR/alg7_cuda" \
          --data data/challenges/alg3 --size $SIZE \
          --method powm --exponent $EXP --n-challenges 1 \
          --json "$ALG7_DIR"/alg7_powm_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      echo "--- alg7 e=${EXP} size=${SIZE}: sieve ---"
      "$BINDIR/alg7_cuda" \
          --data data/challenges/alg3 --size $SIZE \
          --method sieve --exponent $EXP --R 20 --bitbound 32 \
          --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
          --json "$ALG7_DIR"/alg7_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json
      echo ""
    done
  done
else
  echo "SKIP: alg7_cuda not built"
fi

# ============================================================================
# Algorithm 2 — k16 (6542 primes <= 65536; batch kernel; sieve + table)
# Multi-GPU: --n-gpus splits the 256-candidate BATCH across devices (the 26 KB
# prime table fits in each GPU's constant memory, so unlike k32 there is nothing
# to tile — the batch is the split axis). Expect a flat curve: the kernel is
# microseconds and host-side reconstruction dominates; this measures that.
# sieve and table produce identical results (primes generated in-memory for both)
# ============================================================================
section "ALG2 k16 CUDA — all sizes, $N_CHALLENGES challenges, $N_GPUS GPU(s)"

if [ -x "$BINDIR/alg2_k16_cuda" ]; then
  for EXP in 3 5; do
    for SIZE in $SIZES; do
      echo "--- k16 e=${EXP} size=${SIZE}: powm ---"
      "$BINDIR/alg2_k16_cuda" \
          --data data/challenges/k16 --size $SIZE \
          --method powm --exponent $EXP --n-challenges 1 \
          --json "$ALG2_DIR"/alg2_k16_powm_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      echo "--- k16 e=${EXP} size=${SIZE}: sieve ---"
      "$BINDIR/alg2_k16_cuda" \
          --data data/challenges/k16 --size $SIZE \
          --method sieve --exponent $EXP --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
          --json "$ALG2_DIR"/alg2_k16_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      echo "--- k16 e=${EXP} size=${SIZE}: table ---"
      "$BINDIR/alg2_k16_cuda" \
          --data data/challenges/k16 --size $SIZE \
          --method table --exponent $EXP --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
          --json "$ALG2_DIR"/alg2_k16_table_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json
      echo ""
    done
  done
else
  echo "SKIP: alg2_k16_cuda not built"
fi

# ============================================================================
# Algorithm 2 — k32 (813 MB prime table; sieve + table; N_GPUS prime splitting)
# ============================================================================
section "ALG2 k32 CUDA — all sizes, $N_CHALLENGES challenges, $N_GPUS GPU(s)"

if [ -x "$BINDIR/alg2_k32_cuda" ] && [ -f data/k32_primes.bin ]; then
  for EXP in 3 5; do
    for SIZE in $SIZES; do
      echo "--- k32 e=${EXP} size=${SIZE}: powm ---"
      "$BINDIR/alg2_k32_cuda" \
          --data data/challenges/k32 --size $SIZE \
          --method powm --exponent $EXP --primes data/k32_primes.bin \
          --n-challenges 1 --n-gpus $N_GPUS \
          --json "$ALG2_DIR"/alg2_k32_powm_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      echo "--- k32 e=${EXP} size=${SIZE}: sieve ---"
      "$BINDIR/alg2_k32_cuda" \
          --data data/challenges/k32 --size $SIZE \
          --method sieve --exponent $EXP --primes data/k32_primes.bin \
          --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
          --json "$ALG2_DIR"/alg2_k32_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      echo "--- k32 e=${EXP} size=${SIZE}: table ---"
      "$BINDIR/alg2_k32_cuda" \
          --data data/challenges/k32 --size $SIZE \
          --method table --exponent $EXP --primes data/k32_primes.bin \
          --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
          --json "$ALG2_DIR"/alg2_k32_table_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json
      echo ""
    done
  done
else
  echo "SKIP: alg2_k32_cuda not built or k32_primes.bin missing"
fi

# ============================================================================
# Algorithm 2 — k35 (9.6 GB upper table preloaded into RAM; N_GPUS splitting)
# sieve: generates primes on-the-fly (very slow, no binary files needed)
# table: loads prebuilt prime files (fast; requires k32_primes.bin + k35_upper_primes.bin)
# ============================================================================
section "ALG2 k35 CUDA — all sizes, $N_CHALLENGES challenges, $N_GPUS GPU(s)"

if [ -x "$BINDIR/alg2_k35_cuda" ]; then
  for EXP in 3 5; do
    for SIZE in $SIZES; do
      echo "--- k35 e=${EXP} size=${SIZE}: powm ---"
      "$BINDIR/alg2_k35_cuda" \
          --data data/challenges/k35 --size $SIZE \
          --method powm --exponent $EXP \
          --n-challenges 1 --n-gpus $N_GPUS \
          --json "$ALG2_DIR"/alg2_k35_powm_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      echo "--- k35 e=${EXP} size=${SIZE}: sieve ---"
      "$BINDIR/alg2_k35_cuda" \
          --data data/challenges/k35 --size $SIZE \
          --method sieve --exponent $EXP \
          --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
          --json "$ALG2_DIR"/alg2_k35_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      if [ -f data/k32_primes.bin ] && [ -f data/k35_upper_primes.bin ]; then
        echo "--- k35 e=${EXP} size=${SIZE}: table ---"
        "$BINDIR/alg2_k35_cuda" \
            --data data/challenges/k35 --size $SIZE \
            --method table --exponent $EXP \
            --primes32 data/k32_primes.bin --primes35 data/k35_upper_primes.bin \
            --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
            --json "$ALG2_DIR"/alg2_k35_table_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json
      fi
      echo ""
    done
  done
else
  echo "SKIP: alg2_k35_cuda not built"
fi

# ============================================================================
# Algorithm 3 — almost-smooth cofactor (N_GPUS prime splitting)
# ============================================================================
section "ALG3 CUDA — all sizes, $N_CHALLENGES challenges, $N_GPUS GPU(s)"

if [ -x "$BINDIR/alg3_cuda" ]; then
  for EXP in 3 5; do
    for SIZE in $SIZES; do
      echo "--- alg3 e=${EXP} size=${SIZE}: powm ---"
      "$BINDIR/alg3_cuda" \
          --data data/challenges/alg3 --size $SIZE \
          --method powm --exponent $EXP \
          --n-challenges 1 --n-gpus $N_GPUS \
          --json "$ALG3_DIR"/alg3_powm_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      echo "--- alg3 e=${EXP} size=${SIZE}: sieve ---"
      "$BINDIR/alg3_cuda" \
          --data data/challenges/alg3 --size $SIZE \
          --method sieve --exponent $EXP \
          --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
          --json "$ALG3_DIR"/alg3_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      if [ -f data/k32_primes.bin ]; then
        echo "--- alg3 e=${EXP} size=${SIZE}: table ---"
        "$BINDIR/alg3_cuda" \
            --data data/challenges/alg3 --size $SIZE \
            --method table --exponent $EXP --primes data/k32_primes.bin \
            --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
            --json "$ALG3_DIR"/alg3_table_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json
      fi
      echo ""
    done
  done
else
  echo "SKIP: alg3_cuda not built"
fi

# ============================================================================
# Algorithm 4 — pre-filter + composite cofactor (N_GPUS prime splitting)
# ============================================================================
section "ALG4 CUDA — all sizes, $N_CHALLENGES challenges, $N_GPUS GPU(s)"

if [ -x "$BINDIR/alg4_cuda" ]; then
  for EXP in 3 5; do
    for SIZE in $SIZES; do
      echo "--- alg4 e=${EXP} size=${SIZE}: powm ---"
      "$BINDIR/alg4_cuda" \
          --data data/challenges/alg4 --size $SIZE \
          --method powm --exponent $EXP \
          --n-challenges 1 --n-gpus $N_GPUS \
          --json "$ALG4_DIR"/alg4_powm_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      echo "--- alg4 e=${EXP} size=${SIZE}: sieve ---"
      "$BINDIR/alg4_cuda" \
          --data data/challenges/alg4 --size $SIZE \
          --method sieve --exponent $EXP \
          --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
          --json "$ALG4_DIR"/alg4_sieve_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json

      if [ -f data/k32_primes.bin ]; then
        echo "--- alg4 e=${EXP} size=${SIZE}: table ---"
        "$BINDIR/alg4_cuda" \
            --data data/challenges/alg4 --size $SIZE \
            --method table --exponent $EXP --primes data/k32_primes.bin \
            --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
            --json "$ALG4_DIR"/alg4_table_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json
      fi
      echo ""
    done
  done
else
  echo "SKIP: alg4_cuda not built"
fi

# ============================================================================
# Cleanup
# ============================================================================
section "CLEANUP"
rm -rf "$BINDIR"
echo "Results in: $IRIS_BASE/alg{2,3,4,7}/G${N_GPUS}/l40/"
find "$ALG2_DIR" "$ALG3_DIR" "$ALG4_DIR" "$ALG7_DIR" -name "*_${SLURM_JOB_ID}_*.json" 2>/dev/null | wc -l | xargs -I{} echo "  {} JSON files written"
echo "Done — check all_gpu_l40_${SLURM_JOB_ID}.out"

# Edit: 2026-07-22
