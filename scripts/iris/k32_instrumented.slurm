#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --partition=l40s
#SBATCH --qos=besteffort
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --output=k32_instrumented_%j.out
#SBATCH --error=k32_instrumented_%j.err

# ---------------------------------------------------------------------------
# k32 instrumented run (advisor meeting 2026-07-29)
#
# Answers three questions with one job. k32 is deliberately the ONLY variant
# here: its Amdahl ceiling for device reconstruction is 1.45x (recon is 30.8%
# of total at 24576b), so it is the cheapest case in which to establish whether
# a device port could ever pay -- and the answer generalises downward, since
# every variant with a cheaper sieve has a HIGHER recon share.
#
#   1. cbrt cache reads per core  -> "CBRTSTAT|" lines
#      Per-thread hits / misses / probes. kf_cbrt_cache_find is a LINEAR scan,
#      so probes exposes the O(n^2) cost: ~802 factors against an ~802-entry
#      cache is ~322,000 comparisons per reconstruction.
#
#   2. reconstruction phase split -> "RECONPROF|" lines
#      parallel (per-factor cbrt) vs serial (product mod M). This is the number
#      that decides the device-reconstruction question: whatever sits in the
#      serial column cannot be parallelised on a GPU at all.
#
#   3. H2D / D2H transfer cost    -> "XFERPROF|" lines
#      Currently invisible: the cudaEventRecord bracket that produces factor_ms
#      starts AFTER the challenge upload and ends BEFORE the factor download,
#      so neither transfer is counted in any figure reported so far.
#
# Cold and warm are reported separately on purpose. The cold run is where the
# cache misses land (every cbrt is a full mpz_powm); the warm runs are the
# steady state every speedup figure in the thesis is taken from. If warm shows
# ~100% hits, reconstruction is doing no exponentiation at all -- it is cache
# lookups plus a serial product, neither of which a GPU accelerates.
# ---------------------------------------------------------------------------

SIZES="256 384 512 768 1024 2048 4096 8192 16384 24576"
N_CHALLENGES=10

cd $SLURM_SUBMIT_DIR

# ---- Modules --------------------------------------------------------------
module use /opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/modules/all
module load system/CUDA

if module load math/GMP/6.2.1-GCCcore-13.2.0 2>/dev/null; then
    echo "GMP loaded: math/GMP/6.2.1-GCCcore-13.2.0"
elif module load math/GMP/6.3.0-GCCcore-13.2.0 2>/dev/null; then
    echo "GMP loaded: math/GMP/6.3.0-GCCcore-13.2.0"
else
    echo "GMP module not found"; module spider GMP 2>&1 | grep -i "math/GMP"; exit 1
fi

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
N_GPUS=${SLURM_GPUS_ON_NODE:-1}

OUT_DIR="data/results/iris/alg2/G${N_GPUS}/l40"
mkdir -p "$OUT_DIR"

BINDIR="src/iris/build_${SLURM_JOB_ID}"
mkdir -p "$BINDIR"

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "N_GPUS:       $N_GPUS"
echo "OMP threads:  ${SLURM_CPUS_PER_TASK:-?} (reconstruction sizes itself via sched_getaffinity)"
echo "PURPOSE:      cbrt cache counters + recon phase split + H2D/D2H timing"
echo ""

# ---- Build ----------------------------------------------------------------
echo "========================================"
echo "BUILD PHASE"
echo "========================================"
NVCC_FLAGS="-arch=sm_70 -std=c++17 -O3 -I src/common -Xcompiler -fopenmp"
COMMON="src/common/common.cpp src/common/factorization.cpp"
LIBS="-lgmp -lgomp"

echo -n "Building alg2_k32_cuda ... "
nvcc src/iris/alg2_k32_cuda.cu $COMMON $NVCC_FLAGS $LIBS -o "$BINDIR/alg2_k32_cuda" 2>&1 \
    && echo "OK" || { echo "FAILED"; exit 1; }
echo ""

if [ ! -f data/k32_primes.bin ]; then
    echo "ERROR: data/k32_primes.bin missing"; exit 1
fi

# ---- Run: k32 table (flat), both exponents, all canonical sizes ------------
# Flat table only. The delta path is a separate experiment (delta_rerun) and
# mixing them here would confound the transfer numbers: delta uploads ~4x fewer
# bytes at setup, which is exactly what the XFERPROF setup row measures.
for EXP in 3 5; do
  for SIZE in $SIZES; do
    echo ""
    echo "======== k32 table e=${EXP} size=${SIZE} ========"
    "$BINDIR/alg2_k32_cuda" \
        --data data/challenges/k32 --size $SIZE \
        --method table --exponent $EXP --primes data/k32_primes.bin \
        --n-challenges $N_CHALLENGES --n-gpus $N_GPUS \
        --json "$OUT_DIR"/alg2_k32_table_${SLURM_JOB_ID}_${SIZE}_e${EXP}.json
  done
done

# ---- Collect the greppable lines into standalone CSV-ish files -------------
LOG="k32_instrumented_${SLURM_JOB_ID}.out"
SUMMARY_DIR="data/results/instrumented_${SLURM_JOB_ID}"
mkdir -p "$SUMMARY_DIR"

# These greps run against the job's own stdout, which SLURM is still writing;
# everything above has already been flushed (each report ends with fflush).
{ echo "tag,thread,hits,misses,probes"
  grep '^CBRTSTAT|' "$LOG" 2>/dev/null | sed 's/^CBRTSTAT|//; s/|/,/g'
} > "$SUMMARY_DIR/cbrt_stats.csv"

{ echo "tag,parallel_ms,serial_ms,factors,calls"
  grep '^RECONPROF|' "$LOG" 2>/dev/null | sed 's/^RECONPROF|//; s/|/,/g'
} > "$SUMMARY_DIR/recon_prof.csv"

{ echo "tag,phase,calls,bytes,ms"
  grep '^XFERPROF|' "$LOG" 2>/dev/null | sed 's/^XFERPROF|//; s/|/,/g'
} > "$SUMMARY_DIR/xfer_prof.csv"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
for f in cbrt_stats recon_prof xfer_prof; do
    n=$(( $(wc -l < "$SUMMARY_DIR/$f.csv") - 1 ))
    echo "  $SUMMARY_DIR/$f.csv  ($n rows)"
done
echo ""
echo "Warm-run aggregate cbrt lines (expect hit% ~100 if the cache is working):"
grep '^CBRTSTAT|.*warm|ALL|' "$LOG" 2>/dev/null | tail -20
echo ""
rm -rf "$BINDIR"
echo "Done — full detail in $LOG"

# Edit: 2026-07-30
