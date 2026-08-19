#!/bin/bash -l
#SBATCH --job-name=minroot_timelock_gpu
#SBATCH --output=timelock_gpu_%j.out
#SBATCH --error=timelock_gpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=l40s
#SBATCH --qos=besteffort
#SBATCH --gres=gpu:1

# ---------------------------------------------------------------------------
# MinRoot time-lock — GPU batched smoothness search (thesis sub4:twophases)
#
# WHAT THIS MEASURES, AND WHY IT IS DIFFERENT FROM EVERY OTHER JOB HERE
#   Every other benchmark in this project measures PHASE 2 (solving a challenge
#   that is already smooth), because challenge_gen hands the solver a smooth
#   challenge by construction. This job measures PHASE 1 -- the smoothness
#   SEARCH -- because the time-lock puzzle's challenges are real: each of the T
#   chain steps has to be blinded until a smooth candidate turns up.
#
#   So the headline number here is NOT speedup_total. It is search throughput
#   (candidates/s) and the measured hit rate rho(u), which the rest of the
#   thesis can only estimate from the Dickman density.
#
# SINGLE GPU on purpose. The search is embarrassingly parallel and needs no
# cross-device coordination, so there is nothing for a second GPU to show that
# a larger batch does not already show. --gres=gpu:1 keeps the node free.
#
# 64-BIT MODULUS on purpose. p, x and every candidate fit in uint64_t and every
# product in __uint128_t, so the device needs no multi-limb arithmetic. A wider
# modulus needs a 192-bit device mulmod for the blinding step; attack_cuda.cu
# refuses such a puzzle rather than truncating it. See its header.
#
# Submit with:  sbatch scripts/iris/timelock_gpu.slurm
# Output lands in: data/timelock/gpu_<jobid>/
# ---------------------------------------------------------------------------

module use /opt/apps/easybuild/systems/iris/rhel810-20250803/2023b/gpu/modules/all
module load system/CUDA
if module load math/GMP/6.2.1-GCCcore-13.2.0 2>/dev/null; then
    echo "GMP 6.2.1 loaded"
elif module load math/GMP/6.3.0-GCCcore-13.2.0 2>/dev/null; then
    echo "GMP 6.3.0 loaded"
else
    echo "WARNING: no GMP module found; relying on system GMP"
fi
# OpenSSL is available system-wide (crypto.c needs -lcrypto).

set -e
cd "$SLURM_SUBMIT_DIR"

OUT=$SLURM_SUBMIT_DIR/data/timelock/gpu_${SLURM_JOB_ID}
mkdir -p "$OUT"

TL=tools/timelock

# ---------------------------------------------------------------------------
# Build — direct compiler invocations, no Makefile.
#
# The three shared .c files are compiled with gcc AS C and only then linked.
# Do NOT hand them to nvcc directly: nvcc routes .c through the host compiler
# in C++ mode, and these files use the C-legal implicit void* conversions
# (`char *h = malloc(...)`), which is a hard error in C++.
# ---------------------------------------------------------------------------
echo "=== Build ==="

gcc -O2 -std=c11 -c \
    $TL/puzzle_io.c $TL/crypto.c $TL/primes.c

# CPU reference tool (generate + the serial attack we compare against).
gcc -O2 -std=c11 -D_POSIX_C_SOURCE=199309L -Wall -o "$OUT/minroot-timelock" \
    $TL/main.c $TL/generate.c $TL/solve.c $TL/attack.c \
    puzzle_io.o crypto.o primes.o -lgmp -lcrypto
echo "  CPU tool built"

# GPU search. -arch=sm_70 embeds compute_70 PTX, which JITs natively on the L40
# (sm_89) -- the same choice all_gpu_l40.slurm makes, so one binary covers both
# the V100 and L40 partitions.
nvcc -O3 -std=c++17 -arch=sm_70 -I $TL \
     $TL/attack_cuda.cu puzzle_io.o crypto.o primes.o \
     -lgmp -lcrypto -o "$OUT/timelock_gpu"
echo "  GPU tool built"
echo ""

TOOL="$OUT/minroot-timelock"
GPU="$OUT/timelock_gpu"
PUZZLE="$OUT/puzzle_64b.mrt"
MESSAGE="MinRoot time-lock: GPU batched search"

# ---------------------------------------------------------------------------
# Puzzle — same configuration as scripts/aion/timelock_demo.slurm Config A,
# so the CPU numbers there are directly comparable.
#   64-bit p, T=5000, e=5, B=65536  ->  u = 4, rho(4) ~ 1/200
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Generate: 64-bit prime, T=5000, e=5"
echo "============================================================"
$TOOL generate -b 64 -T 5000 -e 5 -m "$MESSAGE" "$PUZZLE"
echo ""

echo "============================================================"
echo "Sequential solve (the honest work the puzzle demands)"
echo "============================================================"
time $TOOL solve "$PUZZLE"
echo ""

echo "============================================================"
echo "CPU attack reference (serial search, B=65536)"
echo "============================================================"
time $TOOL attack -B 65536 "$PUZZLE"
echo ""

# ---------------------------------------------------------------------------
# GPU batch sweep — THE experiment.
#
# Each chain step consumes exactly ONE smooth candidate, but a batch tests N at
# once and cannot stop at its first hit. So batch size trades GPU occupancy
# against wasted work, and the tool reports that waste as "Overshoot":
#
#   small N -> little waste, poor occupancy, more launches per step
#   large N -> full occupancy, most of the batch discarded
#
# Host emulation of this kernel over a T=1000 chain measured overshoot
# 1.93x / 3.18x / 11.82x at N = 256 / 512 / 2048, i.e. essentially linear in N
# once N*rho >> 1. The sweep below should reproduce that shape on real hardware
# while the measured hit rate stays FLAT -- rho(u) is a property of the number
# field, not of the batching.
# ---------------------------------------------------------------------------
for BATCH in 256 1024 4096 16384; do
    echo "============================================================"
    echo "GPU batched search — batch=${BATCH}, seed=20260728"
    echo "============================================================"
    time $GPU "$PUZZLE" -B 65536 --batch $BATCH --seed 20260728
    echo ""
done

# ---------------------------------------------------------------------------
# Reproducibility gate — a fixed seed must reproduce a run exactly.
#
# This is here because tests/prefilter_test.cpp once claimed a cross-rank
# invariant it could not actually check, having reseeded per rank. The counter
# based RNG in attack_cuda.cu (splitmix64 over seed + batch counter + thread id)
# is deliberately independent of scheduling, so the two runs below MUST report
# identical candidate and hit counts.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Reproducibility gate: same seed twice, counts must match"
echo "============================================================"
for rep in 1 2; do
    echo "--- repeat $rep ---"
    $GPU "$PUZZLE" -B 65536 --batch 1024 --seed 424242 \
        | grep -E "Candidates tested|Smooth candidates found|Message"
done
echo ""

echo "============================================================"
echo "Job ${SLURM_JOB_ID} complete.  Artefacts in: $OUT/"
echo ""
echo "READING THE OUTPUT"
echo "  'Measured hit rate' is rho(u) COUNTED, not estimated. It should be"
echo "  flat across the batch sweep and should agree with the CPU reference's"
echo "  'avg attempts per step'. Note it sits ABOVE the asymptotic rho(4) ="
echo "  0.00491 (1 in 204): at finite size the smooth density exceeds rho(u)"
echo "  by roughly (1 + gamma*u/ln B) = 1.21 here, so expect ~1 in 170."
echo ""
echo "  'Overshoot' is the price of batching and is the reason a GPU search"
echo "  does not convert its raw throughput advantage into an equal wall-clock"
echo "  advantage over the serial CPU search."
echo "============================================================"
