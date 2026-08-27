#!/usr/bin/env bash
# Calibrate and generate the two defence puzzles.
#
#   ./setup.sh                # 20-minute puzzle at 1024 bits, plus the 64-bit attack puzzle
#   MINUTES=15 ./setup.sh     # shorter talk
#   BITS=2048  ./setup.sh     # fewer, slower rounds
#
# Calibration is MEASURED on this machine, not assumed: a short probe puzzle is
# solved, the per-round cost is read off, and T is scaled to the target.
# Nothing here needs the cluster.

set -euo pipefail

MINUTES=${MINUTES:-20}
ATTACK_SECS=${ATTACK_SECS:-30}   # target wall time for the CPU half of compare.sh
BITS=${BITS:-1024}
PROBE_T=${PROBE_T:-300}
ARCH=${ARCH:-sm_75}          # 2080 Super = Turing = sm_75 (V100 would be sm_70)
SRC=${SRC:-..}               # the timelock sources live one level up
MSG_A=${MSG_A:-"The delay held for the whole talk."}
MSG_B=${MSG_B:-"Solved by searching in parallel on the GPU, serially on the CPU."}

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- build ----
say "Building"
if [ ! -d "$SRC" ]; then
  echo "Cannot find the timelock sources at: $SRC" >&2
  echo "Set SRC=/path/to/minroot_tp/tools/timelock and re-run." >&2
  exit 1
fi
# Build out-of-tree so the repo stays clean.  make(1) cannot cope with a path
# containing a colon, which the NAS mount has, so copy the sources first.
rm -rf build && mkdir -p build
cp "$SRC"/*.c "$SRC"/*.h build/
( cd build && gcc -O2 -std=c11 -D_POSIX_C_SOURCE=199309L -w \
      -o minroot-timelock ./*.c -lgmp -lcrypto )
TOOL=$PWD/build/minroot-timelock
echo "  built: $TOOL"

# Optional GPU build.  attack_cuda.cu batches the smoothness search on the
# device -- the only part of this demo that uses any parallelism at all
# (solve is a sequential chain by definition, and attack.c searches serially).
GPUTOOL=""
if command -v nvcc >/dev/null 2>&1; then
  cp "$SRC"/attack_cuda.cu build/ 2>/dev/null || true
  if [ -f build/attack_cuda.cu ]; then
    # The .c files must be compiled by gcc AS C first: nvcc routes .c through
    # the host compiler in C++ mode, where `char *h = malloc(...)` is an error.
    ( cd build \
      && gcc -O2 -std=c11 -D_POSIX_C_SOURCE=199309L -w -c puzzle_io.c crypto.c primes.c \
      && nvcc -O3 -std=c++17 -arch="$ARCH" -I. \
              attack_cuda.cu puzzle_io.o crypto.o primes.o \
              -lgmp -lcrypto -o timelock_gpu ) \
      && GPUTOOL=$PWD/build/timelock_gpu \
      && echo "  built: $GPUTOOL  (arch=$ARCH)"
  fi
else
  echo "  nvcc not found -- skipping the GPU tool (CPU demo still works)"
fi

# ------------------------------------------------------------ calibrate ----
say "Calibrating on this machine ($BITS-bit, probe T=$PROBE_T)"
"$TOOL" generate -b "$BITS" -T "$PROBE_T" -m probe probe.mrt >/dev/null
PROBE_S=$("$TOOL" solve probe.mrt | awk '/Solved in/ {print $4}' | tr -d 's')
if [ -z "${PROBE_S:-}" ]; then echo "Probe solve failed." >&2; exit 1; fi

T_A=$(awk -v s="$PROBE_S" -v n="$PROBE_T" -v m="$MINUTES" \
      'BEGIN { printf "%d", (m*60) / (s/n) }')
PER_STEP_MS=$(awk -v s="$PROBE_S" -v n="$PROBE_T" 'BEGIN{printf "%.3f", 1000*s/n}')
echo "  one forward round : ${PER_STEP_MS} ms"
echo "  T for ${MINUTES} min      : ${T_A}"

# ------------------------------------------------------- puzzle A (slow) ----
say "Puzzle A — the delay (run this THROUGH the talk)"
time "$TOOL" generate -b "$BITS" -T "$T_A" -m "$MSG_A" puzzle_delay.mrt
echo "  generation is cheap because the chain is walked BACKWARDS;"
echo "  solving must walk it forwards, one e-th root per round."

# ----------------------------------------------------- puzzle B (attack) ----
# 64 bits is not a choice, it is a constraint: the smoothness search only
# terminates when u = |p|/log2(B) is small.  At B=2^16 that caps |p| at 64.
say "Puzzle B — the comparison (64-bit)"
# Size T so the SERIAL half of compare.sh lands near ATTACK_SECS.  The attack is
# a lottery, so calibrate on a 200-round probe rather than a handful of rounds.
"$TOOL" generate -b 64 -T 200 -m probe probeB.mrt >/dev/null
PB=$("$TOOL" attack probeB.mrt | awk '/Attack solved in/ {print $5}')
T_B=$(awk -v s="$PB" -v n=200 -v t="$ATTACK_SECS" 'BEGIN{ printf "%d", t/(s/n) }')
[ "$T_B" -lt 50 ] && T_B=50
echo "  one serial round   : $(awk -v s="$PB" 'BEGIN{printf "%.1f", 1000*s/200}') ms"
echo "  T for ${ATTACK_SECS}s serial : $T_B"
"$TOOL" generate -b 64 -T "$T_B" -m "$MSG_B" puzzle_attack.mrt
echo "  64-bit / B=2^16 gives u=4, so rho(4) ~ 1/200 attempts per round."

# --------------------------------------------------------------- ready ----
say "Ready"
cat <<EOF
THE CENTREPIECE -- same puzzle, serial vs batched:

    ./compare.sh

  Runs the CPU search then the GPU search on puzzle_attack.mrt and prints the
  speedup. About ${ATTACK_SECS}s plus a few seconds. This is the parallel result.

RUNNING THROUGH THE TALK -- start this before you speak:

    $TOOL solve puzzle_delay.mrt

  A progress line every 5% with elapsed and ETA. Single-threaded, and that is
  the point -- the chain cannot be parallelised, which is what a VDF claims.

EOF
if [ -z "$GPUTOOL" ]; then
  echo "  NOTE: nvcc was not found, so compare.sh runs the CPU half only."
  echo "        Install the CUDA toolkit and re-run ./setup.sh."
fi
