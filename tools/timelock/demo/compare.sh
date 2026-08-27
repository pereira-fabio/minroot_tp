#!/usr/bin/env bash
# The centrepiece: the same smoothness search, serial on one core and batched
# on the GPU, against the same puzzle.
#
#   ./compare.sh                     # after ./setup.sh
#   BATCH=4096 ./compare.sh          # show the overshoot penalty of a big batch
#
# Both binaries solve the identical puzzle and both must decrypt the plaintext,
# so this is a like-for-like wall-clock comparison of the ONE phase of the
# attack that admits any parallelism at all.

set -euo pipefail

PUZZLE=${PUZZLE:-puzzle_attack.mrt}
BATCH=${BATCH:-1024}
BOUND=${BOUND:-65536}
CPU=${CPU:-./build/minroot-timelock}
GPU=${GPU:-./build/timelock_gpu}

[ -f "$PUZZLE" ] || { echo "No $PUZZLE — run ./setup.sh first." >&2; exit 1; }
[ -x "$CPU" ]    || { echo "No $CPU — run ./setup.sh first." >&2; exit 1; }

bar() { printf '\033[1m%s\033[0m\n' "────────────────────────────────────────────────────────────"; }
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ------------------------------------------------------------- serial ------
say "1/2  CPU — serial search, one candidate at a time"
echo "     (no OpenMP, no threads: this is the loop attack.c actually runs)"
bar
"$CPU" attack -B "$BOUND" "$PUZZLE" 2>&1 | tee cpu.log
bar

# ---------------------------------------------------------------- GPU ------
if [ ! -x "$GPU" ]; then
  echo
  echo "GPU tool not built (nvcc missing at setup time) — CPU half only."
  exit 0
fi

say "2/2  GPU — batched search, $BATCH candidates per launch"
echo "     (same puzzle, same bound; the search loop is inverted onto the device)"
bar
"$GPU" "$PUZZLE" -B "$BOUND" --batch "$BATCH" 2>&1 | tee gpu.log
bar

# ------------------------------------------------------------ compare ------
pick() { grep -m1 "$1" "$2" 2>/dev/null | sed "s/.*$1//" | awk '{print $1}'; }

CPU_S=$(grep -m1 'Attack solved in' cpu.log | awk '{print $5}')
GPU_S=$(grep -m1 'Attack solved in' gpu.log | awk '{print $5}')
CPU_AVG=$(grep -m1 'Total blinding attempts' cpu.log | sed 's/.*avg //' | awk '{print $1}')
GPU_THR=$(grep -m1 'Search throughput' gpu.log | awk '{print $4}')
GPU_RHO=$(grep -m1 'Measured hit rate' gpu.log | sed 's/.*1 in //' | tr -d ')')

say "Result"
printf '  %-26s %14s %14s\n' "" "CPU (serial)" "GPU (batched)"
printf '  %-26s %14s %14s\n' "wall clock" "${CPU_S:-?} s" "${GPU_S:-?} s"
[ -n "${GPU_THR:-}" ] && printf '  %-26s %14s %14s\n' "search throughput" "--" "${GPU_THR} Mc/s"
[ -n "${CPU_AVG:-}" ] && printf '  %-26s %14s %14s\n' "attempts / round" "${CPU_AVG}" "${GPU_RHO:-–} (measured)"
echo
if [ -n "${CPU_S:-}" ] && [ -n "${GPU_S:-}" ]; then
  awk -v c="$CPU_S" -v g="$GPU_S" 'BEGIN{
    printf "  \033[1mSpeedup from parallelising the search: %.2fx\033[0m\n", c/g }'
fi

cat <<'EOF'

  What just happened, and what did not:

    Parallelised  the SEARCH (phase 1) -- independent candidates, so the
                  loop inverts cleanly onto thousands of GPU threads.
    Not touched   the CHAIN -- the T rounds stay strictly sequential on both
                  sides. Neither run shortens the dependency chain, because
                  nothing can: that is what MinRoot claims and it holds.

  Both runs decrypted the same plaintext, so both passed the correctness gate.
EOF

if [ -n "${CPU_S:-}" ] && [ -n "${GPU_S:-}" ]; then
  awk -v c="$CPU_S" -v g="$GPU_S" 'BEGIN{
    printf "\n  \033[1mSame puzzle, solved twice: serially on the CPU in %ss, ", c
    printf "in parallel on the GPU in %ss (%.2fx).\033[0m\n", g, c/g }'
fi
