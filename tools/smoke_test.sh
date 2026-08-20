#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — build everything, run every solver once, check every result
#
# Answers one question: does this checkout work on this machine? It is not a
# benchmark. Sizes are small so the whole thing takes a couple of minutes, and
# at these sizes the attack is far SLOWER than the sequential baseline -- that
# is the expected result, not a failure (see the two regimes in README.md).
#
#   ./tools/smoke_test.sh              CPU, and GPU too if nvcc and a card exist
#   ./tools/smoke_test.sh --cpu        CPU only
#   ./tools/smoke_test.sh --gpu        GPU only
#   ./tools/smoke_test.sh --size 512   smaller/faster (default 1024)
#   ./tools/smoke_test.sh --slow       also k35 and alg7 (minutes, see below)
#   ./tools/smoke_test.sh --keep       keep generated challenges on exit
#
# k35 and alg7 are behind --slow because the sieve range is set by the
# smoothness bound, not by the challenge size: k35 generates 1.4 billion primes
# on every run against k32's 194 million, so it takes minutes on a small
# machine no matter how small the challenge is. alg7 sweeps 2^20 lifted
# candidates for the same reason.
#
# Every solver prints "SUCCESS: <e>-th root is correct!" when the correctness
# gate passes. This script fails if any run does not.
# =============================================================================
set -u

SIZE=1024
NCH=3
RUN_CPU=1
RUN_GPU=auto
KEEP=0
RANKS=2
SLOW=0

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)   RUN_GPU=0 ;;
        --gpu)   RUN_CPU=0; RUN_GPU=1 ;;
        --size)  SIZE="$2"; shift ;;
        --ranks) RANKS="$2"; shift ;;
        --slow)  SLOW=1 ;;
        --keep)  KEEP=1 ;;
        -h|--help) sed -n '2,24p' "$0" | sed 's|^# \{0,1\}||'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

cd "$(dirname "$0")/.." || exit 1

PASS=0; FAIL=0; SKIP=0
RESULTS=""

record() {  # record <status> <name> <detail>
    RESULTS="${RESULTS}$1|$2|$3
"
    case "$1" in
        PASS) PASS=$((PASS+1)) ;;
        FAIL) FAIL=$((FAIL+1)) ;;
        SKIP) SKIP=$((SKIP+1)) ;;
    esac
}

# Run a command, then decide by what it printed. A solver that exits 0 but
# prints FAILURE has not passed, and one that prints nothing has not either.
check() {  # check <name> <command...>
    local name="$1"; shift
    local out rc
    out=$("$@" 2>&1); rc=$?
    if echo "$out" | grep -q "SUCCESS: .*root is correct"; then
        record PASS "$name" "correctness gate passed"
    elif echo "$out" | grep -q "FAILURE: .*root is incorrect"; then
        record FAIL "$name" "WRONG ROOT — correctness gate failed"
    elif [ $rc -ne 0 ]; then
        local why="exit $rc"
        [ $rc -gt 128 ] && why="$why (signal $((rc-128)))"
        record FAIL "$name" "$why: $(echo "$out" | grep -iE 'error|cannot|abort' | head -1)"
    else
        record FAIL "$name" "no correctness line in output"
    fi
}

# The powm method is the sequential baseline: it times mpz_powm and reconstructs
# nothing, so it prints no correctness line and `check` cannot be used on it.
# What matters is that it produced a positive measurement -- every speedup in
# this project is that number divided by an attack time.
check_baseline() {  # check_baseline <name> <command...>
    local name="$1"; shift
    local out rc ms
    out=$("$@" 2>&1); rc=$?
    ms=$(echo "$out" | sed -n 's/.*powm (sequential baseline):[^=]*= *\([0-9.][0-9.]*\) ms.*/\1/p' | head -1)
    if [ -n "$ms" ] && awk -v m="$ms" 'BEGIN{exit !(m+0>0)}'; then
        record PASS "$name" "baseline ${ms} ms"
    elif [ $rc -ne 0 ]; then
        local why="exit $rc"
        [ $rc -gt 128 ] && why="$why (signal $((rc-128)))"
        record FAIL "$name" "$why"
    else
        record FAIL "$name" "no baseline measurement in output"
    fi
}

echo "============================================================"
echo " minroot_tp smoke test — ${SIZE}-bit, ${NCH} challenges"
echo "============================================================"

# ---- build -----------------------------------------------------------------
echo
echo "--- building ---"
if [ "$RUN_CPU" = 1 ]; then
    make challenges >/dev/null || { echo "challenge generator failed to build"; exit 1; }
    make >/dev/null 2>&1 || { echo "CPU build FAILED — run 'make' to see why"; exit 1; }
    echo "CPU solvers built."
fi
if [ "$RUN_GPU" != 0 ]; then
    if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1 \
       && nvidia-smi -L >/dev/null 2>&1; then
        make gpu >/dev/null 2>&1 || { echo "GPU build FAILED — run 'make gpu' to see why"; exit 1; }
        echo "GPU solvers built ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))."
        RUN_GPU=1
    else
        echo "No nvcc or no GPU visible — skipping the GPU half."
        RUN_GPU=0
    fi
fi

# ---- challenges ------------------------------------------------------------
echo
echo "--- generating challenges (${SIZE}-bit) ---"
for spec in "16" "32" "35" "32 alg3" "32 alg4" "32 alg4_multi"; do
    # shellcheck disable=SC2086
    ./challenges "$NCH" "$SIZE" $spec >/dev/null 2>&1 \
        || echo "  warning: generation failed for: $spec"
done
echo "done."

K16=data/challenges/k16;  K32=data/challenges/k32;  K35=data/challenges/k35
A3=data/challenges/alg3;  A4=data/challenges/alg4

# ---- CPU -------------------------------------------------------------------
if [ "$RUN_CPU" = 1 ]; then
    echo
    echo "--- CPU solvers (${RANKS} MPI ranks) ---"
    M() { mpirun -n "$RANKS" "$@"; }

    check "cpu k16   sieve" M ./alg2_k16 $K16/challenge_0_16_$SIZE.txt   $K16/modulo_16_$SIZE.txt   --method sieve
    check "cpu k32   sieve" M ./alg2_k32 $K32/challenge_0_32_$SIZE.txt   $K32/modulo_32_$SIZE.txt   --method sieve
    check "cpu alg3  sieve" M ./alg3     $A3/challenge_0_alg3_$SIZE.txt  $A3/modulo_alg3_$SIZE.txt  --method sieve
    check "cpu alg4  sieve" M ./alg4     $A4/challenge_0_alg4_$SIZE.txt  $A4/modulo_alg4_$SIZE.txt  --method sieve
    check_baseline "cpu k16  powm" M ./alg2_k16 $K16/challenge_0_16_$SIZE.txt $K16/modulo_16_$SIZE.txt --method powm
    if [ "$SLOW" = 1 ]; then
        check "cpu k35   sieve" M ./alg2_k35 $K35/challenge_0_35_$SIZE.txt  $K35/modulo_35_$SIZE.txt  --method sieve
        check "cpu alg7  sieve" M ./alg7     $A3/challenge_0_alg3_$SIZE.txt $A3/modulo_alg3_$SIZE.txt --method sieve --R 1048576
    else
        record SKIP "cpu k35   sieve" "1.4 B primes per run — use --slow"
        record SKIP "cpu alg7  sieve" "2^20 candidate lifts — use --slow"
    fi
fi

# ---- GPU -------------------------------------------------------------------
if [ "$RUN_GPU" = 1 ]; then
    echo
    echo "--- GPU solvers (1 device) ---"
    # Device memory in MiB, to decide whether the big prime tables fit.
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    VRAM=${VRAM:-0}

    G() { ./"$1"_cuda --data "$2" --size "$SIZE" --method "$3" --n-challenges "$NCH" --n-gpus 1; }

    check "gpu k16   sieve" G alg2_k16 "$K16" sieve
    check "gpu k32   sieve" G alg2_k32 "$K32" sieve
    check "gpu alg3  sieve" G alg3     "$A3"  sieve
    check "gpu alg4  sieve" G alg4     "$A4"  sieve

    if [ "$SLOW" = 1 ]; then
        check "gpu alg7  sieve" G alg7 "$A3" sieve
        # k35's table method uploads ~9.6 GB; its sieve method uploads nothing
        # but still generates 1.4 B primes on the host.
        if [ "$VRAM" -ge 12000 ]; then
            check "gpu k35   sieve" G alg2_k35 "$K35" sieve
        else
            record SKIP "gpu k35   sieve" "needs >12 GB VRAM, this card has ${VRAM} MiB"
        fi
    else
        record SKIP "gpu alg7  sieve" "use --slow"
        record SKIP "gpu k35   sieve" "use --slow"
    fi
fi

# ---- report ----------------------------------------------------------------
echo
echo "============================================================"
printf '%s' "$RESULTS" | while IFS='|' read -r st name detail; do
    [ -z "$st" ] && continue
    case "$st" in
        PASS) printf '  \033[32m%-4s\033[0m %-18s %s\n' "$st" "$name" "$detail" ;;
        FAIL) printf '  \033[31m%-4s\033[0m %-18s %s\n' "$st" "$name" "$detail" ;;
        *)    printf '  \033[33m%-4s\033[0m %-18s %s\n' "$st" "$name" "$detail" ;;
    esac
done
echo "============================================================"
echo "  $PASS passed, $FAIL failed, $SKIP skipped"
echo

if [ "$KEEP" = 0 ]; then
    echo "(generated challenges left in data/challenges — remove with 'rm -rf data')"
fi

[ "$FAIL" -eq 0 ] || exit 1
echo "All solvers produce correct roots on this machine."
echo "Speedups at ${SIZE} bits are far below 1x by design — see README.md."
