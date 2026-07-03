#!/bin/bash -l
#SBATCH --job-name=minroot_tstep
#SBATCH --output=minroot_tstep_%j.out
#SBATCH --error=minroot_tstep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# T-step VDF comparison (empirical) — sequential solve vs smoothness attack
# across a sweep of T and prime size.
#
# Demonstrates the full-VDF delay model end-to-end: the honest evaluator runs
# T sequential modroots; the attacker computes each modroot via the smoothness
# method. We sweep T to show how the sequential delay and the attack delay both
# scale with T (the per-step ratio is what matters — see the wiki page
# t-step-vdf-comparison). The tool is SERIAL (1 core); the HPC just gives a long,
# undisturbed run since large T takes a while.
#
# Feasibility: the attack needs ~1/rho(u) blinding attempts per step,
# u = |p| / log2(B). We pick (bits, B) so u = 4 (~150 attempts/step after the
# 2026-06-12 trial_divide fix; before it, a tool bug inflated this ~1000x and
# the job timed out):
#     64-bit  ->  B = 2^16 = 65536       (~15 ms/step;  T up to 50000)
#     96-bit  ->  B = 2^24 = 16777216    (~2.3 s/step;  T up to 1000; ~1.1M-prime table)
# Larger primes need B >= 2^32 (cbrt table too large for this serial tool).
# Per-config T lists keep the whole sweep to ~1.5 h (SBATCH --time=6h has margin).
#
# Output: data/timelock/tstep_<jobid>/  with a tstep_results.csv for plotting.
# Submit:  sbatch scripts/timelock_tstep.slurm
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0
# OpenSSL is available system-wide on AION nodes (no module needed).

# Force '.' as the decimal separator so awk printf never writes "0,00" into the
# comma-delimited CSV (some node locales default to a comma).
export LC_ALL=C

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Build (Makefile now portable: -D_POSIX_C_SOURCE for clock_gettime)
# ---------------------------------------------------------------------------
echo "=== Build ==="
make -C tools/timelock || { echo "build failed"; exit 1; }
TOOL=$SLURM_SUBMIT_DIR/tools/timelock/minroot-timelock
echo ""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT=$SLURM_SUBMIT_DIR/data/timelock/tstep_${SLURM_JOB_ID}
mkdir -p $OUT
CSV=$OUT/tstep_results.csv
echo "bits,B,u,T,solve_s,attack_s,speedup,avg_attempts_per_step" > $CSV

# "bits B Tlist" — u = |p|/log2(B) = 4 (~150 attempts/step after the trial_divide fix).
# Tlist is sized to the measured per-step cost so the whole sweep finishes in ~1.5 h:
#   64-bit / B=2^16 : ~15 ms/step  -> T up to 50000  (~12 min for the largest)
#   96-bit / B=2^24 : ~2.3 s/step  -> T up to 1000   (~38 min for the largest;
#                                      cbrt table is ~1.1M entries, ~43 MB)
CONFIGS=(
    "64 65536    1000,2000,5000,10000,20000,50000"
    "96 16777216 100,200,500,1000"
)

echo "=== Job info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Configs (bits B Tlist):"
printf '          %s\n' "${CONFIGS[@]}"
echo "CSV:      $CSV"
echo ""

# ---------------------------------------------------------------------------
# One (bits, B, T) point: generate -> sequential solve -> smoothness attack
# ---------------------------------------------------------------------------
run_point() {
    local bits=$1 B=$2 T=$3
    local u=$(awk "BEGIN{printf \"%.1f\", $bits/(log($B)/log(2))}")
    local pz=$OUT/p_${bits}b_T${T}.mrt
    local slog=$OUT/solve_${bits}b_T${T}.log
    local alog=$OUT/attack_${bits}b_T${T}.log

    echo "------------------------------------------------------------"
    echo "  bits=$bits  B=$B  u=$u  T=$T"
    echo "------------------------------------------------------------"

    $TOOL generate -b $bits -T $T -e 5 -m "tstep ${bits}b T=${T}" $pz

    echo "--- sequential solve ---"
    $TOOL solve $pz | tee $slog
    echo "--- smoothness attack (B=$B) ---"
    $TOOL attack -B $B $pz | tee $alog
    echo ""

    # Parse "=== Solved in X s ===" / "=== Attack solved in Y s ===" / "avg Z per step"
    local ss=$(awk '/=== Solved in/{print $4}' $slog)
    local as=$(awk '/=== Attack solved in/{print $5}' $alog)
    local av=$(awk -F'avg ' '/avg .* per step/{split($2,a," ");print a[1]}' $alog)
    local sp=""
    if [ -n "$ss" ] && [ -n "$as" ]; then
        sp=$(awk "BEGIN{ if ($as>0) printf \"%.3f\", $ss/$as; else print \"\" }")
    fi
    echo "${bits},${B},${u},${T},${ss},${as},${sp},${av}" >> $CSV
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
for cfg in "${CONFIGS[@]}"; do
    set -- $cfg
    bits=$1; B=$2; Tlist=$3
    for T in ${Tlist//,/ }; do
        run_point $bits $B $T
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "T-step sweep complete (job ${SLURM_JOB_ID})."
echo "Results: $CSV"
echo ""
column -s, -t $CSV 2>/dev/null || cat $CSV
echo ""
echo "speedup = solve_s / attack_s  (>1 means the attack beats the sequential VDF delay)"
echo "Expect solve_s and attack_s both ~linear in T; the ratio is the per-step T-step result."
echo "============================================================"

# Edit: 2026-06-13
