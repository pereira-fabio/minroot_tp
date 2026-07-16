#!/bin/bash -l
#SBATCH --job-name=minroot_merge_cbrt
#SBATCH --output=minroot_merge_cbrt_%j.out
#SBATCH --error=minroot_merge_cbrt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

# ---------------------------------------------------------------------------
# One-time merge of the split cbrt caches into one full cache per (size, exp).
# (Advisor M1, 2026-07-16.)  Pure file I/O over caches already on the HPC —
# no cbrt is recomputed.
#
#   data/cbrt/<variant>_e<exp>/N<n>/<file>.cbrt*.<rank>   (existing split)
#        -> data/cbrt/cbrt_<size>_e<exp>.cbrt             (one merged full cache)
#
# k16 is skipped (its dense table is a different, binary format).  A cbrt
# conflict (same prime, two values => different moduli) aborts without writing.
#
# Run:   sbatch scripts/aion/merge_cbrt.slurm
# ---------------------------------------------------------------------------

cd $SLURM_SUBMIT_DIR

# Python 3 (stdlib only — no third-party packages needed). The login shell
# (bash -l) picks up the conda `base` python; if `python3` is not found, load
# your Python module here, e.g.:  module load lang/Python/3.11.3-GCCcore-12.3.0
PY=$(command -v python3 || true)
if [ -z "$PY" ]; then
    echo "python3 not found on PATH — load a Python module in this script and resubmit."
    exit 1
fi
echo "Using python3: $PY"
$PY --version

CBRT_ROOT=data/cbrt
echo "=== cbrt cache tree before merge ==="
du -sh $CBRT_ROOT 2>/dev/null
ls -d $CBRT_ROOT/*_e*/ 2>/dev/null | sed 's/^/  /'
echo ""

# 1) Dry run — preview counts + surface any cbrt conflicts (writes nothing).
echo "=== DRY RUN (preview) ==="
$PY analysis/merge_cbrt_cache.py --dry-run $CBRT_ROOT

# 2) Real merge — writes data/cbrt/cbrt_<size>_e<exp>.cbrt.
#    Exits non-zero (and writes nothing) if any modulus-mismatch conflict is found.
echo ""
echo "=== REAL MERGE ==="
$PY analysis/merge_cbrt_cache.py $CBRT_ROOT
rc=$?
if [ $rc -ne 0 ]; then
    echo "Merge aborted (exit $rc) — likely a cbrt conflict. No full cache written."
    exit $rc
fi

echo ""
echo "=== merged full caches ==="
ls -lh $CBRT_ROOT/cbrt_*_e*.cbrt 2>/dev/null | sed 's/^/  /'
echo "Done."

# Edit: 2026-07-16
