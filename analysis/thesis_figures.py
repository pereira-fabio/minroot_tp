#!/usr/bin/env python3
"""Curated thesis figures (Overleaf-ready PDFs, no embedded title — the LaTeX
\\caption supplies it) at the exact paths the THESIS.md placeholders reference.

DATA-DRIVEN (2026-07-23): every curve is computed directly from the benchmark
JSON on disk (the 2026-07-15 batch under results/results/{aion,iris}/…), NOT
hand-transcribed. Speedups follow the results-join methodology: the baseline is
the median AION N1 (2-rank) e3 `mpz_powm` baseline per size, and every point is
BASE[size] / warm_avg.total_ms (or /factor_ms for the GPU "sieve step" curves).
Point the loader at a different tree with  MINROOT_RESULTS=<dir>.

The plotting code (layout, styling, annotations) is unchanged from the prior
version — only the numbers now come from JSON, so the figures stay current with
the server automatically and can never drift from the data again.

ERROR BARS (2026-07-31): every measured series carries a 95% CONFIDENCE INTERVAL.
Built from SE = std/sqrt(n) over metadata.runs_warm warm runs, scaled by Student's
t at 97.5% with n-1 df -- NOT 1.96, because n is small and non-uniform here
(9, 14, 19, 29, 39 or 49), so the multiplier runs 2.306..2.011. Each plotted point
is a ratio speedup = BASE/T with BASE held fixed, so the relative error passes
straight through: err(speedup) = speedup * err(T)/T.

The interval is taken on log(speedup) and mapped back (ERRMODE=ci95log, default):
[v*exp(-ts), v*exp(+ts)]. Speedup is a strictly positive ratio and timing jitter is
multiplicative, so log-normal is the natural model; the lower bound can never reach
zero, and on the log axes used throughout the bar renders symmetrically. Taken
linearly instead (ERRMODE=ci95), alg4 table at 24576b has ts>1 beyond 16 ranks and
its interval would run to -5.6x. Set MINROOT_ERRMODE=ci95|se to switch.

Median interval is 0.3% of the value -- the benchmark is highly repeatable. The
exception is k16, at 1.9-2.7% everywhere because its sieve is sub-millisecond at
every modulus so timer granularity never becomes negligible.

CACHE-MISS FILTER (2026-07-31): warm runs with warm_runs.cbrt_misses > 0 are
excluded before the mean and spread are taken -- see _warm_stats. A miss costs one
full-width mpz_powm (1.8 s at 24576b against a ~15 ms steady-state reconstruction),
so such a run measures cache population rather than the steady state this thesis
reports. It is not a cosmetic filter: one miss-affected run in 49 was dragging
alg4's table method at 24576b from 49.05x down to 16.24x at 128 ranks, with a 134%
confidence interval that collapses to 0.14% once excluded.

Produces:
  figures/k16_paperscale_speedup.pdf   (§7.2 — k16 paper-scale scaling)
  figures/k16_large_moduli_speedup.pdf (§7.2 / §7.11.1 — k16 large-moduli scaling)
  figures/k16_gpu_speedup.pdf          (§7.12 — single-V100 k16 factor-only)
  figures/k16_cpu_vs_gpu.pdf           (§7.12 — k16 GPU sieve/total vs AION CPU)
  figures/k32_paperscale_speedup.pdf   (§7.3 — k32 paper-scale scaling, all <1×)
  figures/k32_large_moduli_speedup.pdf (§7.3 / §7.11.2 — k32 large-moduli scaling)
  figures/k32_cpu_vs_gpu.pdf           (§7.3 / §7.12 — k32 GPU vs AION CPU, full range)
  figures/{k35,alg3,alg4,alg7}_paperscale_speedup.pdf   (§7.4–7.6, §6.2 — data-driven)
  figures/{k35,alg3,alg4,alg7}_large_moduli_speedup.pdf (§7.4–7.6, §6.2 — data-driven)
  figures/{k35,alg3,alg4,alg7}_cpu_vs_gpu.pdf           (§7.4–7.6, §6.2 — data-driven)
  figures/large_moduli_scaling.pdf     (§7.11 — all six algorithms, sieve, 24576b)
  figures/gap1_two_regime.pdf          (§7.9 — k32 sieve speedup vs modulus)

Usage: python3 thesis_figures.py <FIG_DIR> [--png]   (env: MINROOT_RESULTS=results/results)
"""
import os, sys, json, re, glob, math, statistics, collections
from functools import lru_cache
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

# ── typography ───────────────────────────────────────────────────────────────
# Serif at 14 pt, Times where available, so the figures sit closer to the body
# text than matplotlib's DejaVu Sans default.
#
# 14 pt is chosen against the INCLUSION width, not the page: the two-panel
# figures are 11 in wide and included at 1.3\textwidth (~8.5 in), so they are
# scaled DOWN by ~0.77 and 14 pt renders at ~10.8 pt beside 10 pt body text.
# The single-panel figures (6.4 in at 0.8\textwidth) land at ~11.4 pt.
#
# Times New Roman is not redistributable and is absent on most Linux hosts, so
# the family list falls through to the metric-compatible clones (Nimbus Roman is
# URW's Times, Liberation Serif is the Croscore equivalent). Whichever is found
# first is also handed to mathtext, which unlike font.serif takes a single name
# and warns loudly if that name does not resolve.
_SERIF_PREFS = ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L",
                "Liberation Serif", "FreeSerif", "DejaVu Serif"]

def _first_available(names):
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for n in names:
        if n in have:
            return n
    return names[-1]

_SERIF = _first_available(_SERIF_PREFS)

plt.rcParams.update({"font.size": 14})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = _SERIF_PREFS
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = _SERIF
plt.rcParams["mathtext.it"] = f"{_SERIF}:italic"
plt.rcParams["mathtext.bf"] = f"{_SERIF}:bold"

FIGDIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "figures"
EXT = "png" if "--png" in sys.argv else "pdf"

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LAYER — compute every speedup from the benchmark JSON on disk.
#  results-join methodology: BASE[size] = median AION N1 e3 powm baseline;
#  speedup = BASE[size] / warm_avg.<which>.  All reads cached (one open/file).
# ═══════════════════════════════════════════════════════════════════════════
RESULTS = os.environ.get("MINROOT_RESULTS", "results/results")
EXP = 3
NODES = [1, 2, 4, 8, 16, 32, 64]           # AION node counts …
RANKS = [2, 4, 8, 16, 32, 64, 128]         # … = these MPI-rank counts (x-axis)

SIZES = [256, 384, 512, 768, 1024, 2048]                 # paper-scale moduli
LARGE_SIZES = [4096, 8192, 16384, 24576]                 # large-moduli (current canonical)
SIZES_ALL = SIZES + LARGE_SIZES                          # the full 10-size sweep

@lru_cache(maxsize=None)
def _ls(d):
    try: return tuple(os.listdir(d))
    except OSError: return ()

@lru_cache(maxsize=None)
def _load(path):
    try: return json.load(open(path))
    except Exception: return None

def _fsize(name):
    m = re.search(r"_(\d+)(?:_e\d+)?\.json$", name)
    return int(m.group(1)) if m else None

def _filetag(variant):
    # IRIS files for k16/k32/k35 all live in ONE dir (iris/alg2/G<k>/…), so
    # method+size is NOT enough to pick the right file — the filename's leading
    # algorithm tag (alg2_k16 / alg2_k32 / alg2_k35 / alg3 / alg4 / alg7) does.
    return {"k16": "alg2_k16", "k32": "alg2_k32", "k35": "alg2_k35"}.get(variant, variant)

def _find(d, tag, method, suffix):
    """First *.json in dir d named '<tag>_<method>_…<suffix>'.

    AION writes alg7 with an extra '_mpi' segment (alg7_mpi_sieve_…) while IRIS
    writes it plain (alg7_sieve_…), so both spellings are accepted. Without this
    every alg7 CPU curve silently came back empty. The plain form is tried first
    so it wins wherever both exist."""
    for pre in (f"{tag}_{method}_", f"{tag}_mpi_{method}_"):
        for nm in _ls(d):
            if nm.startswith(pre) and nm.endswith(suffix):
                return os.path.join(d, nm)
    return None

def _aion_dir(variant, node):
    parent = "alg2" if variant in ("k16", "k32", "k35") else variant
    g = sorted(glob.glob(f"{RESULTS}/aion/{parent}/{variant}_*_N{node}"))
    return g[0] if g else None

def _iris_dirs(alg, gcount, hw):
    """Candidate dirs for one (alg, gpu-count, hw), NEWEST FIRST.

    The L40 sweep was repeated several times into dated leaves (l40, l40-25-07,
    l40-27-07, l40-28-07, l40-29-07). Later runs supersede earlier ones, and one
    difference is substantive rather than cosmetic: the original 2026-07-17 l40/
    sweep predates the k35 work-balanced multi-GPU split (alg2_k35_cuda.cu, "Work-
    balanced upper-slice split"), so its k35 numbers measure a configuration in
    which GPU 0 carried Phase 1 *plus* a full quarter of the upper range and
    pinned factor_ms. Nine later runs agree on 1416-1419 ms against that job's
    1799 ms; every other variant reproduces across all of them to 0.3%.

    Sorting is plain lexical (reverse), not date-parsed, because one leaf is
    named l40-27-09 — a month typo for July. Lexical ordering puts the day field
    first and so still ranks it correctly; a date parser would read it as
    September and wrongly prefer it. The bare 'l40' leaf is a prefix of the
    dated ones, so it sorts last = oldest, which is what we want.
    """
    parent = "alg2" if alg in ("k16", "k32", "k35") else alg
    base = f"{RESULTS}/iris/{parent}/G{gcount}"
    if not hw:
        return [base]
    ds = sorted(glob.glob(f"{base}/{hw}*"), reverse=True)
    return ds or [f"{base}/{hw}"]

def _build_base():
    acc = collections.defaultdict(list)
    for v in ("k16", "k32", "k35", "alg3", "alg4", "alg7"):
        d = _aion_dir(v, 1)
        if not d:
            continue
        for nm in _ls(d):
            if "_powm_" not in nm or not nm.endswith(".json"):
                continue
            j = _load(os.path.join(d, nm))
            if not j or j.get("metadata", {}).get("exponent") != EXP:
                continue
            b = j.get("metadata", {}).get("powm_baseline_ms")
            if b:
                acc[_fsize(nm)].append(b)
    return {s: statistics.median(v) for s, v in acc.items() if v}

BASE = _build_base()

# ── error bars (advisor requests 2026-07-30, 2026-07-31) ─────────────────────
# ERRMODE selects what the bars mean:
#   "ci95log" — 95% CI built on log(speedup) and mapped back   (DEFAULT)
#   "ci95" — the same interval taken linearly in the speedup itself
#   "se"   — +/-1 standard error of the mean
#
# Both start from SE = std / sqrt(n) over the n warm runs, read from
# warm_stats[which].std and metadata.runs_warm.
#
# For the CI the multiplier is Student's t at 97.5% with n-1 degrees of freedom,
# NOT the normal 1.96. That matters here: n is not fixed across the corpus (AION
# runs carry 9, 14, 19, 29, 39 or 49 warm runs), so the multiplier ranges from
# 2.306 at n=9 down to 2.011 at n=49. Using 1.96 throughout would understate the
# narrowest-sample intervals by ~18%.
#
# Every plotted quantity is a RATIO, speedup = BASE/T, with the baseline held
# fixed (the thesis treats the median N1 powm as a reference constant, not a
# random variable), so first-order propagation preserves the relative error:
#       speedup = B/T   =>   err(speedup) = |d(B/T)/dT| · err(T)
#                                         = (B/T²)·err(T) = speedup · err(T)/T.
# Returns None when a run reported no spread (n < 2) so the caller omits the bar.
ERRMODE = os.environ.get("MINROOT_ERRMODE", "ci95log")   # ci95log | ci95 | se

# Student's t, two-sided 95%, by degrees of freedom. Exact for df <= 30, then
# anchors; _t95 takes the largest anchor <= df, which errs wide (conservative).
_T95 = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365,
        8:2.306, 9:2.262, 10:2.228, 11:2.201, 12:2.179, 13:2.160, 14:2.145,
        15:2.131, 16:2.120, 17:2.110, 18:2.101, 19:2.093, 20:2.086, 21:2.080,
        22:2.074, 23:2.069, 24:2.064, 25:2.060, 26:2.056, 27:2.052, 28:2.048,
        29:2.045, 30:2.042, 35:2.030, 38:2.024, 40:2.021, 45:2.014, 48:2.011,
        50:2.009, 60:2.000,
        80:1.990, 100:1.984, 120:1.980}
_T95_KEYS = sorted(_T95)

def _t95(df):
    if df <= 0:
        return None
    if df in _T95:
        return _T95[df]
    if df > _T95_KEYS[-1]:
        return 1.960                      # normal limit
    lo = max(k for k in _T95_KEYS if k <= df)
    return _T95[lo]

def _warm_stats(j, which):
    """(mean, sample std, n) for `which` over the warm runs that were pure cache hits.

    A run that misses the cube-root cache pays one full-width mpz_powm per miss --
    1.8 s at 24576 bits, against a ~15 ms steady-state reconstruction -- so it
    measures cache population, not the steady-state cost this thesis reports (see
    the cache policy in the methodology). Such runs are identified objectively by
    warm_runs.cbrt_misses > 0 and dropped; the mean and spread are recomputed from
    what remains, so the stored warm_avg/warm_stats are bypassed rather than
    patched.

    This matters for alg4's table method: 7 of its 70 AION configs carry a miss,
    all of them at 24576 bits and all the same challenge seen once per rank sweep
    (7 affected runs of 2080). That one run took 2 misses and spent 3.64 s in
    reconstruction against a 15.2 ms median, dragging the reported speedup from
    49.1x to 16.2x at 128 ranks.

    NB (2026-08-03): an earlier version of this note read "77 of 140 configs" and
    blamed a COMPOSITE cofactor factored by Pollard's rho. Both were wrong, and by
    the same slip: the counting glob 'alg4_*' also matched alg4_multi_* (the §6.2
    extended-cofactor variant, whose 70 table configs ALL miss -- 7 + 70 = 77).
    Algorithm 4 as specified accepts the same SINGLE medium prime as Algorithm 3;
    the composite cofactor belongs to alg4_multi. alg3 and alg4 agree on
    steady-state reconstruction to 1%, and alg3's zero misses reflect a complete
    pre-populated cache for that modulus rather than a different acceptance rule.
    Match 'alg4_[0-9]*' when counting, never 'alg4_*'.

    IRIS JSONs do not record cbrt_misses, so those fall through to the stored
    summary unchanged (they carry no misses in any case).
    """
    wr = j.get("warm_runs") or {}
    vals, miss = wr.get(which), wr.get("cbrt_misses")
    if vals and miss and len(vals) == len(miss):
        keep = [v for v, m in zip(vals, miss) if not m]
        if len(keep) >= 2:
            mean = sum(keep) / len(keep)
            var = sum((x - mean) ** 2 for x in keep) / (len(keep) - 1)
            return mean, math.sqrt(var), len(keep)
    mean = (j.get("warm_avg") or {}).get(which)
    sd   = ((j.get("warm_stats") or {}).get(which) or {}).get("std")
    n    = (j.get("metadata") or {}).get("runs_warm") or 0
    return mean, sd, n

def _err_ratio(value, mean, sd, n):
    """Plotted error half-width on a ratio, from the (mean, std, n) of the divisor."""
    if value is None or sd is None or not mean or not n or n < 2:
        return None
    se_rel = (sd / math.sqrt(n)) / mean     # relative standard error
    if ERRMODE == "se":
        return value * se_rel
    return value * se_rel * _t95(n - 1)     # 95% CI half-width

# Minimum cache-clean runs required before a point is plotted at all.
#
# _warm_stats drops miss-affected runs, but it falls back to the stored summary
# when fewer than 2 survive -- which silently republishes the CONTAMINATED mean.
# That fallback is invisible in the output and it bit alg7's table method, whose
# cube-root cache was never populated: 0 clean runs of 48 at 24576b, 0 of 49 at
# 16384b, yet the loader still returned a number (0.075x / 0.095x) that was read
# as an algorithmic result. Its factoring matches the sieve to within 4% at every
# size, so the deficit was never in the prime access pattern at all.
#
# A point below this threshold is returned as None and simply not drawn. Five is
# a judgement call, set against the n in {9,14,19,29,39,49} the corpus otherwise
# carries: below it the mean is not worth a confidence interval.
MIN_CLEAN = 5

def _clean_count(j, which="total_ms"):
    """How many warm runs of `which` were free of cube-root cache misses.

    Returns None when the JSON does not record cbrt_misses (IRIS), meaning
    'no evidence of contamination' rather than 'zero clean runs'."""
    wr = j.get("warm_runs") or {}
    vals, miss = wr.get(which), wr.get("cbrt_misses")
    if not vals or not miss or len(vals) != len(miss):
        return None
    return sum(1 for m in miss if not m)

def cpu(variant, method, size, node, which="total_ms", err=False):
    """AION CPU speedup for one (variant, method, size, node).
    With err=True, returns the 95% CI half-width on that speedup instead.
    Returns None when fewer than MIN_CLEAN cache-clean runs back the point."""
    d = _aion_dir(variant, node)
    if not d:
        return None
    f = _find(d, _filetag(variant), method, f"_{size}.json")
    if not f:
        return None
    j = _load(f)
    if not j or j.get("metadata", {}).get("exponent") != EXP:
        return None
    c = _clean_count(j, which)
    if c is not None and c < MIN_CLEAN:
        return None
    t, sd, n = _warm_stats(j, which)
    b = BASE.get(size)
    v = b / t if (b and t) else None
    return _err_ratio(v, t, sd, n) if err else v

def cpu_row(variant, method, size, which="total_ms", err=False):
    return [cpu(variant, method, size, n, which, err) for n in NODES]

def gpu(alg, method, size, hw="v100", gcount=4, which="total_ms", err=False):
    """IRIS GPU speedup vs the same AION N1 baseline (the GPU's own baseline is broken).
    Uses the requested GPU count and walks DOWN if that exact count is absent:
      • V100 lives in the pre-split non-leaf dirs (G1..G3) and in G4/v100; alg3's
        V100 sweep only reached G3, so its 4-GPU point falls back to G3 (alg3 GPU
        is reconstruction-bound, so G3 ≈ G4).  Single-GPU headlines pass gcount=1.
      • L40 only exists at G4/l40 — never fall back to a non-leaf dir (that is V100)."""
    cands = []
    for g in range(gcount, 0, -1):
        cands.extend(_iris_dirs(alg, g, hw))         # hw leaves, newest first
        if hw == "v100":
            cands.extend(_iris_dirs(alg, g, None))   # pre-split non-leaf V100 dir
    for d in cands:
        f = _find(d, _filetag(alg), method, f"_{size}_e{EXP}.json")
        if not f:
            continue
        j = _load(f)
        if not j:
            continue
        t, sd, n = _warm_stats(j, which)
        b = BASE.get(size)
        if b and t:
            v = b / t
            return _err_ratio(v, t, sd, n) if err else v
    return None

def _peak(row):
    vals = [(RANKS[i], v) for i, v in enumerate(row) if v]
    return max(vals, key=lambda p: p[1]) if vals else (None, None)

SIZE_COLOR = {s: c for s, c in zip(SIZES, plt.cm.viridis([i/5 for i in range(6)]))}

# ── §7.2  k16 table-method speedup_total (rows keyed by modulus, cols = RANKS) ─
K16_PAPERSCALE = {s: cpu_row("k16", "table", s) for s in SIZES}
K16_PAPERSCALE_ERR = {s: cpu_row("k16", "table", s, err=True) for s in SIZES}
# ── §7.12  single-V100 k16 factor-only speedup (the "sieve step" headline) ─────
K16_GPU = {s: gpu("k16", "sieve", s, "v100", 1, "factor_ms") for s in SIZES}
K16_GPU_ERR = {s: gpu("k16", "sieve", s, "v100", 1, "factor_ms", err=True) for s in SIZES}


def baseline_line(ax):
    ax.axhline(1.0, color="black", ls="--", lw=1.1, alpha=0.8)
    ax.text(0.985, 1.06, "sequential baseline (1×)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9.8, color="black", alpha=0.75)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda y, _: f"{y:g}×"))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(True, which="major", ls="-", alpha=0.22); ax.set_axisbelow(True)


def eplot(ax, x, y, ye, fmt, **kw):
    """ax.plot() with ±1 standard-error bars.

    Falls back to a plain plot when no series member has a usable SE. Missing
    per-point SEs become 0.0 (no bar drawn there) since matplotlib rejects None
    inside yerr. Bars are clipped so the lower cap cannot reach <= 0, which would
    break the log y-axis every figure here uses."""
    if ye is None or not any(e for e in ye):
        return ax.plot(x, y, fmt, **kw)
    lo, hi = [], []
    for v, e in zip(y, ye):
        e = e or 0.0
        if not v:
            lo.append(e); hi.append(e); continue
        if ERRMODE == "ci95log":
            # Interval built on log(speedup) and mapped back. The quantity is a
            # positive ratio and run-to-run timing variation is multiplicative,
            # so a log-normal model is the natural one. e/v is the relative
            # half-width, so the interval is v*exp(±e/v): the lower bound can
            # never reach zero, and on the log axis every figure here uses, the
            # bar renders SYMMETRICALLY instead of appearing skewed downward.
            r = e / v
            lo.append(v * (1.0 - math.exp(-r)))
            hi.append(v * (math.exp(r) - 1.0))
        else:
            hi.append(e)
            # keep the lower cap strictly positive on a log scale
            lo.append(min(e, v * 0.999))
    # A missing point is None (no data, or too few cache-clean runs to average).
    # errorbar does arithmetic on y, so None has to become NaN -- matplotlib then
    # leaves a gap in the line instead of raising.
    y = [float("nan") if v is None else v for v in y]
    return ax.errorbar(x, y, yerr=[lo, hi], fmt=fmt,
                       capsize=2.5, elinewidth=0.9, capthick=0.9, **kw)


def peak_marker(ax, x, y, label=None):
    """Design-point peak: black diamond plus its value, with headroom.

    The label sits above the marker, so on an autoscaled axis it lands on (or
    over) the top spine whenever the peak is itself the highest point plotted --
    which is exactly the case on the TABLE panels, since the table method beats
    the sieve everywhere. Raising the top limit afterwards keeps the label inside
    the axes. On the side-by-side figures the two panels share a y-axis, so the
    larger of the two peaks wins and both panels keep the same scale."""
    ax.plot(x, y, "D", color="#111111", ms=7, zorder=5)
    ax.text(x, y * 1.17, label or f"{y:.1f}×", ha="right", va="bottom",
            fontsize=10.5, color="#111111")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, y * 1.9))


def savefig(fig, name, tight=True):
    """tight=False for figures that place their own colourbar axes — tight_layout
    reflows them on top of the panels."""
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, f"{name}.{EXT}")
    if tight:
        fig.tight_layout()
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_k16_paperscale():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for s in SIZES:
        eplot(ax, RANKS, K16_PAPERSCALE[s], K16_PAPERSCALE_ERR[s], "o-", color=SIZE_COLOR[s], label=f"{s}b")
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    ax.axvline(16, color="grey", ls=":", lw=1, alpha=0.7)
    baseline_line(ax)
    pk = max(v for v in K16_PAPERSCALE[2048] if v)
    ax.text(16, ax.get_ylim()[0] * 1.4, f" N8 peak ({pk:.1f}×) ", rotation=90,
            va="bottom", ha="right", fontsize=9.8, color="grey")
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup over sequential")
    ax.legend(title="modulus", fontsize=11.2, ncol=2, loc="lower left")
    savefig(fig, "k16_paperscale_speedup")


# ── §7.11.1  k16 large-moduli speedup_total; sieve ≡ table for k16 ───────────
K16_LARGE = {s: cpu_row("k16", "sieve", s) for s in LARGE_SIZES}
K16_LARGE_ERR = {s: cpu_row("k16", "sieve", s, err=True) for s in LARGE_SIZES}
LARGE_COLOR = {s: c for s, c in zip(LARGE_SIZES, plt.cm.viridis([i/3 for i in range(4)]))}


def fig_k16_large_moduli():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for s in LARGE_SIZES:
        eplot(ax, RANKS, K16_LARGE[s], K16_LARGE_ERR[s], "o-", color=LARGE_COLOR[s], label=f"{s}b")
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    # Log y-axis, tightened to the data (all ~20–190×, a single decade). Unlike the
    # paper-scale sibling, nothing here is near the 1× baseline, so we drop it and
    # let the curves + their growth-with-size fill the plot instead of being cramped
    # above an empty 1×–20× band.
    ax.set_yscale("log"); ax.set_ylim(18, 220)
    ax.set_yticks([20, 30, 50, 100, 200])
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda y, _: f"{y:g}×"))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(True, which="major", ls="-", alpha=0.22)
    ax.grid(True, which="minor", ls=":", alpha=0.12); ax.set_axisbelow(True)
    pk_rank, pk = _peak(K16_LARGE[24576])
    ax.axvline(pk_rank, color="grey", ls=":", lw=1, alpha=0.7)
    ax.text(pk_rank, 19, f" peak ({pk:.1f}× @24576b) ", rotation=90,
            va="bottom", ha="right", fontsize=9.8, color="grey")
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup over sequential")
    ax.legend(title="modulus", fontsize=11.2, ncol=2, loc="lower center")
    savefig(fig, "k16_large_moduli_speedup")


def fig_k16_gpu():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    xs = SIZES; ys = [K16_GPU[s] for s in SIZES]
    eplot(ax, xs, ys, [K16_GPU_ERR[s] for s in SIZES], "o-", color="#27ae60", lw=2, markersize=7,
          label="single V100 (k16, factor-only)")
    for s, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}×" if y >= 100 else f"{y:.1f}×", (s, y), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10.5, color="#1e7a44")
    ax.axhline(42, color="#c0392b", ls="--", lw=1.3, alpha=0.85)
    ax.text(xs[0], 46, "paper target 42×", color="#c0392b", fontsize=11.2, va="bottom")
    ax.set_xscale("log", base=2); ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    baseline_line(ax)
    ax.set_xlabel("modulus size (bits)")
    ax.set_ylabel("speedup over sequential (sieve step)")
    ax.legend(fontsize=11.2, loc="lower right")
    savefig(fig, "k16_gpu_speedup")


# ── §7.12  k16 GPU (4× V100 / 4× L40, end-to-end total) vs AION CPU, paper-scale ─
K16_GPU_TOTAL = {s: gpu("k16", "sieve", s, "v100", 4, "total_ms") for s in SIZES}
K16_GPU_L40   = {s: gpu("k16", "sieve", s, "l40",  4, "total_ms") for s in SIZES}
K16_GPU_TOTAL_ERR = {s: gpu("k16", "sieve", s, "v100", 4, "total_ms", err=True) for s in SIZES}
K16_GPU_L40_ERR   = {s: gpu("k16", "sieve", s, "l40",  4, "total_ms", err=True) for s in SIZES}


def fig_k16_cpu_vs_gpu():
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    # GPU: sieve step (the big factor-only headline) vs honest end-to-end (recon-bound)
    eplot(ax, SIZES, [K16_GPU[s] for s in SIZES], [K16_GPU_ERR[s] for s in SIZES],
          "o-", color="#1a9850", lw=2, ms=7, label="V100 GPU — sieve step (factor_ms)")
    eplot(ax, SIZES, [K16_GPU_L40[s] for s in SIZES], [K16_GPU_L40_ERR[s] for s in SIZES],
          "o-.", color="#8e44ad", lw=1.7, ms=5, label="L40 GPU — end-to-end (total)")
    eplot(ax, SIZES, [K16_GPU_TOTAL[s] for s in SIZES], [K16_GPU_TOTAL_ERR[s] for s in SIZES],
          "o--", color="#66bd63", lw=1.8, ms=6, label="V100 GPU — end-to-end (total)")
    # CPU: best rank config (N8) and the ceiling (N64), both end-to-end
    eplot(ax, SIZES, [K16_PAPERSCALE[s][3] for s in SIZES],
          [K16_PAPERSCALE_ERR[s][3] for s in SIZES], "o-", color="#1f77b4", lw=1.8,
            label="AION CPU — N8 (16r, best)")
    eplot(ax, SIZES, [K16_PAPERSCALE[s][6] for s in SIZES],
          [K16_PAPERSCALE_ERR[s][6] for s in SIZES], "o-", color="#ff7f0e", lw=1.8,
            label="AION CPU — N64 (128r)")
    ax.axhline(42, color="#c0392b", ls=":", lw=1.2, alpha=0.8)
    ax.text(1150, 46, "paper target 42×", color="#c0392b", fontsize=10.5, va="bottom")
    ax.set_xscale("log", base=2); ax.set_xticks(SIZES)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    baseline_line(ax)
    ax.set_xlabel("modulus size (bits)")
    ax.set_ylabel("speedup over sequential")
    ax.legend(fontsize=10.2, loc="upper left")
    savefig(fig, "k16_cpu_vs_gpu")


# ── §7.11.1  large-moduli strong scaling at 24576b (all six algorithms, sieve) ─
LM_24576 = {alg: (RANKS, cpu_row(alg, "sieve", 24576))
            for alg in ("k16", "k32", "alg3", "alg4", "alg7", "k35")}
LM_24576_ERR = {alg: cpu_row(alg, "sieve", 24576, err=True)
               for alg in ("k16", "k32", "alg3", "alg4", "alg7", "k35")}
LM_COLOR = {"k16": "#1f77b4", "k32": "#ff7f0e", "alg3": "#d62728",
            "alg4": "#9467bd", "alg7": "#2ca02c", "k35": "#8c564b"}

def fig_large_moduli():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for alg, (xs, ys) in LM_24576.items():
        eplot(ax, xs, ys, LM_24576_ERR[alg], "o-", color=LM_COLOR[alg], label=alg)
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    ax.axvline(32, color="grey", ls=":", lw=1, alpha=0.7)
    baseline_line(ax)
    ax.text(32, ax.get_ylim()[0] * 1.4, " k16 peak (N16) ", rotation=90,
            va="bottom", ha="right", fontsize=9.8, color="grey")
    ax.axhline(20, color="#c0392b", ls="--", lw=1.0, alpha=0.7)
    ax.text(2, 21.5, "paper 20× (Algo 3/7 target)", color="#c0392b", fontsize=9.1, va="bottom")
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup over sequential")
    ax.legend(title="algorithm @ 24576b (sieve)", fontsize=10.5, ncol=2, loc="upper left")
    savefig(fig, "large_moduli_scaling")

# ── §7.9  Gap 1: k32 (128r, sieve) speedup vs modulus, both regimes ──────────
GAP1 = {s: cpu("k32", "sieve", s, 64) for s in SIZES_ALL if s >= 384}
GAP1_ERR = {s: cpu("k32", "sieve", s, 64, err=True) for s in SIZES_ALL if s >= 384}

def fig_gap1():
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    xs = sorted(GAP1); ys = [GAP1[s] for s in xs]
    ax.axvspan(300, 3000, color="#f2d4d4", alpha=0.4)      # sieve-dominated regime
    eplot(ax, xs, ys, [GAP1_ERR[s] for s in xs], "o-", color="#c0392b", lw=2,
          label="k32 sieve, 128 ranks")
    ax.axvline(3000, color="grey", ls=":", lw=1, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384, 24576])
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter()); ax.tick_params(axis="x", rotation=45)
    baseline_line(ax)
    big_s = GAP1.get(24576); big_t = cpu("k32", "table", 24576, 64)
    ax.text(430, 0.02, "paper-scale\n256–2048b:\nsieve dominates,\n$\\ll 1\\times$",
            fontsize=10.5, va="center", color="#7a2020")
    ax.text(9000, 4, f"large moduli:\nbaseline dominates\n→ {big_s:.0f}× ({big_t:.0f}× table)",
            fontsize=10.5, va="center", color="#1e5a1e")
    ax.text(3000, ax.get_ylim()[1] * 0.5, " break-even ≈3000b ", rotation=90,
            va="top", ha="right", fontsize=9.8, color="grey")
    ax.set_xlabel("modulus size (bits)"); ax.set_ylabel("speedup over sequential")
    ax.legend(fontsize=11.2, loc="lower right")
    savefig(fig, "gap1_two_regime")

# ── §7.3  k32 paper-scale speedup_total (sieve) + 2048b table ────────────────
K32_PAPERSCALE = {s: cpu_row("k32", "sieve", s) for s in SIZES}
K32_PAPERSCALE_ERR = {s: cpu_row("k32", "sieve", s, err=True) for s in SIZES}
K32_PAPERSCALE_2048_TABLE = cpu_row("k32", "table", 2048)
K32_PAPERSCALE_2048_TABLE_ERR = cpu_row("k32", "table", 2048, err=True)

def fig_k32_paperscale():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for s in SIZES:
        eplot(ax, RANKS, K32_PAPERSCALE[s], K32_PAPERSCALE_ERR[s], "o-", color=SIZE_COLOR[s], label=f"{s}b")
    eplot(ax, RANKS, K32_PAPERSCALE_2048_TABLE, K32_PAPERSCALE_2048_TABLE_ERR, "o--", color=SIZE_COLOR[2048], lw=1.7,
            label="2048b (table)")
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    baseline_line(ax)
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup over sequential")
    ax.legend(title="modulus (sieve)", fontsize=10.2, ncol=2, loc="upper left")
    savefig(fig, "k32_paperscale_speedup")


# ── §7.11.2  k32 large-moduli speedup_total (sieve | table) ──────────────────
K32_LARGE       = {s: cpu_row("k32", "sieve", s) for s in LARGE_SIZES}
K32_LARGE_TABLE = {s: cpu_row("k32", "table", s) for s in LARGE_SIZES}
K32_LARGE_ERR       = {s: cpu_row("k32", "sieve", s, err=True) for s in LARGE_SIZES}
K32_LARGE_TABLE_ERR = {s: cpu_row("k32", "table", s, err=True) for s in LARGE_SIZES}

def fig_k32_large_moduli():
    # side-by-side: sieve (left) vs table (right), shared log-y axis
    fig, (axs, axt) = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for s in LARGE_SIZES:
        eplot(axs, RANKS, K32_LARGE[s], K32_LARGE_ERR[s], "o-", color=LARGE_COLOR[s], label=f"{s}b")
        eplot(axt, RANKS, K32_LARGE_TABLE[s], K32_LARGE_TABLE_ERR[s], "o-", color=LARGE_COLOR[s], label=f"{s}b")
    for ax in (axs, axt):
        ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
        ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
        baseline_line(ax)
        ax.set_xlabel("MPI ranks")
    # design-point peak on each panel (N64 / 24576b)
    ps = K32_LARGE[24576][-1]; pt = K32_LARGE_TABLE[24576][-1]
    peak_marker(axs, 128, ps)
    peak_marker(axt, 128, pt)
    axs.set_ylabel("speedup over sequential")
    axs.set_title("sieve method", fontsize=14, fontweight="bold")
    axt.set_title("table method", fontsize=14, fontweight="bold")
    axs.legend(title="modulus", fontsize=11.2, ncol=2, loc="upper left")
    savefig(fig, "k32_large_moduli_speedup")


# ── §7.12  k32 GPU (4×V100 sieve / 4×L40 table) vs AION CPU N64, full range ──
K32_CPU_N64_SIEVE = [cpu("k32", "sieve", s, 64) for s in SIZES_ALL]
K32_CPU_N64_TABLE = [cpu("k32", "table", s, 64) for s in SIZES_ALL]
K32_GPU_G4        = [gpu("k32", "sieve", s, "v100", 4, "total_ms") for s in SIZES_ALL]
K32_GPU_L40       = [gpu("k32", "table", s, "l40",  4, "total_ms") for s in SIZES_ALL]
K32_CPU_N64_SIEVE_ERR = [cpu("k32", "sieve", s, 64, err=True) for s in SIZES_ALL]
K32_CPU_N64_TABLE_ERR = [cpu("k32", "table", s, 64, err=True) for s in SIZES_ALL]
K32_GPU_G4_ERR        = [gpu("k32", "sieve", s, "v100", 4, "total_ms", err=True) for s in SIZES_ALL]
K32_GPU_L40_ERR       = [gpu("k32", "table", s, "l40",  4, "total_ms", err=True) for s in SIZES_ALL]

def fig_k32_cpu_vs_gpu():
    # side-by-side: CPU sieve (left) vs CPU table (right), each vs the same GPUs.
    # GPU sieve/table agree within ~2% at large moduli (§7.12), so the two GPU
    # curves are shown identically in both panels.
    fig, (axs, axt) = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    eplot(axs, SIZES_ALL, K32_CPU_N64_SIEVE, K32_CPU_N64_SIEVE_ERR, "o-", color="#1f77b4", lw=1.9, label="AION CPU — N64 (128r)")
    eplot(axt, SIZES_ALL, K32_CPU_N64_TABLE, K32_CPU_N64_TABLE_ERR, "o-", color="#ff7f0e", lw=1.9, label="AION CPU — N64 (128r)")
    for ax in (axs, axt):
        eplot(ax, SIZES_ALL, K32_GPU_L40, K32_GPU_L40_ERR, "o-.", color="#8e44ad", lw=1.7, ms=5, label="IRIS 4×L40")
        eplot(ax, SIZES_ALL, K32_GPU_G4, K32_GPU_G4_ERR, "o--", color="#1a9850", lw=1.7, ms=6, label="IRIS 4×V100")
        ax.set_xscale("log", base=2); ax.set_xticks(SIZES_ALL)
        ax.get_xaxis().set_major_formatter(mt.ScalarFormatter()); ax.tick_params(axis="x", rotation=45)
        baseline_line(ax)
        ax.set_xlabel("modulus size (bits)")
        ax.legend(fontsize=10.5, loc="upper left")
    axs.set_ylabel("speedup over sequential")
    axs.set_title("sieve method", fontsize=14, fontweight="bold")
    axt.set_title("table method", fontsize=14, fontweight="bold")
    savefig(fig, "k32_cpu_vs_gpu")


# ═══════════════════════════════════════════════════════════════════════════
#  k35 / Algorithm 3 / Algorithm 4 / Algorithm 7 — data-driven (§7.11.6, §7.12)
#  Same three placements as k16/k32: paper-scale, large-moduli, GPU-vs-CPU.
# ═══════════════════════════════════════════════════════════════════════════
ALG_LABEL = {"k35": "k35", "alg3": "Algorithm 3", "alg4": "Algorithm 4", "alg7": "Algorithm 7"}
LM_SIZES = [4096, 8192, 16384, 24576]
LM_COLOR4 = {s: c for s, c in zip(LM_SIZES, plt.cm.viridis([0.1, 0.42, 0.72, 0.95]))}

# ── paper-scale sieve (7 ranks) + 2048b table ───────────────────────────────
PS = {alg: {s: cpu_row(alg, "sieve", s) for s in SIZES}
      for alg in ("k35", "alg3", "alg4", "alg7")}
PS_TABLE_2048 = {alg: cpu_row(alg, "table", 2048) for alg in ("k35", "alg3", "alg4")}
PS_ERR = {alg: {s: cpu_row(alg, "sieve", s, err=True) for s in SIZES}
         for alg in ("k35", "alg3", "alg4", "alg7")}
PS_TABLE_2048_ERR = {alg: cpu_row(alg, "table", 2048, err=True) for alg in ("k35", "alg3", "alg4")}

# ── full paper-scale TABLE sweep, for the variants shown as two panels ────────
# alg7 only. Its table method is the one case where sieve and table diverge for
# a reason worth a figure, and the MIN_CLEAN filter leaves the panel sparse --
# which is the honest rendering, since the missing cells have no uncontaminated
# runs to average. The large-moduli equivalent is NOT built: at 4096b and above
# only 4 of 28 cells clear MIN_CLEAN, too few to trace a rank sweep.
PS_BOTH = {alg: {s: cpu_row(alg, "table", s) for s in SIZES} for alg in ("alg7",)}
PS_BOTH_ERR = {alg: {s: cpu_row(alg, "table", s, err=True) for s in SIZES} for alg in ("alg7",)}

# ── large-moduli sieve (7 ranks); sizes 4096/8192/16384/24576 ────────────────
LM = {alg: {s: cpu_row(alg, "sieve", s) for s in LM_SIZES}
      for alg in ("k35", "alg3", "alg4", "alg7")}
# ── large-moduli TABLE method (k35/alg3/alg4) ────────────────────────────────
# alg7 is excluded because its table sweep is UNMEASURED at these sizes, not
# because it is slow: every run at 16384b and 24576b took cube-root cache misses
# and only 4 of 28 cells clear MIN_CLEAN, too few to trace a rank sweep. Its
# paper-scale table data does survive the filter and is plotted by fig_ps.
LM_TABLE = {alg: {s: cpu_row(alg, "table", s) for s in LM_SIZES}
            for alg in ("k35", "alg3", "alg4")}
LM_ERR = {alg: {s: cpu_row(alg, "sieve", s, err=True) for s in LM_SIZES}
         for alg in ("k35", "alg3", "alg4", "alg7")}
LM_TABLE_ERR = {alg: {s: cpu_row(alg, "table", s, err=True) for s in LM_SIZES}
               for alg in ("k35", "alg3", "alg4")}
# design-point peaks (N64/24576b): sieve, table.  alg7 has no usable table method.
LM_PEAK = {alg: (cpu(alg, "sieve", 24576, 64),
                 cpu(alg, "table", 24576, 64) if alg != "alg7" else None)
           for alg in ("k35", "alg3", "alg4", "alg7")}

# ── GPU-vs-CPU across the 10 canonical sizes: CPU N64 sieve/table, GPU G4 ────
#   cs/ct = AION CPU N64 sieve/table;  g = 4×V100 (sieve total);  l40 = 4×L40 total.
#   alg7's table sweep is cache-contaminated at large moduli → ct = None here
#   (single-panel figure); see the MIN_CLEAN note.
CVG = {}
for alg in ("k35", "alg3", "alg4", "alg7"):
    CVG[alg] = {
        "cs":  [cpu(alg, "sieve", s, 64) for s in SIZES_ALL],
        "ct":  ([cpu(alg, "table", s, 64) for s in SIZES_ALL]) if alg != "alg7" else None,
        "g":   [gpu(alg, "sieve", s, "v100", 4, "total_ms") for s in SIZES_ALL],
        "l40": [gpu(alg, "sieve", s, "l40",  4, "total_ms") for s in SIZES_ALL],
        "cs_se":  [cpu(alg, "sieve", s, 64, err=True) for s in SIZES_ALL],
        "ct_se":  ([cpu(alg, "table", s, 64, err=True) for s in SIZES_ALL]) if alg != "alg7" else None,
        "g_se":   [gpu(alg, "sieve", s, "v100", 4, "total_ms", err=True) for s in SIZES_ALL],
        "l40_se": [gpu(alg, "sieve", s, "l40",  4, "total_ms", err=True) for s in SIZES_ALL],
    }

def fig_ps(alg):
    if alg in PS_BOTH:      # side-by-side sieve | table (alg7)
        fig, (axs, axt) = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
        for s in SIZES:
            eplot(axs, RANKS, PS[alg][s], PS_ERR[alg][s], "o-", color=SIZE_COLOR[s], label=f"{s}b")
            eplot(axt, RANKS, PS_BOTH[alg][s], PS_BOTH_ERR[alg][s], "o-", color=SIZE_COLOR[s], label=f"{s}b")
        for ax in (axs, axt):
            ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
            ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
            baseline_line(ax); ax.set_xlabel("MPI ranks")
        axs.set_ylabel("speedup over sequential")
        axs.set_title("sieve method", fontsize=14, fontweight="bold")
        # The table panel is deliberately sparse: points backed by fewer than
        # MIN_CLEAN cache-clean runs are not drawn (see the MIN_CLEAN note).
        axt.set_title("table method  (cache-clean runs only)", fontsize=14, fontweight="bold")
        axs.legend(title=f"{ALG_LABEL[alg]} — modulus", fontsize=10.2, ncol=2, loc="upper left")
        savefig(fig, f"{alg}_paperscale_speedup")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for s in SIZES:
        eplot(ax, RANKS, PS[alg][s], PS_ERR[alg][s], "o-", color=SIZE_COLOR[s], label=f"{s}b")
    if alg in PS_TABLE_2048:
        eplot(ax, RANKS, PS_TABLE_2048[alg], PS_TABLE_2048_ERR[alg], "o--", color=SIZE_COLOR[2048], lw=1.7, label="2048b (table)")
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    baseline_line(ax)
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup over sequential")
    ax.legend(title=f"{ALG_LABEL[alg]} — modulus (sieve)", fontsize=10.2, ncol=2, loc="upper left")
    savefig(fig, f"{alg}_paperscale_speedup")

def fig_lm(alg):
    ps, pt = LM_PEAK[alg]
    if alg in LM_TABLE:  # side-by-side sieve | table (k35/alg3/alg4)
        fig, (axs, axt) = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
        for s in LM_SIZES:
            eplot(axs, RANKS, LM[alg][s], LM_ERR[alg][s], "o-", color=LM_COLOR4[s], label=f"{s}b")
            eplot(axt, RANKS, LM_TABLE[alg][s], LM_TABLE_ERR[alg][s], "o-", color=LM_COLOR4[s], label=f"{s}b")
        for ax in (axs, axt):
            ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
            ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
            baseline_line(ax); ax.set_xlabel("MPI ranks")
        peak_marker(axs, 128, ps)
        peak_marker(axt, 128, pt)
        axs.set_ylabel("speedup over sequential")
        axs.set_title("sieve method", fontsize=14, fontweight="bold")
        axt.set_title("table method", fontsize=14, fontweight="bold")
        axs.legend(title=f"{ALG_LABEL[alg]} — modulus", fontsize=10.5, ncol=2, loc="upper left")
        savefig(fig, f"{alg}_large_moduli_speedup")
        return
    # single-panel sieve (alg7 — no per-rank table sweep measured)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for s in LM_SIZES:
        eplot(ax, RANKS, LM[alg][s], LM_ERR[alg][s], "o-", color=LM_COLOR4[s], label=f"{s}b")
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    baseline_line(ax)
    if pt is not None:
        peak_marker(ax, 128, pt, label=f"table {pt:.1f}×")
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup over sequential")
    ax.legend(title=f"{ALG_LABEL[alg]} @ large moduli (sieve)", fontsize=10.5, ncol=2, loc="upper left")
    savefig(fig, f"{alg}_large_moduli_speedup")

def fig_cvg(alg):
    d = CVG[alg]
    if d["ct"] is not None:  # side-by-side: CPU sieve | CPU table, each vs the GPUs
        fig, (axs, axt) = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
        eplot(axs, SIZES_ALL, d["cs"], d["cs_se"], "o-", color="#1f77b4", lw=1.9, label="AION CPU — N64 (128r)")
        eplot(axt, SIZES_ALL, d["ct"], d["ct_se"], "o-", color="#ff7f0e", lw=1.9, label="AION CPU — N64 (128r)")
        for ax in (axs, axt):
            eplot(ax, SIZES_ALL, d["l40"], d["l40_se"], "o-.", color="#8e44ad", lw=1.7, ms=5, label="IRIS 4×L40")
            eplot(ax, SIZES_ALL, d["g"], d["g_se"], "o--", color="#1a9850", lw=1.7, ms=6, label="IRIS 4×V100")
            ax.set_xscale("log", base=2); ax.set_xticks(SIZES_ALL)
            ax.get_xaxis().set_major_formatter(mt.ScalarFormatter()); ax.tick_params(axis="x", rotation=45)
            baseline_line(ax); ax.set_xlabel("modulus size (bits)")
            ax.legend(fontsize=10.5, loc="upper left")
        axs.set_ylabel("speedup over sequential")
        axs.set_title("sieve method", fontsize=14, fontweight="bold")
        axt.set_title("table method", fontsize=14, fontweight="bold")
        axs.legend(title=ALG_LABEL[alg], fontsize=10.5, loc="upper left")
        savefig(fig, f"{alg}_cpu_vs_gpu")
        return
    # single-panel (alg7 — no usable table curve at these sizes; see MIN_CLEAN)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    eplot(ax, SIZES_ALL, d["cs"], d["cs_se"], "o-", color="#1f77b4", lw=1.9, label="AION CPU — N64 sieve")
    eplot(ax, SIZES_ALL, d["l40"], d["l40_se"], "o-.", color="#8e44ad", lw=1.7, ms=5, label="IRIS 4×L40")
    eplot(ax, SIZES_ALL, d["g"], d["g_se"], "o--", color="#1a9850", lw=1.7, ms=6, label="IRIS 4×V100")
    ax.set_xscale("log", base=2); ax.set_xticks(SIZES_ALL)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter()); ax.tick_params(axis="x", rotation=45)
    baseline_line(ax)
    ax.set_xlabel("modulus size (bits)"); ax.set_ylabel("speedup over sequential")
    ax.legend(title=ALG_LABEL[alg], fontsize=10.5, loc="upper left")
    savefig(fig, f"{alg}_cpu_vs_gpu")


# ═══════════════════════════════════════════════════════════════════════════
#  Measurement variability — one standalone figure per algorithm.
#
#  Inline ±SE bars have a presentational flaw on the log speedup axes used
#  everywhere else: a SYMMETRIC error drawn on a log scale renders
#  asymmetrically, since log(1 + e/v) != -log(1 - e/v). At alg4's largest
#  spreads the lower whisker is drawn 2.2x longer than the upper, which reads
#  as a downward skew that is not in the data.
#
#  Plotting the RELATIVE standard error on its own removes the problem
#  entirely: it is a magnitude, not a range about a point, so there is no
#  direction to distort. The grid also shows something the inline bars cannot
#  — how the noise moves across BOTH axes at once (rank count and modulus).
#
#  Colour is log-scaled because the values span four orders of magnitude
#  (0.008% to 67%); a linear map would render everything except alg4's table
#  method as a uniform blank.
# ═══════════════════════════════════════════════════════════════════════════
VAR_ALGS = ("k16", "k32", "k35", "alg3", "alg4", "alg7")

def fig_variability(alg):
    import numpy as np
    from matplotlib.colors import LogNorm

    # alg7's table method is omitted here because its runs are cache-contaminated
    # (0 clean runs of 48 at 24576b), so a spread computed over them would
    # describe cache population rather than run-to-run variability. This is a gap
    # in the corpus, not a property of the algorithm -- on the paper-scale cells
    # that DO clear MIN_CLEAN it beats the sieve by 1.2-1.4x. See fig_ps.
    methods = [m for m in ("sieve", "table")
               if not (alg == "alg7" and m == "table")
               and any(cpu(alg, m, s, n, err=True) is not None
                       for s in SIZES_ALL for n in NODES)]
    if not methods:
        return

    grids = {}
    for m in methods:
        g = np.full((len(SIZES_ALL), len(NODES)), np.nan)
        for i, s in enumerate(SIZES_ALL):
            for j, n in enumerate(NODES):
                v = cpu(alg, m, s, n)
                e = cpu(alg, m, s, n, err=True)
                if v and e is not None and v > 0:
                    g[i, j] = max(e / v * 100.0, 1e-3)   # % of value; floor for LogNorm
        grids[m] = g

    fig, axes = plt.subplots(1, len(methods), figsize=(5.9 * len(methods), 4.9),
                             squeeze=False)
    axes = axes[0]
    norm = LogNorm(vmin=0.01, vmax=200.0)
    im = None
    for k, (ax, m) in enumerate(zip(axes, methods)):
        im = ax.imshow(grids[m], origin="lower", aspect="auto",
                       cmap="YlOrRd", norm=norm)
        ax.set_xticks(range(len(NODES)))
        ax.set_xticklabels([str(r) for r in RANKS])
        # Both panels share the modulus axis, so only the leftmost is labelled --
        # repeating the values on the right panel put them hard against the left
        # panel's cells at this wspace. The tick POSITIONS stay on every panel:
        # they anchor the minor-tick grid drawn below.
        ax.set_yticks(range(len(SIZES_ALL)))
        ax.set_yticklabels([f"{s}" for s in SIZES_ALL] if k == 0 else [])
        ax.set_xlabel("MPI ranks")
        ax.set_title(f"{m} method", fontsize=14, fontweight="bold")
        # label only the cells that matter: anything a reader should not trust blindly
        for i in range(len(SIZES_ALL)):
            for j in range(len(NODES)):
                val = grids[m][i, j]
                if np.isnan(val) or val < 5.0:
                    continue
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold",
                        color="white" if val > 25 else "black")
        ax.set_xticks(np.arange(-0.5, len(NODES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SIZES_ALL), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.6)
        ax.tick_params(which="minor", length=0)
    axes[0].set_ylabel("modulus size (bits)")
    # Reserve space on the right and place the colourbar in its own axes, so it
    # cannot be reflowed on top of the panels.
    fig.subplots_adjust(left=0.075, right=0.88, top=0.87, bottom=0.12, wspace=0.12)
    cax = fig.add_axes([0.905, 0.12, 0.018, 0.75])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(("95% CI half-width" if ERRMODE.startswith("ci95") else "standard error")
                 + ", % of measured speedup", fontsize=11.9)
    cb.ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda y, _: f"{y:g}%"))
    cb.ax.tick_params(labelsize=8)
    # Suptitle disabled: the LaTeX \caption already names the algorithm and states
    # what the colour means, so an embedded title duplicates it. Every other figure
    # in this set is captionless for the same reason (see the module docstring).
    # The panel titles ("sieve method" / "table method") stay -- they label the two
    # axes rather than the figure, and nothing outside the image supplies them.
    #
    # fig.suptitle(f"{ALG_LABEL.get(alg, alg)} — run-to-run variability "
    #              + ("(95% confidence interval)"
    #                 if ERRMODE.startswith("ci95") else "(standard error)"),
    #              fontsize=14.7, fontweight="bold")
    savefig(fig, f"{alg}_variability", tight=False)


if __name__ == "__main__":
    fig_k16_paperscale()
    fig_k16_large_moduli()
    fig_k16_gpu()
    fig_k16_cpu_vs_gpu()
    fig_k32_paperscale()
    fig_k32_large_moduli()
    fig_k32_cpu_vs_gpu()
    for _a in ("k35", "alg3", "alg4", "alg7"):
        fig_ps(_a)
        fig_lm(_a)
        fig_cvg(_a)
    fig_large_moduli()
    fig_gap1()
    for _a in VAR_ALGS:
        fig_variability(_a)
