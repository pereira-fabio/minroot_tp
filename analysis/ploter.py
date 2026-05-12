"""
ploter.py — Benchmark charts for MinRoot VDF cryptanalysis (k16/k32/k35).

Usage (run from project root):
    analysis/env/bin/python3 analysis/ploter.py 'results/**/*.json' [--out DIR] [--format png|pdf] [--ranks N]

Options:
    --out DIR       Output directory (default: figures/)
    --format FMT    png or pdf (default: png)
    --ranks N       Rank count for comparison plots (default: max available)

Example:
    analysis/env/bin/python3 analysis/ploter.py 'results/**/*.json' --out figures/ --format png

Figures:
    01_speedup_total.png            speedup vs challenge size (end-to-end)
    02_speedup_recon.png            speedup vs challenge size (reconstruction only)
    03_total_time_scaling.png       total_ms vs size (log-log) with powm baseline
    04_factor_recon_scaling.png     factor_ms / recon_ms (two subplots)
    05_timing_breakdown_<N>b.png    bar chart per challenge size
    06_recon_breakdown.png          local_cbrt / gather / combine vs size
    07_rank_load_balance_<N>b.png   primes found per MPI rank
    09_strong_scaling_k<K>_<M>.png  time & speedup vs MPI ranks (per algo/method)
    10_parallel_efficiency.png      scaling efficiency vs ranks
    11_algorithm_comparison_<M>.png k16 vs k32 vs k35 vs alg3 speedup at max ranks
    12_alg3_hit_stats.png           smooth vs almost-smooth classification + q₀ bit distribution
    13_effective_speedup.png        inversions-per-unit-time × success-probability vs k32 baseline
"""

import json
import sys
import os
import argparse
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

PALETTE = [
    ("#1f77b4", "o"),
    ("#ff7f0e", "s"),
    ("#2ca02c", "^"),
    ("#d62728", "D"),
    ("#9467bd", "v"),
    ("#8c564b", "P"),
    ("#e377c2", "X"),
    ("#17becf", "h"),
]

BAR_COLORS = {
    "factor":     "#1f77b4",
    "recon":      "#ff7f0e",
    "total":      "#2ca02c",
    "powm":       "#d62728",
    "local_cbrt": "#9467bd",
    "gather":     "#8c564b",
    "combine":    "#e377c2",
}

# Nominal challenge sizes — actual challenge_bits from mpz_sizeinbase may be up to
# bitbound bits smaller than the requested target (e.g. 237 instead of 256 for k16).
# All actual values are snapped to the nearest nominal before plotting so that
# k16/k32/k35 share the same x-axis positions for the same "challenge size".
NOMINAL_SIZES = [256, 512, 1024, 2048, 4096, 6144, 7000, 8192, 12288, 16384, 20480, 24576]

def snap_to_nominal(bits):
    return min(NOMINAL_SIZES, key=lambda n: abs(n - bits))

def dedup_datasets(datasets):
    """After snapping, drop duplicate (config_key, challenge_bits) pairs.
    When duplicates exist, prefer the dataset with hit_stats (richer run)."""
    seen = {}
    for d in datasets:
        key = (config_key(d), d["metadata"]["challenge_bits"])
        if key not in seen or ("hit_stats" in d and "hit_stats" not in seen[key]):
            seen[key] = d
    return list(seen.values())

# Fixed colors per algorithm for comparison plots
K_COLORS = {
    16: ("#1f77b4", "o"),
    32: ("#ff7f0e", "s"),
    35: ("#2ca02c", "^"),
}
ALG3_COLOR = ("#d62728", "D")   # red diamond — distinct from k32

def is_alg3(d):
    return d.get("_source_algo") == "alg3" or "hit_stats" in d

def algo_name(d):
    return "alg3" if is_alg3(d) else f"k{d['metadata']['k_size']}"

# Theoretical success probabilities from Biryukov et al. (CRYPTO 2024)
P_THEORY = {
    "k16":  2**(-16.0),
    "k32":  2**(-24.9),
    "k35":  2**(-21.0),
    "alg3": 2**(-18.0),
}

OUTPUT_FMT = "png"   # overwritten in main() from --format

# ── Data loading ──────────────────────────────────────────────────────────────

def load_file(path):
    with open(path) as f:
        d = json.load(f)
    # Inject stable algorithm identity from filename so alg3 files without
    # hit_stats (old runs) are not misclassified as k32.
    basename = os.path.basename(path)
    if basename.startswith("alg3_"):
        d["_source_algo"] = "alg3"
    return d

def config_key(d):
    """Unique key: (algo, method, mpi_ranks).  alg3 and k32 share k_size=32 but
    are distinguished by hit_stats or the _source_algo tag injected at load time."""
    m = d["metadata"]
    return (algo_name(d), m["method"], m["mpi_ranks"])

def config_label(d):
    m = d["metadata"]
    r, t = m["mpi_ranks"], m["threads_per_rank"]
    return f"{algo_name(d)} {m['method']} ({r}R×{t}T)"

def group_by_config(datasets):
    """dict[(k_size, method, mpi_ranks)] -> list[datasets], sorted by challenge_bits."""
    groups = defaultdict(list)
    for d in datasets:
        groups[config_key(d)].append(d)
    for k in groups:
        groups[k].sort(key=lambda d: d["metadata"]["challenge_bits"])
    return dict(groups)

def filter_by_ranks(groups, target_ranks):
    """Keep only configs where mpi_ranks == target_ranks."""
    return {k: v for k, v in groups.items() if k[2] == target_ranks}

def group_for_scaling(datasets):
    """dict[(algo, method)] -> dict[mpi_ranks] -> list[datasets sorted by bits].
    Powm method excluded — it doesn't benefit from more ranks."""
    result = defaultdict(lambda: defaultdict(list))
    for d in datasets:
        m = d["metadata"]
        if m["method"] == "powm":
            continue
        result[(algo_name(d), m["method"])][m["mpi_ranks"]].append(d)
    for km, rank_dict in result.items():
        for r in rank_dict:
            rank_dict[r].sort(key=lambda d: d["metadata"]["challenge_bits"])
    return dict(result)

def get_powm_baseline(datasets, target_ranks=None):
    """powm_baseline_ms per challenge_bits (stored in every JSON file)."""
    baseline = {}
    for d in datasets:
        bits = d["metadata"]["challenge_bits"]
        pv   = d["metadata"].get("powm_baseline_ms", 0.0)
        if pv > 0:
            if target_ranks is None or d["metadata"]["mpi_ranks"] == target_ranks:
                if bits not in baseline:
                    baseline[bits] = pv
    return baseline

def assign_styles(groups):
    styles = {}
    for i, k in enumerate(sorted(groups.keys())):
        styles[k] = PALETTE[i % len(PALETTE)]
    return styles

def pick_representative_sizes(sizes, n=4):
    """n sizes approximately equally log-spaced from the given sorted list."""
    if len(sizes) <= n:
        return list(sizes)
    log_min, log_max = np.log2(min(sizes)), np.log2(max(sizes))
    targets = [log_min + i * (log_max - log_min) / (n - 1) for i in range(n)]
    result = []
    for t in targets:
        closest = min(sizes, key=lambda s: abs(np.log2(s) - t))
        if closest not in result:
            result.append(closest)
    return sorted(result)

def all_bits_in(groups):
    return sorted(set(
        d["metadata"]["challenge_bits"]
        for ds in groups.values() for d in ds
    ))

# ── Robust statistics ─────────────────────────────────────────────────────────

def _valid_mask(d):
    attempts = d.get("warm_runs", {}).get("blinding_attempts", [])
    if not attempts:
        return None
    return [a <= 1 for a in attempts]

def robust_mean(d, key, sub="warm_runs"):
    vals = d.get(sub, {}).get(key, [])
    if not vals:
        return d.get("warm_avg", {}).get(key, 0.0)
    mask = _valid_mask(d)
    valid = [v for v, m in zip(vals, mask if mask else [True] * len(vals)) if m]
    return sum(valid) / len(valid) if valid else 0.0

def robust_speedup_total(d):
    powm  = d["metadata"].get("powm_baseline_ms", 0.0)
    total = robust_mean(d, "total_ms")
    return powm / total if total > 0 else 0.0

def robust_speedup_recon(d):
    powm  = d["metadata"].get("powm_baseline_ms", 0.0)
    recon = robust_mean(d, "recon_ms")
    return powm / recon if recon > 0 else 0.0

def robust_stddev(d, key):
    vals = d.get("warm_runs", {}).get(key, [])
    if len(vals) < 2:
        return 0.0
    mask  = _valid_mask(d)
    valid = [v for v, m in zip(vals, mask if mask else [True] * len(vals)) if m]
    if len(valid) < 2:
        return 0.0
    mu = sum(valid) / len(valid)
    return (sum((v - mu) ** 2 for v in valid) / (len(valid) - 1)) ** 0.5

# ── Utilities ─────────────────────────────────────────────────────────────────

def save(fig, path, title=""):
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def log2_fmt(v, _):
    return f"{int(v)}"

def ms_fmt(v, _):
    if v >= 100:  return f"{v:.0f}"
    if v >= 10:   return f"{v:.0f}"
    if v >= 1:    return f"{v:.1f}"
    if v >= 0.1:  return f"{v:.1f}"
    if v >= 0.01: return f"{v:.2f}"
    return f"{v:.3f}"

LOWER  = "↓ lower is better"
HIGHER = "↑ higher is better"

def hint_title(ax, title, hint, title_fontsize=12):
    """Set an axis title with a smaller italic hint line below it."""
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=16)
    ax.text(0.5, 1.0, hint, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8.5, color='#888888',
            style='italic', clip_on=False)

def apply_log_y_style(ax, fmt=None, subs=(1.0,)):
    """Log y-axis with labeled ticks at selected sub-decades and light gridlines.
    subs=(1.0,) → decades only; subs=(1.,2.,5.) → 1-2-5 progression per decade."""
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=subs))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    if fmt is not None:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.6, color="grey")
    ax.set_axisbelow(True)

def set_size_ticks(ax, sizes):
    ticks = sorted(set(int(s) for s in sizes))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(log2_fmt))
    if len(ticks) > 5:
        ax.tick_params(axis="x", rotation=45)
    ax.xaxis.set_minor_locator(ticker.NullLocator())

# ── Figure 01: Speedup (total) vs challenge size ──────────────────────────────

def plot_speedup_total(groups, styles, out_dir):
    sizes = all_bits_in(groups)
    fig, ax = plt.subplots(figsize=(9, 5))
    for ck, datasets in sorted(g for g in groups.items() if g[0][1] != "powm"):
        xs = [d["metadata"]["challenge_bits"] for d in datasets]
        ys = [robust_speedup_total(d) for d in datasets]
        col, mk = styles[ck]
        ax.plot(xs, ys, marker=mk, color=col, linewidth=1.8,
                markersize=6, label=config_label(datasets[0]))
    ax.axhline(1, color="grey", linewidth=0.8, linestyle=":", label="breakeven (1×)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    apply_log_y_style(ax)
    set_size_ticks(ax, sizes)
    ax.set_xlabel("Challenge size (bits)")
    ax.set_ylabel("Speedup over sequential powm")
    ax.legend(loc="upper left", framealpha=0.9)
    hint_title(ax, "End-to-End Speedup vs Sequential powm", HIGHER)
    save(fig, os.path.join(out_dir, f"01_speedup_total.{OUTPUT_FMT}"), "")

# ── Figure 02: Speedup (recon only) vs challenge size ────────────────────────

def plot_speedup_recon(groups, styles, out_dir):
    sizes = all_bits_in(groups)
    fig, ax = plt.subplots(figsize=(9, 5))
    for ck, datasets in sorted(g for g in groups.items() if g[0][1] != "powm"):
        xs = [d["metadata"]["challenge_bits"] for d in datasets]
        ys = [robust_speedup_recon(d) for d in datasets]
        col, mk = styles[ck]
        ax.plot(xs, ys, marker=mk, color=col, linewidth=1.8,
                markersize=6, label=config_label(datasets[0]))
    ax.axhline(1, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    apply_log_y_style(ax)
    set_size_ticks(ax, sizes)
    ax.set_xlabel("Challenge size (bits)")
    ax.set_ylabel("Speedup (reconstruction only)")
    ax.legend(loc="upper left", framealpha=0.9)
    hint_title(ax, "Reconstruction-Only Speedup vs Sequential powm", HIGHER)
    save(fig, os.path.join(out_dir, f"02_speedup_recon.{OUTPUT_FMT}"), "")

# ── Figure 03: Total time (log-log) with powm baseline ───────────────────────

def plot_total_time_scaling(groups, styles, powm_baseline, out_dir):
    sizes = all_bits_in(groups)
    fig, ax = plt.subplots(figsize=(9, 5))
    if powm_baseline:
        ss = sorted(powm_baseline)
        ax.plot(ss, [powm_baseline[s] for s in ss],
                color=BAR_COLORS["powm"], linewidth=2, linestyle="--",
                marker="x", markersize=6, label="powm baseline (sequential)", zorder=5)
    for ck, datasets in sorted(g for g in groups.items() if g[0][1] != "powm"):
        xs = [d["metadata"]["challenge_bits"] for d in datasets]
        ys = [robust_mean(d, "total_ms") for d in datasets]
        col, mk = styles[ck]
        ax.plot(xs, ys, marker=mk, color=col, linewidth=1.8,
                markersize=6, label=config_label(datasets[0]))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    apply_log_y_style(ax, fmt=ms_fmt)
    set_size_ticks(ax, sizes)
    ax.set_xlabel("Challenge size (bits)")
    ax.set_ylabel("Total time (ms)")
    ax.legend(loc="upper left", framealpha=0.9)
    hint_title(ax, "Total Execution Time vs Challenge Size", LOWER)
    save(fig, os.path.join(out_dir, f"03_total_time_scaling.{OUTPUT_FMT}"), "")

# ── Figure 04: Factor + recon scaling ────────────────────────────────────────

def plot_factor_recon_scaling(groups, styles, out_dir):
    sizes = all_bits_in(groups)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ck, datasets in sorted(g for g in groups.items() if g[0][1] != "powm"):
        xs      = [d["metadata"]["challenge_bits"] for d in datasets]
        factors = [robust_mean(d, "factor_ms") for d in datasets]
        recons  = [robust_mean(d, "recon_ms")  for d in datasets]
        col, mk = styles[ck]
        lbl = config_label(datasets[0])
        ax1.plot(xs, factors, marker=mk, color=col, linewidth=1.8, markersize=6, label=lbl)
        ax2.plot(xs, recons,  marker=mk, color=col, linewidth=1.8, markersize=6, label=lbl)
    for ax, ttl, ylbl in [
        (ax1, "Factoring Time",      "Factoring time (ms)"),
        (ax2, "Reconstruction Time", "Reconstruction time (ms)"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        apply_log_y_style(ax, fmt=ms_fmt)
        set_size_ticks(ax, sizes)
        ax.set_xlabel("Challenge size (bits)")
        ax.set_ylabel(ylbl)
        hint_title(ax, ttl, LOWER)
        ax.legend(fontsize=8, framealpha=0.9)
    save(fig, os.path.join(out_dir, f"04_factor_recon_scaling.{OUTPUT_FMT}"),
         "Factoring and Reconstruction Scaling")

# ── Figure 05: Timing breakdown bar chart per size ───────────────────────────

def plot_timing_breakdown_per_size(groups, styles, powm_baseline, sizes, out_dir):
    attack = {k: v for k, v in groups.items() if k[1] != "powm"}
    for bits in sizes:
        relevant = [(ck, d) for ck, ds in sorted(attack.items())
                    for d in ds if d["metadata"]["challenge_bits"] == bits]
        if not relevant:
            continue
        n = len(relevant)
        x = np.arange(n)
        w = 0.22
        fig, ax = plt.subplots(figsize=(max(7, n * 1.4), 5))
        factors = [robust_mean(d, "factor_ms") for _, d in relevant]
        recons  = [robust_mean(d, "recon_ms")  for _, d in relevant]
        totals  = [robust_mean(d, "total_ms")  for _, d in relevant]
        ferrs   = [robust_stddev(d, "factor_ms") for _, d in relevant]
        rerrs   = [robust_stddev(d, "recon_ms")  for _, d in relevant]
        terrs   = [robust_stddev(d, "total_ms")  for _, d in relevant]
        labels  = [config_label(d) for _, d in relevant]
        b1 = ax.bar(x - w, factors, w, label="Factoring",      color=BAR_COLORS["factor"],
                    yerr=ferrs, capsize=4, error_kw={"elinewidth": 1.2})
        b2 = ax.bar(x,     recons,  w, label="Reconstruction", color=BAR_COLORS["recon"],
                    yerr=rerrs, capsize=4, error_kw={"elinewidth": 1.2})
        b3 = ax.bar(x + w, totals,  w, label="Total",          color=BAR_COLORS["total"],
                    yerr=terrs, capsize=4, error_kw={"elinewidth": 1.2})
        pv = powm_baseline.get(bits)
        if pv:
            ax.hlines(pv, -0.5, n - 0.5, colors=BAR_COLORS["powm"],
                      linewidths=2, linestyles="--", zorder=5)
            pp = mpatches.Patch(facecolor="none", edgecolor=BAR_COLORS["powm"],
                                linestyle="--", linewidth=2,
                                label=f"powm baseline ({pv:.1f} ms)")
            ax.legend(handles=[b1, b2, b3, pp], loc="upper left")
        else:
            ax.legend(loc="upper left")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yscale("log")
        apply_log_y_style(ax, fmt=ms_fmt)
        ax.set_ylabel("Time (ms)")
        hint_title(ax, f"Timing Breakdown — {bits}-bit challenge", LOWER)
        save(fig, os.path.join(out_dir, f"05_timing_breakdown_{bits}b.{OUTPUT_FMT}"), "")

# ── Figure 06: Reconstruction breakdown vs size ───────────────────────────────

def plot_recon_breakdown_scaling(groups, styles, out_dir):
    mpi_groups = {k: v for k, v in groups.items()
                  if k[1] != "powm" and
                  any(d["metadata"]["mpi_ranks"] > 1 for d in v)}
    if not mpi_groups:
        print("  [SKIP] recon breakdown: no MPI datasets")
        return
    sizes = all_bits_in(mpi_groups)
    fig, ax = plt.subplots(figsize=(10, 5))
    for ck, datasets in sorted(mpi_groups.items()):
        xs = [d["metadata"]["challenge_bits"] for d in datasets]
        lc = [robust_mean(d, "local_cbrt_ms") for d in datasets]
        ga = [robust_mean(d, "gather_ms")     for d in datasets]
        co = [robust_mean(d, "combine_ms")    for d in datasets]
        col, mk = styles[ck]
        lbl = config_label(datasets[0])
        ax.plot(xs, lc, marker=mk, color=col, linewidth=1.8, markersize=5,
                label=f"{lbl} — local cbrt")
        ax.plot(xs, ga, marker=mk, color=col, linewidth=1,   markersize=4,
                linestyle="--", alpha=0.7, label=f"{lbl} — gather")
        ax.plot(xs, co, marker=mk, color=col, linewidth=1,   markersize=4,
                linestyle=":",  alpha=0.7, label=f"{lbl} — combine")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    apply_log_y_style(ax, fmt=ms_fmt)
    set_size_ticks(ax, sizes)
    ax.set_xlabel("Challenge size (bits)")
    ax.set_ylabel("Time (ms)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    hint_title(ax, "Reconstruction Phase Breakdown vs Size", LOWER)
    save(fig, os.path.join(out_dir, f"06_recon_breakdown.{OUTPUT_FMT}"), "")

# ── Figure 07: Per-rank prime load balance ────────────────────────────────────

def plot_rank_load_balance(datasets, sizes, out_dir):
    for bits in sizes:
        mpi_at = [d for d in datasets
                  if d["metadata"]["challenge_bits"] == bits
                  and d["metadata"]["mpi_ranks"] > 1
                  and d.get("prime_stats", {}).get("found_per_rank")]
        if not mpi_at:
            continue
        fig, axes = plt.subplots(1, len(mpi_at),
                                 figsize=(5 * len(mpi_at), 4), squeeze=False)
        for i, d in enumerate(mpi_at):
            found = d["prime_stats"]["found_per_rank"]
            mean  = np.mean(found)
            ax    = axes[0][i]
            ax.bar(range(len(found)), found, color=PALETTE[i % len(PALETTE)][0], alpha=0.8)
            ax.axhline(mean, color="black", linewidth=1.2, linestyle="--",
                       label=f"mean = {mean:.1f}")
            ax.set_title(config_label(d))
            ax.set_xlabel("MPI rank")
            ax.set_ylabel("Primes found")
            ax.legend(fontsize=9)
        save(fig, os.path.join(out_dir, f"07_rank_load_balance_{bits}b.{OUTPUT_FMT}"),
             f"Primes Found per MPI Rank — {bits}-bit challenge")

# ── Figure 09: Strong scaling (time & speedup vs MPI ranks) ──────────────────

def plot_strong_scaling(datasets, out_dir):
    """One figure per (k_size, method): left = total_ms vs ranks, right = speedup vs ranks.
    Four representative challenge sizes shown as separate lines."""
    scaling = group_for_scaling(datasets)

    for (algo, method), rank_dict in sorted(scaling.items()):
        all_ranks = sorted(rank_dict.keys())
        if len(all_ranks) < 2:
            continue

        # Sizes present at every rank count
        common = sorted(
            set.intersection(*[set(d["metadata"]["challenge_bits"] for d in rank_dict[r])
                                for r in all_ranks])
        )
        if not common:
            common = sorted(set(
                d["metadata"]["challenge_bits"]
                for r in all_ranks for d in rank_dict[r]
            ))

        # Power-of-2 nominal sizes only (256, 512, 1024, … 16384)
        pow2 = [s for s in common if s > 0 and (s & (s - 1)) == 0]
        rep_sizes = pow2 if pow2 else common
        r_ref = all_ranks[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        for i, bits in enumerate(rep_sizes):
            col, mk = PALETTE[i % len(PALETTE)]
            pts = []
            for r in all_ranks:
                match = [d for d in rank_dict[r] if d["metadata"]["challenge_bits"] == bits]
                if match:
                    pts.append((r, robust_mean(match[0], "total_ms")))
            if len(pts) < 2:
                continue
            ranks   = [p[0] for p in pts]
            total   = [p[1] for p in pts]
            t_ref   = pts[0][1]
            speedup = [t_ref / t if t > 0 else 0 for t in total]

            ax1.plot(ranks, total,   marker=mk, color=col, linewidth=1.8,
                     markersize=6, label=f"{bits} bits")
            ax2.plot(ranks, speedup, marker=mk, color=col, linewidth=1.8,
                     markersize=6, label=f"{bits} bits")

        ideal = [r / r_ref for r in all_ranks]
        ax2.plot(all_ranks, ideal, "k--", linewidth=1.2, alpha=0.5, label="ideal")
        ax2.axhline(1, color="grey", linewidth=0.6, linestyle=":")

        for ax in (ax1, ax2):
            ax.set_xscale("log", base=2)
            ax.set_xticks(all_ranks)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(log2_fmt))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xlabel("MPI ranks")

        ax1.set_yscale("log")
        apply_log_y_style(ax1, fmt=ms_fmt, subs=(1., 2., 5.))
        ax1.set_ylabel("Total time (ms)  [log scale]")
        hint_title(ax1, f"{algo} {method}: Time vs Ranks", LOWER)
        ax1.legend(title="Challenge", fontsize=8)

        ax2.grid(axis="y", which="major", alpha=0.25, linewidth=0.6, color="grey")
        ax2.set_axisbelow(True)
        ax2.set_ylabel(f"Speedup over {r_ref} ranks")
        hint_title(ax2, f"{algo} {method}: Strong Scaling", HIGHER)
        ax2.legend(title="Challenge", fontsize=8)

        save(fig,
             os.path.join(out_dir, f"09_strong_scaling_{algo}_{method}.{OUTPUT_FMT}"),
             "")

# ── Figure 10: Parallel efficiency vs ranks ────────────────────────────────────

def plot_parallel_efficiency(datasets, out_dir):
    """efficiency(N) = T(R_min) * R_min / (N * T(N)).  One curve per (k_size, method)
    at the largest common challenge size (where scaling differences are most visible)."""
    scaling = group_for_scaling(datasets)
    all_rank_vals = sorted(set(d["metadata"]["mpi_ranks"] for d in datasets))

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    for idx, ((algo, method), rank_dict) in enumerate(sorted(scaling.items())):
        all_ranks = sorted(rank_dict.keys())
        if len(all_ranks) < 2:
            continue
        common = sorted(
            set.intersection(*[set(d["metadata"]["challenge_bits"] for d in rank_dict[r])
                                for r in all_ranks])
        )
        if not common:
            continue
        bits = common[-1]   # largest common challenge size

        pts = []
        for r in all_ranks:
            match = [d for d in rank_dict[r] if d["metadata"]["challenge_bits"] == bits]
            if match:
                pts.append((r, robust_mean(match[0], "total_ms")))
        if len(pts) < 2:
            continue

        t_ref, r_ref = pts[0][1], pts[0][0]
        efficiency = [(t_ref * r_ref) / (r * t) if t > 0 else 0 for r, t in pts]
        ranks = [p[0] for p in pts]

        col, mk = PALETTE[idx % len(PALETTE)]
        ax.plot(ranks, efficiency, marker=mk, color=col, linewidth=1.8,
                markersize=6, label=f"{algo} {method} ({bits} bits)")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", label="ideal (100 %)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_rank_vals)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(log2_fmt))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Parallel efficiency  T₀·R₀ / (R·Tᴿ)")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    hint_title(ax, "Parallel Efficiency vs MPI Ranks", HIGHER)
    save(fig, os.path.join(out_dir, f"10_parallel_efficiency.{OUTPUT_FMT}"), "")

# ── Figure 11: k16 vs k32 vs k35 algorithm comparison ─────────────────────────

def plot_algorithm_comparison(datasets, out_dir):
    """Two subplots per method: total time (log-log) and speedup vs challenge size.
    Uses max available rank count so every algorithm is at peak performance."""
    max_ranks = max(d["metadata"]["mpi_ranks"] for d in datasets)
    methods   = sorted(set(d["metadata"]["method"] for d in datasets
                            if d["metadata"]["method"] != "powm"))

    for method in methods:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # powm baseline (same regardless of k-size for equal bittargets)
        powm_pts = {}
        for d in datasets:
            if d["metadata"]["mpi_ranks"] == max_ranks:
                bits = d["metadata"]["challenge_bits"]
                pv   = d["metadata"].get("powm_baseline_ms", 0.0)
                if pv > 0 and bits not in powm_pts:
                    powm_pts[bits] = pv
        if powm_pts:
            ss = sorted(powm_pts)
            ax1.plot(ss, [powm_pts[s] for s in ss],
                     color=BAR_COLORS["powm"], linewidth=2, linestyle="--",
                     marker="x", markersize=5, label="powm (sequential)", zorder=5)

        all_bits = []
        for k_size in [16, 32, 35]:
            subset = sorted(
                [d for d in datasets
                 if d["metadata"]["k_size"]    == k_size
                 and not is_alg3(d)
                 and d["metadata"]["method"]   == method
                 and d["metadata"]["mpi_ranks"] == max_ranks],
                key=lambda d: d["metadata"]["challenge_bits"],
            )
            if not subset:
                continue
            col, mk = K_COLORS[k_size]
            bits    = [d["metadata"]["challenge_bits"] for d in subset]
            total   = [robust_mean(d, "total_ms") for d in subset]
            speedup = [robust_speedup_total(d) for d in subset]
            all_bits.extend(bits)
            ax1.plot(bits, total,   marker=mk, color=col, linewidth=1.8,
                     markersize=6, label=f"k{k_size}")
            ax2.plot(bits, speedup, marker=mk, color=col, linewidth=1.8,
                     markersize=6, label=f"k{k_size}")

        # alg3 curve (same method, max ranks)
        alg3_sub = sorted(
            [d for d in datasets if is_alg3(d)
             and d["metadata"]["method"]   == method
             and d["metadata"]["mpi_ranks"] == max_ranks],
            key=lambda d: d["metadata"]["challenge_bits"],
        )
        if alg3_sub:
            col, mk = ALG3_COLOR
            bits    = [d["metadata"]["challenge_bits"] for d in alg3_sub]
            total   = [robust_mean(d, "total_ms") for d in alg3_sub]
            speedup = [robust_speedup_total(d) for d in alg3_sub]
            all_bits.extend(bits)
            ax1.plot(bits, total,   marker=mk, color=col, linewidth=1.8,
                     markersize=6, label="alg3")
            ax2.plot(bits, speedup, marker=mk, color=col, linewidth=1.8,
                     markersize=6, label="alg3")

        ax2.axhline(1, color="grey", linewidth=0.8, linestyle=":", label="breakeven")

        all_bits = sorted(set(all_bits))
        for ax in (ax1, ax2):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            set_size_ticks(ax, all_bits)
            ax.set_xlabel("Challenge size (bits)")
            ax.legend(framealpha=0.9)
        apply_log_y_style(ax1, fmt=ms_fmt)
        apply_log_y_style(ax2)
        ax1.set_ylabel("Total time (ms)")
        ax2.set_ylabel("Speedup over sequential powm")
        hint_title(ax1, f"k16 vs k32 vs k35 vs alg3 — {method}  ({max_ranks} ranks)", LOWER)
        hint_title(ax2, f"Speedup vs Sequential powm — {method}  ({max_ranks} ranks)", HIGHER)

        save(fig,
             os.path.join(out_dir, f"11_algorithm_comparison_{method}.{OUTPUT_FMT}"),
             "")

# ── Figure 12: alg3 hit statistics ────────────────────────────────────────────

def plot_hit_stats(datasets, out_dir):
    """Stacked bar (smooth vs almost-smooth) and cofactor bit-size distribution
    for alg3 results at the maximum rank count."""
    max_ranks = max(d["metadata"]["mpi_ranks"] for d in datasets)
    alg3_data = sorted(
        [d for d in datasets if is_alg3(d) and "hit_stats" in d
         and d["metadata"]["method"] == "sieve"
         and d["metadata"]["mpi_ranks"] == max_ranks],
        key=lambda d: d["metadata"]["challenge_bits"],
    )
    if not alg3_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sizes  = [d["metadata"]["challenge_bits"] for d in alg3_data]
    x      = np.arange(len(sizes))
    labels = [str(b) for b in sizes]

    n_smooth  = [d["hit_stats"]["smooth_count"]        for d in alg3_data]
    n_almost  = [d["hit_stats"]["almost_smooth_count"] for d in alg3_data]

    ax1.bar(x, n_almost, label="almost-smooth (q₀ hit)",
            color=BAR_COLORS["recon"])
    ax1.bar(x, n_smooth, bottom=n_almost, label="B-smooth (all-small primes)",
            color=BAR_COLORS["factor"])
    for xi, ns, na in zip(x, n_smooth, n_almost):
        total = ns + na
        ax1.text(xi, total + 0.1, str(total), ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_xlabel("Challenge size (bits)")
    ax1.set_ylabel("Run count")
    ax1.legend(framealpha=0.9)
    hint_title(ax1, f"alg3 Hit Classification  ({max_ranks} ranks)", HIGHER)

    # cofactor bit-size distribution (right subplot)
    for xi, d in zip(x, alg3_data):
        cbits = d["hit_stats"].get("cofactor_bits", [])
        if not cbits:
            continue
        cbits_np = np.array(cbits, dtype=float)
        # filter out cofactor_bits == 1 (B-smooth runs contribute no q₀)
        cbits_np = cbits_np[cbits_np > 1]
        if len(cbits_np) == 0:
            continue
        ax2.errorbar(xi, cbits_np.mean(),
                     yerr=[[cbits_np.mean() - cbits_np.min()],
                            [cbits_np.max() - cbits_np.mean()]],
                     fmt="o", color=ALG3_COLOR[0], capsize=4)

    ax2.axhline(33, color="grey", linewidth=0.8, linestyle="--", label="min q₀ (33 bits)")
    ax2.axhline(64, color="grey", linewidth=0.8, linestyle=":",  label="max q₀ (64 bits)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_xlabel("Challenge size (bits)")
    ax2.set_ylabel("q₀ bit size  (mean ± range)")
    ax2.set_ylim(0, 80)
    ax2.legend(framealpha=0.9)
    hint_title(ax2, "Medium Prime q₀ Bit-Size Distribution", HIGHER)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, f"12_alg3_hit_stats.{OUTPUT_FMT}"), "")

# ── Figure 13: effective throughput comparison ─────────────────────────────────

def plot_effective_speedup(datasets, out_dir):
    """Inversions-per-powm-unit × p_success, normalised to k32 = 1.0.
    Shows the comparative advantage of each algorithm accounting for success
    probability from Biryukov et al. (CRYPTO 2024)."""
    max_ranks = max(d["metadata"]["mpi_ranks"] for d in datasets)
    method    = "sieve"

    # Build per-algo curves: eff_rate = speedup_total * P_THEORY[algo]
    curves = {}
    for k_size in [16, 32, 35]:
        subset = sorted(
            [d for d in datasets
             if d["metadata"]["k_size"]    == k_size
             and not is_alg3(d)
             and d["metadata"]["method"]   == method
             and d["metadata"]["mpi_ranks"] == max_ranks],
            key=lambda d: d["metadata"]["challenge_bits"],
        )
        if not subset:
            continue
        p = P_THEORY.get(f"k{k_size}", 0)
        curves[f"k{k_size}"] = {
            "bits":     [d["metadata"]["challenge_bits"] for d in subset],
            "eff_rate": [robust_speedup_total(d) * p for d in subset],
            "color":    K_COLORS[k_size][0],
            "marker":   K_COLORS[k_size][1],
        }

    alg3_sub = sorted(
        [d for d in datasets if is_alg3(d)
         and d["metadata"]["method"]   == method
         and d["metadata"]["mpi_ranks"] == max_ranks],
        key=lambda d: d["metadata"]["challenge_bits"],
    )
    if alg3_sub:
        p = P_THEORY["alg3"]
        curves["alg3"] = {
            "bits":     [d["metadata"]["challenge_bits"] for d in alg3_sub],
            "eff_rate": [robust_speedup_total(d) * p for d in alg3_sub],
            "color":    ALG3_COLOR[0],
            "marker":   ALG3_COLOR[1],
        }

    if "k32" not in curves:
        return

    # Normalise each curve against k32 effective rate at shared sizes
    k32_lookup = dict(zip(curves["k32"]["bits"], curves["k32"]["eff_rate"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    all_bits = []
    for label, c in curves.items():
        norm = []
        for b, r in zip(c["bits"], c["eff_rate"]):
            if b in k32_lookup and k32_lookup[b] > 0:
                norm.append((b, r / k32_lookup[b]))
        if not norm:
            continue
        bx, ny = zip(*norm)
        all_bits.extend(bx)
        ax.plot(bx, ny, marker=c["marker"], color=c["color"],
                linewidth=1.8, markersize=6, label=label)

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", label="k32 baseline")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    set_size_ticks(ax, sorted(set(all_bits)))
    apply_log_y_style(ax)
    ax.set_xlabel("Challenge size (bits)")
    ax.set_ylabel("Effective inversions / k32 (= speedup_total × p_success, normalised)")
    ax.legend(framealpha=0.9)
    hint_title(ax,
        f"Effective Throughput vs k32 Baseline  ({max_ranks} ranks, sieve)",
        HIGHER)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f"13_effective_speedup.{OUTPUT_FMT}"), "")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_FMT

    parser = argparse.ArgumentParser(
        description="Plot MinRoot VDF benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("json_files", nargs="+",
                        help="JSON result files (glob patterns accepted)")
    parser.add_argument("--out",    default="figures",
                        help="Output directory (default: figures/)")
    parser.add_argument("--format", default="png", choices=["png", "pdf"],
                        dest="fmt", help="Output format (default: png)")
    parser.add_argument("--ranks",  type=int, default=None,
                        help="Rank count for comparison plots (default: max available)")
    args = parser.parse_args()
    OUTPUT_FMT = args.fmt

    paths = []
    for pattern in args.json_files:
        expanded = glob.glob(pattern, recursive=True)
        paths.extend(expanded if expanded else [pattern])
    if not paths:
        print("ERROR: no JSON files found"); sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {len(paths)} JSON file(s)...")
    datasets = []
    for p in sorted(paths):
        try:
            datasets.append(load_file(p))
        except Exception as e:
            print(f"  WARNING: skipping {p}: {e}")
    if not datasets:
        print("ERROR: no valid JSON files loaded"); sys.exit(1)

    # Snap each dataset's challenge_bits to the nearest nominal size so that
    # k16/k32/k35 share the same x-axis position for the same challenge target.
    for d in datasets:
        d["metadata"]["challenge_bits"] = snap_to_nominal(d["metadata"]["challenge_bits"])
    datasets = dedup_datasets(datasets)

    all_sizes    = sorted(set(d["metadata"]["challenge_bits"] for d in datasets))
    all_rank_vals = sorted(set(d["metadata"]["mpi_ranks"]    for d in datasets))
    max_ranks    = max(all_rank_vals)
    target_ranks = args.ranks if args.ranks else max_ranks

    print(f"  {len(datasets)} datasets")
    print(f"  Challenge sizes:  {all_sizes}")
    print(f"  MPI rank counts:  {all_rank_vals}")
    print(f"  Comparison plots: rank count = {target_ranks}")
    print()

    all_groups    = group_by_config(datasets)
    cmp_groups    = filter_by_ranks(all_groups, target_ranks)
    cmp_styles    = assign_styles(cmp_groups)
    powm_baseline = get_powm_baseline(datasets, target_ranks)

    print(f"Generating figures → {args.out}/\n")

    # ── Comparison plots at fixed rank count ──
    plot_speedup_total(cmp_groups, cmp_styles, args.out)
    plot_speedup_recon(cmp_groups, cmp_styles, args.out)
    plot_total_time_scaling(cmp_groups, cmp_styles, powm_baseline, args.out)
    plot_factor_recon_scaling(cmp_groups, cmp_styles, args.out)
    plot_recon_breakdown_scaling(cmp_groups, cmp_styles, args.out)
    plot_timing_breakdown_per_size(cmp_groups, cmp_styles, powm_baseline,
                                   all_sizes, args.out)
    plot_rank_load_balance(datasets, all_sizes, args.out)

    # ── Scaling plots (vary rank count) ──
    plot_strong_scaling(datasets, args.out)
    plot_parallel_efficiency(datasets, args.out)

    # ── Cross-algorithm comparison ──
    plot_algorithm_comparison(datasets, args.out)
    plot_hit_stats(datasets, args.out)
    plot_effective_speedup(datasets, args.out)

    n_out = len([f for f in os.listdir(args.out)
                 if f.endswith(f".{OUTPUT_FMT}")])
    print(f"\nDone. {n_out} {OUTPUT_FMT.upper()} file(s) in {args.out}/")

if __name__ == "__main__":
    main()
