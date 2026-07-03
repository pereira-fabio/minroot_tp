"""
plot_alg4_comparison.py — Head-to-head comparison of alg4 vs alg4_multi.

Reads only results/alg4/**/*.json, tags each file as "alg4" or "alg4_multi"
by basename prefix, then produces one figure per available rank count showing:
  - Speedup over sequential powm  (sieve only)
  - Total time (ms)               (sieve only)
  - Factoring time (ms)           (sieve only)
  - Reconstruction time (ms)      (sieve only)

Usage (run from project root):
    analysis/env/bin/python3 analysis/plot_alg4_comparison.py \
        --out figures/alg4_comparison/ --format png
"""

import json
import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

VARIANT_STYLE = {
    "alg4":       ("#1f77b4", "o", "-",  "alg4 (single)"),
    "alg4_multi": ("#d62728", "D", "--", "alg4_multi"),
}

NOMINAL_SIZES = [256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576]

def snap(bits):
    return min(NOMINAL_SIZES, key=lambda n: abs(n - bits))

def _valid_mask(d):
    attempts = d.get("warm_runs", {}).get("blinding_attempts", [])
    if not attempts:
        return None
    return [a <= 1 for a in attempts]

def robust_mean(d, key):
    vals = d.get("warm_runs", {}).get(key, [])
    if not vals:
        return d.get("warm_avg", {}).get(key, 0.0)
    mask = _valid_mask(d)
    valid = [v for v, m in zip(vals, mask if mask else [True] * len(vals)) if m]
    return sum(valid) / len(valid) if valid else 0.0

def robust_stddev(d, key):
    vals = d.get("warm_runs", {}).get(key, [])
    if len(vals) < 2:
        return 0.0
    mask = _valid_mask(d)
    valid = [v for v, m in zip(vals, mask if mask else [True] * len(vals)) if m]
    if len(valid) < 2:
        return 0.0
    mu = sum(valid) / len(valid)
    return (sum((v - mu) ** 2 for v in valid) / (len(valid) - 1)) ** 0.5

def speedup(d):
    powm  = d["metadata"].get("powm_baseline_ms", 0.0)
    total = robust_mean(d, "total_ms")
    return powm / total if total > 0 else None

def load_alg4_files(root="results/alg4"):
    datasets = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  WARNING: {path}: {e}")
            continue
        bname = os.path.basename(path)
        if bname.startswith("alg4_multi"):
            d["_variant"] = "alg4_multi"
        elif bname.startswith("alg4_"):
            d["_variant"] = "alg4"
        else:
            continue
        d["metadata"]["challenge_bits"] = snap(d["metadata"]["challenge_bits"])
        datasets.append(d)
    return datasets

def ms_fmt(v, _):
    if v >= 10:   return f"{v:.0f}"
    if v >= 1:    return f"{v:.1f}"
    if v >= 0.1:  return f"{v:.1f}"
    return f"{v:.2f}"

def apply_log_y(ax, fmt=None):
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.,)))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    if fmt:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.6, color="grey")
    ax.set_axisbelow(True)

def set_xticks(ax, sizes):
    ticks = sorted(set(int(s) for s in sizes))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
    if len(ticks) > 5:
        ax.tick_params(axis="x", rotation=45)
    ax.xaxis.set_minor_locator(ticker.NullLocator())

def hint(ax, title, sub):
    ax.set_title(title, fontsize=12, pad=16)
    ax.text(0.5, 1.0, sub, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, color="#888888",
            style="italic", clip_on=False)

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_comparison_for_ranks(datasets, ranks, out_dir, fmt):
    sieve = [d for d in datasets
             if d["metadata"]["method"] == "sieve"
             and d["metadata"]["mpi_ranks"] == ranks
             and d["warm_avg"]["total_ms"] > 0]

    if not sieve:
        print(f"  [SKIP] no valid sieve data at {ranks} ranks")
        return

    by_variant = defaultdict(list)
    for d in sieve:
        by_variant[d["_variant"]].append(d)
    for v in by_variant:
        by_variant[v].sort(key=lambda d: d["metadata"]["challenge_bits"])

    if len(by_variant) < 2:
        print(f"  [SKIP] only one variant present at {ranks} ranks")
        return

    all_sizes = sorted(set(d["metadata"]["challenge_bits"] for d in sieve))
    powm_baseline = {}
    for d in sieve:
        b = d["metadata"]["challenge_bits"]
        pv = d["metadata"].get("powm_baseline_ms", 0.0)
        if pv > 0 and b not in powm_baseline:
            powm_baseline[b] = pv

    metrics = [
        ("speedup",   "Speedup over sequential powm",    "↑ higher is better", None),
        ("total_ms",  "Total time (ms)",                 "↓ lower is better",  ms_fmt),
        ("factor_ms", "Factoring time (ms)",             "↓ lower is better",  ms_fmt),
        ("recon_ms",  "Reconstruction time (ms)",        "↓ lower is better",  ms_fmt),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric, title, direction, yfmt) in zip(axes, metrics):
        for variant, ds in sorted(by_variant.items()):
            col, mk, ls, label = VARIANT_STYLE[variant]
            xs, ys, errs = [], [], []
            for d in ds:
                b = d["metadata"]["challenge_bits"]
                if metric == "speedup":
                    y = speedup(d)
                else:
                    y = robust_mean(d, metric)
                    err = robust_stddev(d, metric)
                if y is None or y == 0:
                    continue
                xs.append(b)
                ys.append(y)
                if metric != "speedup":
                    errs.append(err)

            if not xs:
                continue

            if metric != "speedup" and any(e > 0 for e in errs):
                ax.errorbar(xs, ys, yerr=errs, fmt=mk, color=col,
                            linestyle=ls, linewidth=1.8, markersize=6,
                            capsize=3, elinewidth=1, label=label)
            else:
                ax.plot(xs, ys, marker=mk, color=col, linestyle=ls,
                        linewidth=1.8, markersize=6, label=label)

        if metric == "speedup":
            ax.axhline(1, color="grey", linewidth=0.8, linestyle=":",
                       label="breakeven (1×)")
        elif powm_baseline:
            ss = sorted(powm_baseline)
            ax.plot(ss, [powm_baseline[s] for s in ss],
                    color="#888888", linewidth=1.5, linestyle=":", marker="x",
                    markersize=5, label="powm baseline", zorder=2)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        apply_log_y(ax, fmt=yfmt)
        set_xticks(ax, all_sizes)
        ax.set_xlabel("Challenge size (bits)")
        ax.legend(framealpha=0.9)
        hint(ax, f"{title}  ({ranks} MPI ranks)", direction)

    fig.suptitle(f"alg4 vs alg4_multi — sieve, {ranks} MPI ranks",
                 fontsize=14, y=1.01)
    fname = os.path.join(out_dir, f"alg4_vs_multi_{ranks}ranks.{fmt}")
    save(fig, fname)

PALETTE = [
    ("#1f77b4", "o"), ("#ff7f0e", "s"), ("#2ca02c", "^"),
    ("#d62728", "D"), ("#9467bd", "v"), ("#8c564b", "P"),
    ("#e377c2", "X"), ("#17becf", "h"),
]

def pick_representative_sizes(sizes, n=5):
    if len(sizes) <= n:
        return list(sizes)
    lmin, lmax = np.log2(min(sizes)), np.log2(max(sizes))
    targets = [lmin + i * (lmax - lmin) / (n - 1) for i in range(n)]
    result = []
    for t in targets:
        closest = min(sizes, key=lambda s: abs(np.log2(s) - t))
        if closest not in result:
            result.append(closest)
    return sorted(result)


def plot_scaling(datasets, out_dir, fmt):
    """Strong scaling: time & speedup vs MPI ranks for both variants.

    Left column: total_ms vs ranks per challenge size (one subplot per variant).
    Right column: speedup over min-rank baseline, both variants overlaid.
    Bottom row: parallel efficiency per challenge size, both variants.
    """
    sieve = [d for d in datasets
             if d["metadata"]["method"] == "sieve"
             and d["warm_avg"]["total_ms"] > 0]
    if not sieve:
        print("  [SKIP] no valid sieve data for scaling plot")
        return

    all_ranks = sorted(set(d["metadata"]["mpi_ranks"] for d in sieve))
    if len(all_ranks) < 2:
        print("  [SKIP] need at least 2 rank counts for scaling")
        return

    r_ref = all_ranks[0]

    # Build lookup: variant -> challenge_bits -> ranks -> robust_mean(total_ms)
    lookup = defaultdict(lambda: defaultdict(dict))
    for d in sieve:
        v = d["_variant"]
        b = d["metadata"]["challenge_bits"]
        r = d["metadata"]["mpi_ranks"]
        lookup[v][b][r] = robust_mean(d, "total_ms")

    variants = sorted(lookup.keys())
    all_sizes = sorted(set(b for v in lookup.values() for b in v))
    rep_sizes = pick_representative_sizes(all_sizes)

    # ── Figure 1: time vs ranks (one subplot per variant) + speedup overlay ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_left, ax_mid, ax_right = axes

    for vi, variant in enumerate(variants):
        ax = ax_left if vi == 0 else ax_mid
        col_base, _, ls_base, label_base = VARIANT_STYLE[variant]

        for i, bits in enumerate(rep_sizes):
            col, mk = PALETTE[i % len(PALETTE)]
            pts = [(r, lookup[variant][bits][r])
                   for r in all_ranks if r in lookup[variant].get(bits, {})]
            if len(pts) < 2:
                continue
            rx, ty = zip(*pts)
            ax.plot(rx, ty, marker=mk, color=col, linewidth=1.8,
                    markersize=6, label=f"{bits} bits")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        apply_log_y(ax, fmt=ms_fmt)
        ax.set_xticks(all_ranks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Total time (ms)")
        ax.legend(title="Challenge", fontsize=8, framealpha=0.9)
        hint(ax, f"{label_base}: Time vs Ranks", "↓ lower is better")

    # Speedup over r_ref, both variants on same axes
    for vi, variant in enumerate(variants):
        col_base, mk_base, ls_base, label_base = VARIANT_STYLE[variant]
        for i, bits in enumerate(rep_sizes):
            col, mk = PALETTE[i % len(PALETTE)]
            pts = [(r, lookup[variant][bits][r])
                   for r in all_ranks if r in lookup[variant].get(bits, {})]
            if len(pts) < 2:
                continue
            t_ref = lookup[variant][bits].get(r_ref)
            if not t_ref:
                continue
            rx = [p[0] for p in pts]
            sp = [t_ref / p[1] if p[1] > 0 else 0 for p in pts]
            lbl = f"{bits}b ({label_base})" if vi == 0 else f"{bits}b ({label_base})"
            ax_right.plot(rx, sp, marker=mk, color=col, linestyle=ls_base,
                          linewidth=1.8, markersize=6,
                          label=lbl if vi == 0 else "_nolegend_")

    # ideal line
    ideal = [r / r_ref for r in all_ranks]
    ax_right.plot(all_ranks, ideal, "k--", linewidth=1.2, alpha=0.5, label="ideal")
    ax_right.axhline(1, color="grey", linewidth=0.6, linestyle=":")
    ax_right.set_xscale("log", base=2)
    ax_right.set_xticks(all_ranks)
    ax_right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax_right.xaxis.set_minor_locator(ticker.NullLocator())
    ax_right.set_xlabel("MPI ranks")
    ax_right.set_ylabel(f"Speedup over {r_ref} ranks")
    ax_right.legend(title="Size (solid=alg4, dash=multi)", fontsize=7, framealpha=0.9)
    hint(ax_right, "Speedup vs Ranks — both variants", "↑ higher is better")

    fig.suptitle(f"alg4 vs alg4_multi — strong scaling  (sieve)", fontsize=14, y=1.01)
    save(fig, os.path.join(out_dir, f"alg4_scaling_time_speedup.{fmt}"))

    # ── Figure 2: parallel efficiency per challenge size ──────────────────────
    fig2, ax = plt.subplots(figsize=(10, 5))

    for vi, variant in enumerate(variants):
        col_base, mk_base, ls_base, label_base = VARIANT_STYLE[variant]
        for i, bits in enumerate(rep_sizes):
            col, mk = PALETTE[i % len(PALETTE)]
            t_ref = lookup[variant][bits].get(r_ref)
            pts = [(r, lookup[variant][bits][r])
                   for r in all_ranks if r in lookup[variant].get(bits, {})]
            if not t_ref or len(pts) < 2:
                continue
            eff = [(t_ref * r_ref) / (r * t) if t > 0 else 0 for r, t in pts]
            rx  = [p[0] for p in pts]
            lbl = f"{bits}b {label_base}"
            ax.plot(rx, eff, marker=mk, color=col, linestyle=ls_base,
                    linewidth=1.8, markersize=6, label=lbl)

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", label="ideal (100 %)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_ranks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(0, 1.4)
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Parallel efficiency  T₀·R₀ / (R·Tᴿ)")
    ax.legend(title="Size (solid=alg4, dash=multi)", fontsize=8,
              loc="lower left", framealpha=0.9)
    hint(ax, "Parallel Efficiency — alg4 vs alg4_multi  (sieve)", "↑ higher is better")

    fig2.suptitle("alg4 vs alg4_multi — parallel efficiency", fontsize=14, y=1.01)
    save(fig2, os.path.join(out_dir, f"alg4_scaling_efficiency.{fmt}"))

    # ── Figure 3: scaling ratio (T_16 / T_32) per challenge size ─────────────
    # With exactly 2 rank points this is the clearest single-number summary.
    r_hi = all_ranks[-1]
    if r_hi == r_ref:
        return

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for vi, variant in enumerate(variants):
        col, mk, ls, label = VARIANT_STYLE[variant]
        xs, ratios, efficiencies = [], [], []
        for bits in all_sizes:
            t_lo = lookup[variant][bits].get(r_ref)
            t_hi = lookup[variant][bits].get(r_hi)
            if not t_lo or not t_hi or t_hi == 0:
                continue
            xs.append(bits)
            ratios.append(t_lo / t_hi)
            efficiencies.append((t_lo * r_ref) / (r_hi * t_hi))

        if not xs:
            continue
        ax1.plot(xs, ratios, marker=mk, color=col, linestyle=ls,
                 linewidth=1.8, markersize=6, label=label)
        ax2.plot(xs, efficiencies, marker=mk, color=col, linestyle=ls,
                 linewidth=1.8, markersize=6, label=label)

    ideal_sp  = r_hi / r_ref
    ideal_eff = 1.0
    ax1.axhline(ideal_sp,  color="grey", linewidth=0.8, linestyle="--",
                label=f"ideal ({ideal_sp:.0f}×)")
    ax2.axhline(ideal_eff, color="grey", linewidth=0.8, linestyle="--",
                label="ideal (100 %)")

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        set_xticks(ax, all_sizes)
        ax.set_xlabel("Challenge size (bits)")
        ax.legend(framealpha=0.9)

    ax1.set_ylabel(f"Speedup  T({r_ref}) / T({r_hi})")
    ax2.set_ylabel("Parallel efficiency  T₀·R₀ / (R·Tᴿ)")
    ax2.set_ylim(0, 1.4)
    hint(ax1, f"Scaling speedup {r_ref}→{r_hi} ranks by challenge size", "↑ higher is better")
    hint(ax2, f"Parallel efficiency {r_ref}→{r_hi} ranks by challenge size", "↑ higher is better")

    fig3.suptitle(f"alg4 vs alg4_multi — scaling {r_ref}→{r_hi} MPI ranks  (sieve)",
                  fontsize=14, y=1.01)
    save(fig3, os.path.join(out_dir, f"alg4_scaling_ratio.{fmt}"))


def main():
    parser = argparse.ArgumentParser(description="alg4 vs alg4_multi comparison")
    parser.add_argument("--out",    default="figures/alg4_comparison")
    parser.add_argument("--format", default="png", choices=["png", "pdf"], dest="fmt")
    parser.add_argument("--root",   default="results/alg4",
                        help="Root directory containing alg4 results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading from {args.root} ...")
    datasets = load_alg4_files(args.root)
    if not datasets:
        print("ERROR: no alg4 JSON files found"); return

    rank_counts = sorted(set(d["metadata"]["mpi_ranks"] for d in datasets))
    print(f"  {len(datasets)} datasets, rank counts: {rank_counts}")

    for ranks in rank_counts:
        plot_comparison_for_ranks(datasets, ranks, args.out, args.fmt)

    print("\nGenerating scaling plots ...")
    plot_scaling(datasets, args.out, args.fmt)

    total = sum(1 for f in os.listdir(args.out) if f.endswith(f".{args.fmt}"))
    print(f"\nDone. {total} {args.fmt.upper()} file(s) in {args.out}/")

if __name__ == "__main__":
    main()
