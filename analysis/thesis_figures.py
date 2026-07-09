#!/usr/bin/env python3
"""Curated thesis figures (Overleaf-ready PDFs, no embedded title — the LaTeX
\\caption supplies it) at the exact paths the THESIS.md placeholders reference.

IMPORTANT: these plot the values *as reported in the THESIS.md tables* so the
figures and the text agree exactly. Numbers are transcribed from the canonical
(e5) tables in §7.2 (k16 table method) and §7.12 (GPU k16 factor-only), NOT
recomputed from the 2026-07-03/04 confirmation batch (which uses slightly
different per-challenge baselines and would disagree with the printed tables).

Produces:
  figures/k16_paperscale_speedup.pdf   (§7.2)
  figures/k16_gpu_speedup.pdf          (§7.12)

Usage: python3 thesis_figures.py <FIG_DIR> [--png]
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

FIGDIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "figures"
EXT = "png" if "--png" in sys.argv else "pdf"

SIZES = [256, 384, 512, 768, 1024, 2048]
RANKS = [2, 4, 8, 16, 32, 64, 128]
SIZE_COLOR = {s: c for s, c in zip(SIZES, plt.cm.viridis([i/5 for i in range(6)]))}

# ── §7.2  k16 table-method speedup_total  (THESIS.md §7.2 table) ─────────────
#   rows keyed by modulus, columns = RANKS (2 … 128)
K16_PAPERSCALE = {
    256:  [0.76, 0.72, 0.45, 0.41, 0.38, 0.30, 0.29],
    384:  [1.09, 1.06, 0.98, 0.92, 0.80, 0.62, 0.52],
    512:  [1.85, 1.97, 2.02, 1.62, 1.31, 0.97, 0.63],
    768:  [3.90, 4.37, 3.90, 3.51, 3.03, 2.38, 1.76],
    1024: [5.45, 6.74, 7.36, 6.77, 5.63, 4.18, 3.24],
    2048: [12.9, 19.8, 26.4, 27.1, 21.6, 14.9, 9.8],
}
# ── §7.12  single-V100 k16 factor-only speedup  (THESIS.md §7.12 table) ──────
K16_GPU = {256: 35.6, 384: 46.4, 512: 71.7, 768: 147.4, 1024: 221.8, 2048: 800.6}


def baseline_line(ax):
    ax.axhline(1.0, color="black", ls="--", lw=1.1, alpha=0.8)
    ax.text(0.985, 1.06, "sequential baseline (1×)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="black", alpha=0.75)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda y, _: f"{y:g}×"))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(True, which="major", ls="-", alpha=0.22); ax.set_axisbelow(True)


def savefig(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, f"{name}.{EXT}")
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_k16_paperscale():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for s in SIZES:
        ax.plot(RANKS, K16_PAPERSCALE[s], "o-", color=SIZE_COLOR[s], label=f"{s}b")
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    ax.axvline(16, color="grey", ls=":", lw=1, alpha=0.7)
    baseline_line(ax)
    ax.text(16, ax.get_ylim()[0] * 1.4, " N8 peak (27.1×) ", rotation=90,
            va="bottom", ha="right", fontsize=7, color="grey")
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup over sequential")
    ax.legend(title="modulus", fontsize=8, ncol=2, loc="lower left")
    savefig(fig, "k16_paperscale_speedup")


def fig_k16_gpu():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    xs = SIZES; ys = [K16_GPU[s] for s in SIZES]
    ax.plot(xs, ys, "^-", color="#27ae60", lw=2, markersize=7,
            label="single V100 (k16, factor-only)")
    for s, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}×" if y >= 100 else f"{y:.1f}×", (s, y), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=7.5, color="#1e7a44")
    ax.axhline(42, color="#c0392b", ls="--", lw=1.3, alpha=0.85)
    ax.text(xs[0], 46, "paper target 42×", color="#c0392b", fontsize=8, va="bottom")
    ax.set_xscale("log", base=2); ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    baseline_line(ax)
    ax.set_xlabel("modulus size (bits)")
    ax.set_ylabel("speedup over sequential (sieve step)")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "k16_gpu_speedup")


# ── §7.11.1  large-moduli strong scaling at 24576b  (THESIS.md §7.11 tables) ─
LM_24576 = {  # (ranks, speedup_total) at 24576 bits
    "k16":  ([2, 4, 8, 16, 32, 64, 128], [21.58, 42.30, 80.10, 138.60, 189.88, 174.30, 116.91]),
    "k32":  ([2, 8, 32, 64, 128],        [1.77, 5.38, 17.60, 30.73, 43.20]),
    "alg3": ([2, 8, 32, 64, 128],        [1.74, 5.30, 17.64, 30.27, 42.88]),
    "alg4": ([2, 8, 32, 64, 128],        [1.66, 5.12, 17.33, 29.15, 40.73]),
}
LM_COLOR = {"k16": "#1f77b4", "k32": "#ff7f0e", "alg3": "#d62728", "alg4": "#9467bd"}

def fig_large_moduli():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for alg, (xs, ys) in LM_24576.items():
        ax.plot(xs, ys, "o-", color=LM_COLOR[alg], label=alg)
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter())
    ax.axvline(32, color="grey", ls=":", lw=1, alpha=0.7)
    baseline_line(ax)
    ax.text(32, ax.get_ylim()[0] * 1.4, " k16 peak (N16) ", rotation=90,
            va="bottom", ha="right", fontsize=7, color="grey")
    ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup over sequential")
    ax.legend(title="algorithm @ 24576b", fontsize=8, loc="upper left")
    savefig(fig, "large_moduli_scaling")

# ── §7.9  Gap 1: k32 (128r, sieve) speedup vs modulus, both regimes ──────────
GAP1 = {  # THESIS.md §7.3 (paper-scale) + §7.11.2 (large moduli), 128-rank sieve
    384: 0.01, 512: 0.01, 768: 0.03, 1024: 0.07, 2048: 0.38,
    4096: 2.19, 8192: 7.90, 12288: 15.56, 16384: 23.13, 20480: 34.47, 24576: 43.20,
}

def fig_gap1():
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    xs = sorted(GAP1); ys = [GAP1[s] for s in xs]
    ax.axvspan(300, 3000, color="#f2d4d4", alpha=0.4)      # sieve-dominated regime
    ax.plot(xs, ys, "o-", color="#c0392b", lw=2, label="k32 sieve, 128 ranks")
    ax.axvline(3000, color="grey", ls=":", lw=1, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384, 24576])
    ax.get_xaxis().set_major_formatter(mt.ScalarFormatter()); ax.tick_params(axis="x", rotation=45)
    baseline_line(ax)
    ax.text(430, 0.02, "paper-scale\n256–2048b:\nsieve dominates,\n$\\ll 1\\times$",
            fontsize=7.5, va="center", color="#7a2020")
    ax.text(9000, 4, "large moduli:\nbaseline dominates\n→ 43× (52× table)",
            fontsize=7.5, va="center", color="#1e5a1e")
    ax.text(3000, ax.get_ylim()[1] * 0.5, " break-even ≈3000b ", rotation=90,
            va="top", ha="right", fontsize=7, color="grey")
    ax.set_xlabel("modulus size (bits)"); ax.set_ylabel("speedup over sequential")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "gap1_two_regime")

if __name__ == "__main__":
    fig_k16_paperscale()
    fig_k16_gpu()
    fig_large_moduli()
    fig_gap1()

# Edit: 2026-07-06
