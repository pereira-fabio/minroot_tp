#!/usr/bin/env python3
"""Clean, single-entry analysis plotter for the MinRoot benchmark results.

Reads every JSON under results/aion/ and results/iris/, joins each timing file
with its _powm_ baseline sibling, and emits an organised image suite:

  <OUT>/scaling/    A. speedup vs parallelism, per algorithm (line per size)
  <OUT>/per_size/   B. speedup vs parallelism, per size    (line per algorithm)
  <OUT>/global/     C. speedup vs modulus, CPU(best rank) vs GPU(G4/G1), per algorithm
                       (+ GPU factor-only highlight for k16)
  <OUT>/per_config/ D. speedup vs modulus, per node/GPU count (line per algorithm)

Metric: speedup_total = baseline[size] / total_ms, where baseline[size] is the
median AION-N1 powm baseline at that size (used for BOTH clusters, since the
IRIS files store the known-bad forward-exponent baseline). GPU factor-only
speedup (baseline / factor_ms) is drawn as a dashed highlight for k16.

Usage: python3 plot_results.py <RESULTS_ROOT> <OUT_DIR> [--exp 3]
"""
import json, glob, re, os, sys, collections, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT     = sys.argv[2] if len(sys.argv) > 2 else "figures/analysis"
EXP     = int(sys.argv[sys.argv.index("--exp")+1]) if "--exp" in sys.argv else 3

SIZES = [256, 384, 512, 768, 1024, 2048]
def nom(b): return min(SIZES, key=lambda x: abs(x - (b or 0)))
NODE2RANK = {1:2, 2:4, 4:8, 8:16, 16:32, 32:64, 64:128}
# alg4_multi (our composite-cofactor extension) is intentionally excluded from
# these figures so readers do not conflate it with the paper's single-prime
# Algorithm 4; it is documented separately in THESIS.md §5.4 / §7.6.
ALGS = ["k16", "k32", "k35", "alg3", "alg4", "alg7"]
ALG_COLOR = {a: c for a, c in zip(ALGS, plt.cm.tab10.colors)}
SIZE_COLOR = {s: c for s, c in zip(SIZES, plt.cm.viridis([i/5 for i in range(6)]))}

# ---------------------------------------------------------------- load
def records():
    for cluster in ("aion", "iris"):
        for f in glob.glob(f"{RESULTS}/{cluster}/**/*.json", recursive=True):
            try: d = json.load(open(f))
            except Exception: continue
            m = d.get("metadata", {}); wa = d.get("warm_avg", {})
            alg = m.get("algorithm")
            if "alg4_multi" in os.path.basename(f): alg = "alg4_multi"
            meth = m.get("method")
            size = nom(m.get("challenge_bits"))
            exp = m.get("exponent")
            if cluster == "aion":
                mm = re.search(r"_N(\d+)", f); nodes = int(mm.group(1)) if mm else m.get("nodes")
                par = NODE2RANK.get(nodes)          # x-axis = MPI ranks
            else:
                # x-axis = GPUs ACTUALLY used. Trust metadata.n_gpus, not the /G<k>/
                # folder: single-GPU-only binaries (k16, alg7) are filed under G2/G3/G4
                # by a multi-GPU job but truthfully record n_gpus=1, so the folder name
                # would fabricate a flat 1->4 "scaling" curve. Fall back to the folder
                # only if metadata lacks the field.
                par = m.get("n_gpus")
                if not par:
                    mm = re.search(r"/G(\d+)/", f); par = int(mm.group(1)) if mm else None
            med = d.get("warm_stats", {}).get("total_ms", {}).get("median", 0)
            yield dict(cluster=cluster, alg=alg, method=meth, size=size, exp=exp,
                       par=par, base=m.get("powm_baseline_ms", 0),
                       factor=wa.get("factor_ms", 0), total=wa.get("total_ms", 0),
                       total_med=med)

RECS = [r for r in records() if r["alg"] in ALGS and r["par"]]

# canonical baseline per size: median AION-N1 (par==2) e3 powm baseline
_b = collections.defaultdict(list)
for r in RECS:
    if r["cluster"] == "aion" and r["method"] == "powm" and r["par"] == 2 and r["exp"] == 3 and r["base"]:
        _b[r["size"]].append(r["base"])
BASE = {s: statistics.median(v) for s, v in _b.items()}
print("baseline[size] (ms):", {s: round(BASE[s], 4) for s in sorted(BASE)})

def speed(r, which="total"):
    # alg4_multi factor_ms is heavy-tailed -> use median total (2026-07-03 decision)
    if r is None: return None
    b = BASE.get(r["size"])
    if which == "total" and r["alg"] == "alg4_multi":
        t = r.get("total_med") or r.get("total")
    else:
        t = r.get(which)
    return b / t if (b and t) else None

# index timing records (exclude powm baseline files)
TIM = [r for r in RECS if r["method"] in ("sieve", "table")]
def pick(cluster, alg, par, exp, size, prefer="table"):
    cand = [r for r in TIM if r["cluster"] == cluster and r["alg"] == alg
            and r["par"] == par and r["exp"] == exp and r["size"] == size]
    if not cand: return None
    for m in (prefer, "sieve", "table"):
        for r in cand:
            if r["method"] == m: return r
    return cand[0]

def fmt_x(v):
    if v >= 100: return f"{v:.0f}×"
    if v >= 1:   return f"{v:.1f}×"
    return f"{v:.2g}×"

def fmt_ms(v, _):
    if v >= 10:   return f"{v:.0f}"
    if v >= 1:    return f"{v:.1f}"
    if v >= 0.01: return f"{v:.2f}"
    return f"{v:.3f}"

def titled(ax, title, direction):
    """Axis title with a centered italic grey subtitle giving the reading direction
    (no arrow), mirroring ploter.py's hint_title style."""
    ax.set_title(title, fontsize=11, pad=16)
    ax.text(0.5, 1.0, direction, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#888", style="italic", clip_on=False)

def peak_of(lines):
    best = None
    for xa, ya in lines:
        for y in ya:
            if y and (best is None or y > best): best = y
    return best

def peak_corner(ax, peak):
    """Peak value in the top-left corner (where the direction hint used to sit)."""
    if peak is None: return
    ax.text(0.015, 0.985, f"peak {fmt_x(peak)}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, fontweight="bold", color="#222",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

def style_speedup_ax(ax, xlabel, ylabel, baseline=True):
    ax.set_yscale("log")
    if baseline:
        ax.axhline(1.0, color="black", ls="--", lw=1.1, alpha=0.8, zorder=1)
        ax.text(0.985, 1.06, "sequential baseline (1×)", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7, color="black", alpha=0.75)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}×"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, which="major", ls="-", alpha=0.22)
    ax.grid(True, which="minor", ls=":", alpha=0.12)

def save(fig, sub, name):
    d = os.path.join(OUT, sub); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + ".png")
    fig.tight_layout(); fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p

XRANKS = [2,4,8,16,32,64,128]; XGPU = [1,2,3,4]

def total_of(r):
    """total_ms for a record; median for the heavy-tailed alg4_multi path."""
    if r is None: return None
    if r["alg"] == "alg4_multi": return r.get("total_med") or r.get("total")
    return r.get("total")

def add_projection(ax, xs, ys, xmax, color, target=1.0):
    """Power-law (log-log) extrapolation of a rising speedup tail past the ceiling.
    Draws a dashed continuation and marks the projected crossing of `target`
    (default 1× break-even). Returns that rank count, or None if not applicable."""
    pts = [(x, y) for x, y in zip(xs, ys) if y]
    if len(pts) < 3 or pts[-1][1] <= pts[-2][1]:      # only extend a rising curve
        return None
    tail = pts[-4:] if len(pts) >= 4 else pts
    a, b = np.polyfit(np.log([p[0] for p in tail]), np.log([p[1] for p in tail]), 1)
    if a <= 0: return None
    xp = np.geomspace(pts[-1][0], xmax, 50)
    ax.plot(xp, np.exp(b + a * np.log(xp)), ls=(0, (4, 3)), color=color, lw=1.6,
            alpha=0.85, zorder=3)
    x_t = np.exp((np.log(target) - b) / a)
    if pts[-1][0] < x_t <= xmax:
        ax.scatter([x_t], [target], marker="X", s=60, color=color, zorder=6,
                   edgecolor="white", linewidth=0.7)
        ax.annotate(f"proj. break-even\n~{x_t:.0f} ranks", xy=(x_t, target),
                    xytext=(0, -24), textcoords="offset points", ha="center",
                    fontsize=7, color=color, fontweight="bold")
    return x_t

# ------------------------------------------------- A. strong scaling (ploter style)
def cat_A():
    """Two panels per (cluster, alg, exp): left = total time vs parallelism
    (lower is better), right = strong-scaling speedup relative to the base
    rank/GPU count with an ideal linear-scaling reference (higher is better)."""
    n = 0
    for cluster, xs, xlab, unit in (("aion", XRANKS, "MPI ranks", "ranks"),
                                    ("iris", XGPU, "GPUs", "GPUs")):
        for alg in ALGS:
            for exp in sorted({r["exp"] for r in TIM if r["cluster"]==cluster and r["alg"]==alg}):
                series = []
                for s in SIZES:
                    pts = [(x, total_of(pick(cluster, alg, x, exp, s))) for x in xs]
                    pts = [(x, t) for x, t in pts if t]
                    if len(pts) >= 2: series.append((s, pts))
                if not series: continue
                r_ref = min(x for _, pts in series for x, _ in pts)     # base parallelism
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
                Lspd = []
                for s, pts in series:
                    ranks = [p[0] for p in pts]; total = [p[1] for p in pts]
                    t_ref = pts[0][1]
                    spd = [t_ref / t for t in total]
                    ax1.plot(ranks, total, "o-", color=SIZE_COLOR[s], label=f"{s}b")
                    ax2.plot(ranks, spd,   "o-", color=SIZE_COLOR[s], label=f"{s}b")
                    Lspd.append((ranks, spd))
                ax2.plot(xs, [x / r_ref for x in xs], "k--", lw=1.2, alpha=0.55, label="ideal")
                for ax in (ax1, ax2):
                    ax.set_xscale("log", base=2); ax.set_xticks(xs)
                    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax.set_xlabel(xlab)
                    ax.grid(True, which="major", ls="-", alpha=0.22); ax.set_axisbelow(True)
                # left: time (lower is better)
                ax1.set_yscale("log")
                ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_ms))
                ax1.set_ylabel("total time (ms)  [log]")
                titled(ax1, f"{cluster.upper()} {alg} (e{exp}): time vs {unit}", "lower is better")
                ax1.legend(title="modulus", fontsize=8, ncol=2)
                # right: strong-scaling speedup vs base (higher is better)
                ax2.set_yscale("log")
                ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}×"))
                ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax2.axhline(1, color="grey", ls=":", lw=0.8)
                u1 = unit if r_ref != 1 else unit.rstrip("s")   # "1 GPU" not "1 GPUs"
                ax2.set_ylabel(f"speedup over {r_ref} {u1}")
                titled(ax2, f"{cluster.upper()} {alg} (e{exp}): strong scaling", "higher is better")
                peak_corner(ax2, peak_of(Lspd))
                ax2.legend(title="modulus", fontsize=8, ncol=2)
                n += 1; save(fig, "scaling", f"{cluster}_{alg}_e{exp}")
    return n

# ------------------------------------------------- B. per-size (per alg lines)
def cat_B():
    n = 0
    for cluster, xs, xlab in (("aion", XRANKS, "MPI ranks"), ("iris", XGPU, "GPUs")):
        for exp in sorted({r["exp"] for r in TIM if r["cluster"]==cluster}):
            for s in SIZES:
                fig, ax = plt.subplots(figsize=(6.2, 4.2)); any_line = False; L = []
                for alg in ALGS:
                    ys = []
                    for x in xs:
                        cand = [r for r in TIM if r["cluster"]==cluster and r["alg"]==alg
                                and r["par"]==x and r["exp"]==exp and r["size"]==s]
                        best = None
                        for m in ("table","sieve"):
                            for r in cand:
                                if r["method"]==m: best=r; break
                            if best: break
                        ys.append(speed(best) if best else None)
                    xy = [(x,y) for x,y in zip(xs,ys) if y]
                    if not xy: continue
                    any_line = True
                    xa=[p[0] for p in xy]; ya=[p[1] for p in xy]
                    ax.plot(xa, ya, "o-", color=ALG_COLOR[alg], label=alg); L.append((xa,ya))
                if not any_line: plt.close(fig); continue
                if cluster=="aion": ax.set_xscale("log", base=2); ax.set_xticks(xs)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                style_speedup_ax(ax, xlab, "speedup vs sequential  (baseline / total)")
                titled(ax, f"{cluster.upper()} {s}b (e{exp}): algorithm comparison", "higher is better")
                peak_corner(ax, peak_of(L))
                ax.legend(fontsize=8, ncol=2)
                n += 1; save(fig, "per_size", f"{cluster}_{s}b_e{exp}")
    return n

# ------------------------------------------------- C. global CPU vs GPU
def cat_C():
    n = 0
    def best_at(cl, alg, par, s):
        cand = [r for r in TIM if r["cluster"]==cl and r["alg"]==alg
                and r["par"]==par and r["size"]==s and r["exp"]==EXP]
        return next((r for m in ("table","sieve") for r in cand if r["method"]==m), None)
    for alg in ALGS:
        fig, ax = plt.subplots(figsize=(6.4, 4.4)); any_line = False; L = []
        # AION: best over all rank counts (fair to k16, which peaks below N64)
        ys = []
        for s in SIZES:
            vals = [speed(best_at("aion", alg, x, s)) for x in XRANKS]
            vals = [v for v in vals if v]
            ys.append(max(vals) if vals else None)
        xy = [(s,y) for s,y in zip(SIZES,ys) if y]
        if xy:
            any_line = True
            xa=[p[0] for p in xy]; ya=[p[1] for p in xy]
            ax.plot(xa, ya, "o-", color="#c0392b", label="AION cluster (best rank)"); L.append((xa,ya))
        # IRIS G4 / G1 (total)
        for label, par, ls in [("IRIS G4", 4, "s-"), ("IRIS G1", 1, "s:")]:
            ys = [speed(best_at("iris", alg, par, s)) for s in SIZES]
            xy = [(s,y) for s,y in zip(SIZES,ys) if y]
            if not xy: continue
            any_line = True
            xa=[p[0] for p in xy]; ya=[p[1] for p in xy]
            ax.plot(xa, ya, ls, color="#2471a3", label=label); L.append((xa,ya))
        # k16 GPU factor-only highlight
        if alg == "k16":
            ys = []
            for s in SIZES:
                cand = [r for r in TIM if r["cluster"]=="iris" and r["alg"]=="k16"
                        and r["par"]==1 and r["size"]==s and r["exp"]==5]
                best = next((r for r in cand), None)
                ys.append(speed(best,"factor") if best else None)
            xy=[(s,y) for s,y in zip(SIZES,ys) if y]
            if xy:
                xa=[p[0] for p in xy]; ya=[p[1] for p in xy]
                ax.plot(xa, ya, "^--", color="#27ae60", label="IRIS G1 (factor-only)"); L.append((xa,ya))
        if not any_line: plt.close(fig); continue
        style_speedup_ax(ax, "modulus size (bits)", "speedup vs sequential  (baseline / total)")
        ax.set_xscale("log", base=2); ax.set_xticks(SIZES)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        titled(ax, f"{alg}: CPU cluster vs GPU (e{EXP})", "higher is better")
        peak_corner(ax, peak_of(L))
        ax.legend(fontsize=8)
        n += 1; save(fig, "global", f"{alg}")
    return n

# ------------------------------------------------- D. per-config (line per alg)
def cat_D():
    n = 0
    for cluster, cfgs, tag in (("aion", XRANKS, "r"), ("iris", XGPU, "G")):
        for par in cfgs:
            for exp in sorted({r["exp"] for r in TIM if r["cluster"]==cluster and r["par"]==par}):
                fig, ax = plt.subplots(figsize=(6.2, 4.2)); any_line=False; L=[]
                for alg in ALGS:
                    ys=[]
                    for s in SIZES:
                        cand=[r for r in TIM if r["cluster"]==cluster and r["alg"]==alg
                              and r["par"]==par and r["size"]==s and r["exp"]==exp]
                        best=next((r for m in ("table","sieve") for r in cand if r["method"]==m),None)
                        ys.append(speed(best) if best else None)
                    xy=[(s,y) for s,y in zip(SIZES,ys) if y]
                    if not xy: continue
                    any_line=True
                    xa=[p[0] for p in xy]; ya=[p[1] for p in xy]
                    ax.plot(xa, ya, "o-", color=ALG_COLOR[alg], label=alg); L.append((xa,ya))
                if not any_line: plt.close(fig); continue
                style_speedup_ax(ax, "modulus size (bits)", "speedup vs sequential  (baseline / total)")
                ax.set_xscale("log", base=2); ax.set_xticks(SIZES)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                cfg = f"{par}r" if cluster=="aion" else f"G{par}"
                titled(ax, f"{cluster.upper()} {cfg} (e{exp}): all algorithms", "higher is better")
                peak_corner(ax, peak_of(L))
                ax.legend(fontsize=8, ncol=2)
                n += 1; save(fig, "per_config", f"{cluster}_{cfg}_e{exp}")
    return n

# ------------------------------------------------- E. projection (AION rank scaling)
def cat_E():
    """Speedup-vs-sequential vs ranks (AION), with the 64-node/128-rank ceiling and a
    power-law extrapolation of the 2048b curve to its projected 1× break-even.
    Complements cat A's ideal-line view with the 'how many ranks to win' story."""
    n = 0
    XMAX = 2048
    for alg in ALGS:
        for exp in sorted({r["exp"] for r in TIM if r["cluster"]=="aion" and r["alg"]==alg}):
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            L = []; proj_xy = None; any_line = False
            for s in SIZES:
                ys = [speed(pick("aion", alg, x, exp, s)) for x in XRANKS]
                xy = [(x, y) for x, y in zip(XRANKS, ys) if y]
                if not xy: continue
                any_line = True
                xa = [p[0] for p in xy]; ya = [p[1] for p in xy]
                ax.plot(xa, ya, "o-", color=SIZE_COLOR[s], label=f"{s}b"); L.append((xa, ya))
                if s == 2048: proj_xy = (xa, ya)
            if not any_line: plt.close(fig); continue
            ax.axvspan(128, XMAX, color="grey", alpha=0.07, zorder=0)
            ax.axvline(128, color="grey", ls="-", lw=1, alpha=0.6)
            if proj_xy: add_projection(ax, proj_xy[0], proj_xy[1], XMAX, SIZE_COLOR[2048])
            ax.set_xscale("log", base=2); ax.set_xticks([2, 8, 32, 128, 512, 2048])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            style_speedup_ax(ax, "MPI ranks", "speedup vs sequential  (baseline / total)")
            ax.text(128, ax.get_ylim()[0] * 1.5, " 64-node ceiling ", rotation=90,
                    va="bottom", ha="right", fontsize=6.5, color="grey")
            titled(ax, f"AION {alg} (e{exp}): projection",
                   "higher is better · dashed = extrapolation beyond the 128-rank ceiling")
            peak_corner(ax, peak_of(L))
            ax.legend(title="modulus", fontsize=8, ncol=2, loc="lower right")
            n += 1; save(fig, "projection", f"aion_{alg}_e{exp}")
    return n

if __name__ == "__main__":
    import matplotlib.ticker
    print("records:", len(RECS), "| timing:", len(TIM))
    a=cat_A(); b=cat_B(); c=cat_C(); d=cat_D(); e=cat_E()
    print(f"generated  A(scaling)={a}  B(per_size)={b}  C(global)={c}  "
          f"D(per_config)={d}  E(projection)={e}  total={a+b+c+d+e}")

# Edit: 2026-07-10
