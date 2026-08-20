#!/usr/bin/env python3
"""crt_best_alpha.py — pick the best per-step exponent alpha for the CRT solver.

Reads the alg_crt alpha-sweep JSONs produced by scripts/aion/alg_crt.slurm
(files named  alg_crt_a<ALPHA>_crt_<JOBID>_<size>.json)  and, for each
(modulus size, node count), reports the alpha with the highest speedup_total.

alpha is taken from the filename (the solver does not record it internally);
size / nodes / speedup_total come from the JSON (metadata + warm_avg).

Usage:
    python3 analysis/crt_best_alpha.py data/results/aion/alg_crt
    python3 analysis/crt_best_alpha.py 'data/results/aion/alg_crt/**/alg_crt_a*_*.json'
"""
import sys
import os
import re
import json
import glob
from collections import defaultdict

ALPHA_RE = re.compile(r"_a(\d+)_")


def find_files(arg):
    if os.path.isdir(arg):
        return glob.glob(os.path.join(arg, "**", "alg_crt_a*_*.json"), recursive=True)
    return glob.glob(arg, recursive=True)


def load(path):
    try:
        with open(path) as fh:
            j = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    m = j.get("metadata", {})
    wa = j.get("warm_avg", {})
    am = ALPHA_RE.search(os.path.basename(path))
    # alpha from filename (AION) or from JSON metadata (GPU); parallelism axis is
    # nodes (AION) or n_gpus (GPU).
    alpha = int(am.group(1)) if am else m.get("alpha")
    size = m.get("challenge_bits")
    nodes = m.get("nodes", m.get("n_gpus"))
    spd = wa.get("speedup_total")
    if alpha is None or size is None or nodes is None or spd is None:
        return None
    return {
        "alpha": int(alpha),
        "size": int(size),
        "nodes": int(nodes),
        "speedup": float(spd),
        "total_ms": wa.get("total_ms"),
        "baseline": m.get("powm_baseline_ms"),
    }


def cell(text, width=15):
    return f"{text:>{width}}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    files = []
    for a in sys.argv[1:]:
        files += find_files(a)
    records = [r for r in (load(f) for f in files) if r]
    if not records:
        print("No matching alg_crt alpha-sweep JSON files found.")
        sys.exit(1)

    # group[(size, nodes)] -> {alpha: record}
    group = defaultdict(dict)
    for r in records:
        group[(r["size"], r["nodes"])][r["alpha"]] = r

    sizes = sorted({s for (s, _) in group})
    nodes = sorted({n for (_, n) in group})
    alphas = sorted({r["alpha"] for r in records})
    print(f"\nParsed {len(records)} runs | alphas swept: {alphas} | "
          f"sizes: {sizes} | node counts: {nodes}")

    # ---- Table 1: best alpha per (size x nodes) ----
    print("\n=== Best alpha per (modulus size x node count) — cell = alpha (speedup_total) ===\n")
    header = cell("bits \\ nodes", 12) + "".join(cell("N" + str(n)) for n in nodes)
    print(header)
    print("-" * len(header))
    for s in sizes:
        line = cell(str(s), 12)
        for n in nodes:
            recs = group.get((s, n))
            if recs:
                b = max(recs.values(), key=lambda r: r["speedup"])
                line += cell(f"{b['alpha']} ({b['speedup']:.2f}x)")
            else:
                line += cell("-")
        print(line)

    # ---- Table 2: full alpha curve at the largest node count ----
    nmax = nodes[-1]
    print(f"\n=== speedup_total vs alpha at N{nmax} (the tradeoff curve; * = best) ===\n")
    header = cell("bits \\ alpha", 12) + "".join(cell("a=" + str(a)) for a in alphas)
    print(header)
    print("-" * len(header))
    for s in sizes:
        recs = group.get((s, nmax), {})
        best_alpha = max(recs, key=lambda a: recs[a]["speedup"]) if recs else None
        line = cell(str(s), 12)
        for a in alphas:
            if a in recs:
                mark = "*" if a == best_alpha else " "
                line += cell(f"{recs[a]['speedup']:.2f}{mark}")
            else:
                line += cell("-")
        print(line)
    print("\n(speedup_total = powm_baseline_ms / total_ms; >1x beats the sequential baseline.)")


if __name__ == "__main__":
    main()
