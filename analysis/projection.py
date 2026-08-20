#!/usr/bin/env python3
"""
projection.py — Theoretical scaling projection beyond the measured rank range.

WHY THIS EXISTS
----------------
The AION allocation has a hard ceiling: 64 nodes / 8192 logical cores (the
number SLURM reports while a job is running), already fully exercised by the
N64 = 128 MPI-rank measurement point in every strong-scaling table in the
wiki. For algorithms that don't yet beat the paper's target at 128 ranks
(k35, Algorithm 7), there is no way to actually test more ranks. This script
fits the empirically observed strong-scaling efficiency from the last few
*measured* points and extrapolates total_ms / speedup_total to larger
HYPOTHETICAL rank counts, purely to estimate how big the remaining gap is.

None of the extrapolated rows are achievable on the current allocation —
they answer "how many ranks would it take", not "what should we request".

METHOD
------
Empirically, total_ms(ranks) follows an approximate power law:

    total_ms(r) ~= A * r^(-p)

where p in (0, 1] is the local parallel efficiency (p=1 => perfect doubling,
i.e. total_ms exactly halves when ranks double; p<1 => sub-linear, usually
because reconstruction/communication overhead is creeping in). `fit_power_law`
estimates A and p via ordinary least squares on (ln ranks, ln total_ms) over
the last `fit_window` measured points (default 3) — i.e. it captures the
*current* scaling regime, including any recent efficiency decline, rather
than an average over the whole curve (which would be over-optimistic, since
the curve generally bends down as recon_ms/combine overhead grows with rank
count — see wiki/results/speedup-scaling.md and wiki/results/reconstruction-bottleneck.md).

HOW TO REFRESH WHEN NEW DATA LANDS
-----------------------------------
Replace the (ranks, total_ms) tuples in DATASETS with the new measured
values (same units, same challenge size) and re-run. Everything else
(fit, extrapolation, ranks-to-target solve, table rendering) recomputes
automatically. No other code change needed.
"""

import math

# Sequential mpz_powm baseline at 24576-bit modulus (ms). Same for every
# algorithm at this modulus size — see wiki/implementation/measurement-methodology.md.
POWM_BASELINE_MS = 1801.0

# Paper's claimed speedup for each algorithm (Table 1, Biryukov et al. 2024).
# k35 is Algorithm 2 at the paper's primary B=2^35 parameter (~42x, not ~43x
# -- ~43x is OUR measured k32 result, see wiki/lint-report.md C6).
PAPER_TARGET = {
    "k35_sieve": 42.0,
    "k35_table": 42.0,
    "alg7": 20.0,
}

# Measured (ranks, total_ms) pairs at 24576 bits, backed out from the published
# speedup_total = POWM_BASELINE_MS / total_ms tables (no raw per-run JSON for
# k35/alg7 is present on this machine -- see wiki/lint-report.md S1 -- so this
# is reconstructed from the rounded speedup_total values already in the wiki,
# which is precise to about +/-1%, plenty for an order-of-magnitude projection).
#
# Sources: wiki/results/alg2-full-scaling.md (k35 sieve/table, jobs
# alg2_12175008/k35_mpi_N*) and wiki/results/alg7-results.md (alg7, jobs
# alg7_12315350/N16, alg7_12315351/N8).
#
# *** PRELIMINARY — predates the AION re-run in progress as of 2026-06-25.
# *** Replace these tuples with the fresh numbers once that batch lands.
DATASETS = {
    "k35_sieve": [
        (2, 7504.2), (4, 4392.7), (8, 2339.0), (16, 1286.4),
        (32, 669.5), (64, 352.5), (128, 203.1),
    ],
    "k35_table": [
        (2, 6670.4), (4, 3463.5), (8, 1731.7), (16, 918.9),
        (32, 480.3), (64, 260.6), (128, 148.0),
    ],
    # Only two rank counts were ever benchmarked for alg7 (N8=16r, N16=32r) --
    # the fit below degenerates to the single observed ratio between them.
    "alg7": [
        (16, 5336.7), (32, 2755.0),
    ],
}


def fit_power_law(points):
    """Least-squares fit of total_ms = A * ranks^-p in log-log space. Returns (A, p)."""
    xs = [math.log(r) for r, _ in points]
    ys = [math.log(t) for _, t in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den          # slope == -p
    intercept = mean_y - slope * mean_x
    return math.exp(intercept), -slope


def ranks_to_reach(a, p, target_total_ms):
    """Invert total_ms(r) = A * r^-p to solve for r at a given total_ms target."""
    return (a / target_total_ms) ** (1.0 / p)


def project(name, points, fit_window=3, hypothetical_ranks=None):
    max_measured_ranks = max(r for r, _ in points)
    default_targets = [128, 256, 512, 1024, 2048, 4096, 8192]
    hypothetical_ranks = hypothetical_ranks or sorted(set(default_targets) | {max_measured_ranks})
    window = points[-fit_window:] if len(points) >= fit_window else points
    a, p = fit_power_law(window)
    eff_per_doubling = 2 ** p  # actual speedup ratio achieved per rank-doubling, ideal = 2.0

    print(f"\n=== {name} ===")
    print(f"Fit window: last {len(window)} measured point(s): {window}")
    print(f"Max ranks actually benchmarked: {max_measured_ranks} "
          f"({'= the 128-rank/64-node ceiling' if max_measured_ranks >= 128 else 'NOT yet run up to the 128-rank ceiling'})")
    print(f"total_ms(r) ~= {a:.1f} * r^-{p:.4f}   "
          f"(local efficiency {p*100:.1f}%, {eff_per_doubling:.3f}x per rank-doubling vs ideal 2.0x)")

    print(f"{'ranks':>8} | {'total_ms':>10} | {'speedup_total':>13} | note")
    print("-" * 60)
    for r in hypothetical_ranks:
        total_ms = a * (r ** -p)
        speedup = POWM_BASELINE_MS / total_ms
        if r == max_measured_ranks:
            note = "fitted (last measured point)"
        elif r > 128:
            note = "HYPOTHETICAL — exceeds 64-node/8192-core ceiling"
        else:
            note = "HYPOTHETICAL — within ceiling but not yet benchmarked at this rank count"
        print(f"{r:8d} | {total_ms:10.2f} | {speedup:12.2f}x | {note}")

    target = PAPER_TARGET.get(name)
    if target:
        target_total_ms = POWM_BASELINE_MS / target
        ranks_needed = ranks_to_reach(a, p, target_total_ms)
        nodes_needed = ranks_needed / 2.0  # 2 MPI ranks/node, per hpc-environment.md
        print(f"\nRanks needed to reach paper's {target:.0f}x target: "
              f"~{ranks_needed:.0f} ranks (~{nodes_needed:.0f} nodes) — "
              f"{ranks_needed/128:.1f}x the 128-rank ceiling.")
    return a, p


def main():
    for name, points in DATASETS.items():
        project(name, points)
    print("\nAll rows above r=128 are theoretical extrapolation, not achievable on the "
          "current 64-node/8192-core allocation. Re-run with fresh DATASETS once the "
          "in-progress AION batch lands.")


if __name__ == "__main__":
    main()
