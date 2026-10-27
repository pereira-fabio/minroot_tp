#!/usr/bin/env python3
"""
ideal_combine.py — Projected speedup_total with an idealized log-depth combine.

WHY THIS EXISTS
----------------
The paper's circuit-depth model has zero communication/synchronization
overhead: its "T" (latency) values assume 2^54.5/2^48/2^40 dedicated ASIC
units and count only the reconstruction multiplication tree's depth
(log2(group_size) ~ 4-5 multiplications, per Biryukov et al.). Our real
implementation gathers all N ranks' partial cbrt products onto rank 0 and
multiplies them SERIALLY (N-1 sequential mulmods) -- this is the one
overhead source big enough to matter (MPI Bcast/Gather/Barrier and OMP lock
contention are both already known to be <1% of total time at this scale; see
wiki/implementation/mpi-openmp-design.md). This script projects what
speedup_total would be if that serial combine were replaced with a log-depth
tree (ceil(log2(N)) rounds instead of N-1), covering every benchmarked
algorithm and method (sieve vs table, where both exist).

This was first deferred as a code change (perf-fix-plan 2026-06-11, item
"D-review") and is now implemented: ksc_reduce_product in
src/common/solver_common.cpp tree-reduces with MPI_Reduce and a custom op.
Correctness is checked locally by `make test` at any rank count; only the
speedup needs the cluster. Until fresh cluster numbers land, this script is
the estimate of what that change buys, computed from the serial-combine data.

METHOD
------
combine_ms(N) for the *current* serial design is observed to be extremely
linear in (N-1) -- see k16's directly-measured combine_ms at 24576 bits
(wiki/results/speedup-scaling.md "Why k16 Peaks at N16"), which gives a
per-multiplication cost c with R^2 effectively 1.0 across all 5 measured rank
counts. This script fits c from that data, then works ENTIRELY off published
speedup_total tables (no separate factor_ms/recon_ms breakdown needed):

    total_measured(N) = powm_baseline_ms / speedup_measured(N)
    combine_serial(N) = c * (N - 1)              -- current design
    combine_ideal(N)  = c * ceil(log2(N))        -- log-depth tree
    total_ideal(N)    = total_measured(N) - combine_serial(N) + combine_ideal(N)
    speedup_ideal(N)  = powm_baseline_ms / total_ideal(N)

This is algebraically identical to subtracting/re-adding combine_ms from a
factor_ms+recon_ms split (total = factor_ms + other_recon + combine, so
total - combine_serial + combine_ideal = factor_ms + other_recon +
combine_ideal, unchanged) but doesn't require knowing the split -- which is
why it can cover every method (sieve/table) for every algorithm, including
the ones with no published component breakdown, using only their
speedup_total tables.

combine_ms is ASSUMED algorithm- and method-independent at fixed modulus
size and rank count -- it's just "rank 0 multiplies N gathered partial cbrt
products," the same operation regardless of which algorithm/method produced
those products (they're all ~modulus-sized integers either way; sieve vs
table changes factor_ms, not what gets reconstructed or how). This is a
mechanistic justification, not an independent empirical cross-check: only
k16 has a directly-published combine_ms breakdown, so the same c fitted
from it is reused for every other row. Treat every non-k16-sieve row as
resting on this one assumption.

CAVEAT
------
This is a projection, not a measurement -- the log-depth combine has not been
run on the cluster yet. It also assumes the log-depth tree adds no
per-round overhead beyond the raw multiplication cost c (a generous, lower-
bound assumption -- a real MPI-based tree reduce would add some communication
latency per round, though that should be small given Bcast/Gather/Barrier are
already <0.1ms each at this scale).

HOW TO REFRESH WHEN NEW DATA LANDS
-----------------------------------
Replace the SPEEDUP_TOTAL tuples below with fresh measured (ranks,
speedup_total) pairs and re-run.
"""

import math

POWM_BASELINE_MS = 1801.0  # 24576-bit baseline, consistent across all algorithms/methods here

# k16's own directly-measured combine_ms, used ONLY to fit c (per-mulmod cost).
# Source: wiki/results/speedup-scaling.md "Why k16 Peaks at N16" (24576 bits).
K16_COMBINE_FIT_POINTS = [
    # (ranks, combine_ms)
    (2, 0.11), (8, 0.71), (32, 3.13), (64, 6.38), (128, 12.80),
]

# (ranks, speedup_total) at 24576 bits, every benchmarked algorithm and method.
# ranks = 2,4,8,16,32,64,128 correspond to N1,N2,N4,N8,N16,N32,N64.
# Sources: wiki/results/alg2-full-scaling.md (k16/k32/k35 sieve+table),
#          wiki/results/alg3-full-scaling.md (Algo3, sieve-only -- no table variant exists),
#          wiki/results/performance-results.md + alg4-post-fix-scaling.md (Algo4 single-prime),
#          wiki/algorithms/algorithm4.md "alg4_multi Confirmed Scaling Results" (Algo4 composite;
#          N2/4-rank point not published there, so that row is omitted, not interpolated).
SPEEDUP_TOTAL = {
    "k16_sieve":    [(2, 21.58), (4, 42.30), (8, 80.10), (16, 138.60), (32, 189.88), (64, 174.30), (128, 116.91)],
    "k16_table":    [(2, 21.50), (4, 43.29), (8, 80.19), (16, 139.25), (32, 189.99), (64, 174.38), (128, 116.75)],
    "k32_sieve":    [(2, 1.77), (4, 2.93), (8, 5.38), (16, 9.71), (32, 17.60), (64, 30.73), (128, 43.20)],
    "k32_table":    [(2, 2.38), (4, 4.05), (8, 7.30), (16, 13.52), (32, 23.24), (64, 39.74), (128, 52.38)],
    "k35_sieve":    [(2, 0.24), (4, 0.41), (8, 0.77), (16, 1.40), (32, 2.69), (64, 5.11), (128, 8.87)],
    "k35_table":    [(2, 0.27), (4, 0.52), (8, 1.04), (16, 1.96), (32, 3.75), (64, 6.91), (128, 12.17)],
    "alg3":         [(2, 1.74), (4, 2.94), (8, 5.30), (16, 9.72), (32, 17.64), (64, 30.27), (128, 42.88)],
    "alg4_single":  [(2, 1.66), (4, 2.83), (8, 5.12), (16, 9.30), (32, 17.33), (64, 29.15), (128, 40.73)],
    "alg4_multi":   [(2, 1.64), (8, 5.09), (16, 9.25), (32, 16.96), (64, 29.06), (128, 41.00)],  # 4-rank point not published
}


def fit_c(points):
    """Fit combine_ms = c * (ranks - 1) via least squares through the origin."""
    num = sum(r * (ranks - 1) for ranks, r in points)
    den = sum((ranks - 1) ** 2 for ranks, _ in points)
    return num / den


def combine_serial(c, ranks):
    return c * (ranks - 1)


def combine_ideal(c, ranks):
    return c * math.ceil(math.log2(ranks))


def project(name, c, points):
    print(f"\n=== {name} ===")
    print(f"{'ranks':>6} | {'total_meas_ms':>13} | {'total_ideal_ms':>14} | "
          f"{'speedup_meas':>12} | {'speedup_ideal':>13} | gain")
    print("-" * 76)
    for ranks, speedup_meas in points:
        total_meas = POWM_BASELINE_MS / speedup_meas
        c_serial = combine_serial(c, ranks)
        c_ideal = combine_ideal(c, ranks)
        total_id = total_meas - c_serial + c_ideal
        speedup_id = POWM_BASELINE_MS / total_id
        gain = (speedup_id / speedup_meas - 1) * 100
        print(f"{ranks:6d} | {total_meas:13.2f} | {total_id:14.2f} | "
              f"{speedup_meas:12.2f} | {speedup_id:13.2f} | {gain:+.1f}%")


def main():
    c = fit_c(K16_COMBINE_FIT_POINTS)
    print(f"Fitted combine cost: c = {c:.4f} ms per mulmod (24576-bit modulus)")
    print("Fit quality check (k16 sieve's own points, should reproduce exactly):")
    for ranks, combine_ms in K16_COMBINE_FIT_POINTS:
        predicted = combine_serial(c, ranks)
        print(f"  ranks={ranks:4d}  measured={combine_ms:6.3f}ms  "
              f"fitted={predicted:6.3f}ms  diff={predicted - combine_ms:+.4f}ms")

    print("\nNote: c is reused as-is for every row below except k16_sieve on the mechanistic "
          "assumption that combine_ms doesn't depend on algorithm or method (sieve/table) -- "
          "this is NOT independently cross-validated for any of them, see the module docstring.")

    for name, points in SPEEDUP_TOTAL.items():
        project(name, c, points)

    print("\n=== Summary at N64 (128 ranks) ===")
    print(f"{'variant':>14} | {'measured':>9} | {'ideal':>9} | gain")
    print("-" * 48)
    for name, points in SPEEDUP_TOTAL.items():
        ranks_128 = [s for r, s in points if r == 128]
        if not ranks_128:
            continue
        speedup_meas = ranks_128[0]
        total_meas = POWM_BASELINE_MS / speedup_meas
        total_id = total_meas - combine_serial(c, 128) + combine_ideal(c, 128)
        speedup_id = POWM_BASELINE_MS / total_id
        gain = (speedup_id / speedup_meas - 1) * 100
        print(f"{name:>14} | {speedup_meas:9.2f} | {speedup_id:9.2f} | {gain:+.1f}%")

    print("\nAll 'ideal' values come from a model of the log-depth combine tree, not from "
          "cluster runs of it -- this is a projection, not a measurement. See "
          "wiki/results/ideal-overhead-free-projection.md.")


if __name__ == "__main__":
    main()
