#!/usr/bin/env python3
"""
cache_residency.py — how far can the smoothness bound rise before the prime
table stops fitting in GPU cache?

WHY THIS EXISTS
---------------
The thesis explains the GPU results with one principle: residency decides
everything. k16's 26 KB table sits in L1 and a single V100 runs its sieve step
800x faster than sequential; k32's 813 MB and k35's 9.6 GB miss every level of
cache and lose to the CPU cluster. Three variants, three regimes -- but they are
three widely-spaced samples of a continuum, and the boundary between them has
never been located.

This script locates it. For every k between the two measured endpoints it
reports the table size and asks whether a slice fits in each device's cache at
1, 2 and 4 GPUs. The supervisor's suggested k=20 is one row of the sweep.

WHY THE ANSWER MATTERS
----------------------
Residency and hit rate pull in opposite directions. Raising k enlarges the
prime set, which raises the smoothness probability rho(|M|/k) and so makes more
challenges attackable -- but it also grows the table, and past some k the table
leaves cache and the sieve becomes memory-bound. The best variant is the largest
k that is still resident, and nobody has computed which k that is.

PRIME COUNTS
------------
pi(2^k) is exact (OEIS A007053), not estimated -- sieving 2^31 to check would
take longer than the question deserves. The k32 row is a consistency gate: this
table's pi(2^32) - pi(2^16) must reproduce the 203,273,679 primes the solvers
actually enumerate, and it does.

PREDICTION vs MEASUREMENT
-------------------------
Sizes and residency are arithmetic and need no cluster. Whether residency makes
the sieve FASTER does not follow from arithmetic and is the part worth measuring:
sub6:delta already showed that making k32's table L2-resident by delta encoding
bought only 1.10x, so residency is necessary for the k16-style collapse but not
sufficient. Pass --measured to overlay real timings from experiments/ksweep/ksweep.slurm
and see whether the predicted boundary is visible in ns-per-prime.

Usage:
    python3 cache_residency.py                # analytic sweep only
    python3 cache_residency.py --delta        # also show delta-encoded sizes
    python3 cache_residency.py --measured results/results/iris/ksweep/G4
"""
import argparse, glob, json, os, re

# pi(2^k), exact -- OEIS A007053.
PI_2K = {
    16: 6542,          17: 12251,         18: 23000,        19: 43390,
    20: 82025,         21: 155611,        22: 295947,       23: 564163,
    24: 1077871,       25: 2063689,       26: 3957809,      27: 7603553,
    28: 14630843,      29: 28192750,      30: 54400028,     31: 105097565,
    32: 203280221,
}

# Device cache capacities (MB). L1 is per-SM, L2 is device-wide and shared.
DEVICES = [
    ("V100 L1/SM", 0.096),
    ("V100 L2",    6.0),
    ("L40 L2",     96.0),
]

# The solvers sieve (2^16, 2^k]: primes below 65537 are served by the dense k16
# cbrt table instead (Gap 8), so they are not in the streamed table.
LOWER = 16


def table_bytes(k, per_prime=4):
    """Bytes for the streamed prime table over (2^16, 2^k]."""
    return (PI_2K[k] - PI_2K[LOWER]) * per_prime


def delta_bytes(k, block=1024):
    """Byte-per-prime delta encoding: one full-width anchor per block of 1024,
    then one half-gap byte each. Matches the measured 3.99x for k32."""
    n = PI_2K[k] - PI_2K[LOWER]
    blocks = (n + block - 1) // block
    return blocks * 4 + n  # anchor per block + 1 byte per prime


def load_measured(d):
    """Read ksweep JSONs from `d`, keyed (k, n_gpus).

    Returns {(k,g): {factor_ms, recon_ms, total_ms, speedup, ns_per_prime}}.
    ns_per_prime is the figure the residency question actually turns on: it
    normalises out the table SIZE, so a flat series means the per-prime cost does
    not care where the table lives, and a step at the predicted boundary means it
    does."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        md = j.get("metadata", {})
        if md.get("algorithm") != "ksweep":
            continue                      # never mix in Algorithm 2 results
        k = md.get("smoothness_bound_bits")
        g = md.get("n_gpus", 1)
        w = j.get("warm_avg", {})
        fm, rm = w.get("factor_ms"), w.get("recon_ms")
        if not k or not fm:
            continue
        n = PI_2K.get(k, 0) - PI_2K[LOWER]
        out[(k, g)] = dict(
            factor_ms=fm, recon_ms=rm or 0.0, total_ms=fm + (rm or 0.0),
            baseline=md.get("powm_baseline_ms", 0.0),
            ns_per_prime=(fm * 1e6 / n) if n else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bytes", type=int, default=4, help="bytes per prime (default 4)")
    ap.add_argument("--delta", action="store_true", help="also show delta-encoded sizes")
    ap.add_argument("--measured", metavar="DIR",
                    help="overlay real timings from a ksweep results directory")
    a = ap.parse_args()

    print("# Prime table size and GPU cache residency vs smoothness bound k")
    print(f"# table spans (2^{LOWER}, 2^k];  primes below 2^{LOWER} use the dense k16 cbrt table")
    print(f"# {a.bytes} bytes per prime"
          + ("  |  delta = 1024-prime blocks, 4-byte anchor + 1 byte/prime" if a.delta else ""))
    print()

    cols = f"{'k':>3} {'primes':>12} {'table':>11}"
    if a.delta:
        cols += f" {'delta':>10} {'ratio':>6}"
    cols += "  | " + "  ".join(f"{d:>10}" for d, _ in DEVICES)
    print(cols)
    print("-" * len(cols))

    # k16 is the special case and sits outside the sweep: its solver scans ALL
    # 6,542 primes <= 2^16 directly rather than a range above 2^16, so it has no
    # "streamed table" in the sense the k>16 rows measure. It is the reference
    # point the sweep runs away from.
    k16_mb = PI_2K[LOWER] * a.bytes / 1e6
    print(f"{LOWER:3d} {PI_2K[LOWER]:12,d} {k16_mb:9.3f}MB"
          + (f" {'—':>8}   {'—':>5} " if a.delta else "")
          + "  | " + "  ".join("G1".rjust(10) for _ in DEVICES)
          + "   <- k16 (measured: wins on GPU; fits even L1)")

    for k in range(LOWER + 1, 33):
        if k not in PI_2K:
            continue
        n = PI_2K[k] - PI_2K[LOWER]
        b = table_bytes(k, a.bytes)
        mb = b / 1e6
        row = f"{k:3d} {n:12,d} {mb:9.2f}MB"
        if a.delta:
            db = delta_bytes(k)
            row += f" {db/1e6:8.2f}MB {b/db:5.2f}x"
        row += "  | "
        cells = []
        for dname, cap in DEVICES:
            # largest GPU count at which a 1/g slice fits
            fits = [g for g in (1, 2, 4) if mb / g <= cap]
            cells.append(("G" + str(min(fits)) if fits else "—").rjust(10))
        row += "  ".join(cells)
        # mark the measured variants
        if k == 20:
            row += "   <- k20 (supervisor's suggestion)"
        elif k == 32:
            row += "   <- k32 (measured: loses to CPU cluster)"
        print(row)

    print()
    print("Cell = smallest GPU count whose 1/g slice fits that cache; '—' = never fits at <=4 GPUs.")
    print()

    # the headline: largest resident k per device, at 1 and 4 GPUs
    print("Largest k whose slice still fits:")
    for dname, cap in DEVICES:
        for g in (1, 4):
            best = max((k for k in PI_2K if k > LOWER
                        and table_bytes(k, a.bytes) / 1e6 / g <= cap), default=None)
            dbest = max((k for k in PI_2K if k > LOWER
                         and delta_bytes(k) / 1e6 / g <= cap), default=None)
            if best is not None:
                mb = table_bytes(best, a.bytes) / 1e6 / g
                extra = ""
                if dbest is not None and dbest > best:
                    extra = (f"   (delta-encoded: k = {dbest}, "
                             f"{delta_bytes(dbest)/1e6/g:.1f} MB/GPU)")
                print(f"  {dname:11} at {g} GPU{'s' if g>1 else ' '}: "
                      f"k = {best:2d}  ({mb:.2f} MB/GPU of {cap:g} MB){extra}")
    print()
    print("Reading it: the largest resident k is the best candidate variant, because raising k")
    print("raises the smoothness hit rate rho(|M|/k) while residency holds. Past that k the sieve")
    print("becomes memory-bound and the k32 outcome follows. What this does NOT settle is whether")
    print("residency alone buys the k16-style collapse -- it does not (sub6:delta): k16 also wins")
    print("because 6,542 primes is barely any work. Residency is necessary, not sufficient.")

    if not a.measured:
        print()
        print("Everything above is arithmetic. Run experiments/ksweep/ksweep.slurm and re-run with")
        print("  --measured results/results/iris/ksweep/G4   to test it against real timings.")
        return

    meas = load_measured(a.measured)
    if not meas:
        print()
        print(f"No ksweep JSONs found under {a.measured}.")
        return

    print()
    print("=" * 78)
    print("MEASURED — does the predicted boundary show up in the timings?")
    print("=" * 78)
    for g in sorted({k[1] for k in meas}):
        rows = sorted((k, v) for (k, gg), v in meas.items() if gg == g)
        if not rows:
            continue
        cap = max(c for _, c in DEVICES)   # device L2, the boundary under test
        print(f"\n{g} GPU(s):")
        print(f"  {'k':>3} {'MB/GPU':>9} {'resident':>9} {'factor ms':>11} "
              f"{'ns/prime':>10} {'vs prev':>9} {'speedup':>9}")
        prev = None
        for k, v in rows:
            mb = table_bytes(k, a.bytes) / 1e6 / g
            res = "yes" if mb <= cap else "NO"
            step = f"{v['ns_per_prime']/prev:8.2f}x" if prev else "        -"
            sp = (v["baseline"] / v["total_ms"]) if v["total_ms"] else 0.0
            print(f"  {k:3d} {mb:9.1f} {res:>9} {v['factor_ms']:11.2f} "
                  f"{v['ns_per_prime']:10.3f} {step} {sp:8.2f}x")
            prev = v["ns_per_prime"]

        # The verdict: is there a discontinuity where residency is lost?
        ks = [k for k, _ in rows]
        boundary = None
        for i in range(1, len(ks)):
            a_mb = table_bytes(ks[i-1], a.bytes) / 1e6 / g
            b_mb = table_bytes(ks[i],   a.bytes) / 1e6 / g
            if a_mb <= cap < b_mb:
                boundary = (ks[i-1], ks[i])
        if boundary:
            lo, hi = boundary
            r_lo = dict(rows)[lo]["ns_per_prime"]
            r_hi = dict(rows)[hi]["ns_per_prime"]
            jump = r_hi / r_lo if r_lo else 0.0
            others = [dict(rows)[ks[i]]["ns_per_prime"] / dict(rows)[ks[i-1]]["ns_per_prime"]
                      for i in range(1, len(ks))
                      if not (ks[i-1] == lo and ks[i] == hi)
                      and dict(rows)[ks[i-1]]["ns_per_prime"]]
            typical = sum(others) / len(others) if others else 1.0
            print(f"\n  Residency is lost between k{lo} and k{hi} "
                  f"({table_bytes(lo,a.bytes)/1e6/g:.0f} -> "
                  f"{table_bytes(hi,a.bytes)/1e6/g:.0f} MB/GPU against a {cap:g} MB L2).")
            print(f"  ns/prime there: {jump:.2f}x, against {typical:.2f}x "
                  f"across the other steps.")
            if jump > 1.5 * typical:
                print("  -> A STEP. Residency is a real discontinuity, not a smooth trend;")
                print("     the k16-vs-k32 contrast is at least partly a cache effect.")
            else:
                print("  -> NO step. The per-prime cost does not care where the table lives,")
                print("     so k16's win is about having little WORK, not about residency.")
                print("     That agrees with sub6:delta: L2 residency bought k32 only 1.10x.")


if __name__ == "__main__":
    main()
