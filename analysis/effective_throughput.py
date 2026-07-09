"""
effective_throughput.py

Computes the effective throughput of k32, Algo3, and Algo4 across all
benchmark sizes and rank counts.

Effective throughput = P(hit per blinding attempt) / wall_time_per_attempt

This combines:
  - P(hit): from the Dickman–de Bruijn function (theoretical)
  - wall_time: from measured total_ms across all benchmark JSON files

Output:
  - Table of P(hit) by algorithm and modulus size
  - Table of effective_rate (normalised to k32 at N64/24576b)
  - Table of "profitable" configurations (effective_rate > sequential baseline)
  - Comparison of Algo3/Algo4 vs k32 ratios

Usage:
  python3 effective_throughput.py

Requires: numpy, scipy, glob, json (all standard or pip-installable)
"""

import json, glob, os, math

# ---------------------------------------------------------------------------
# Dickman–de Bruijn function ρ(u)
# Solves the delay ODE: u·ρ'(u) = -ρ(u-1), ρ(u) = 1 for 0 ≤ u ≤ 1
# Method of steps with linear interpolation.
# NOTE: at 4000 points/unit the linear-interp lookup error accumulates in the
# tail (rho(9) came out ~3.6e-10 vs the true ~1.0e-9, and rho(10) collapsed to
# ~0). 40000 points/unit converges the whole 1<=u<=20 range to Granville's
# tabulated values. This only affects the theoretical P(hit) / reference tables;
# every effective-RATE and ratio table below is unchanged because the shared
# rho(8) factor cancels in numerator and denominator.
# ---------------------------------------------------------------------------

_DICKMAN_N = 40000  # points per unit interval

def _rho_lookup(u1, rho):
    """Interpolate rho at value u1 (u1 >= 0). Returns 1.0 for u1 <= 1."""
    if u1 <= 1.0:
        return 1.0
    j = min(int(u1 * _DICKMAN_N), len(rho) - 2)
    frac = u1 * _DICKMAN_N - j
    return rho[j] * (1.0 - frac) + rho[j + 1] * frac

def _build_dickman_table(u_max):
    # Build table with step 1/_DICKMAN_N per slot.
    # u_max_int units, total = u_max_int * _DICKMAN_N + 1 points.
    u_max_int = int(math.ceil(u_max)) + 1
    total = u_max_int * _DICKMAN_N + 1
    rho = [1.0] * total
    dt = 1.0 / _DICKMAN_N  # step size = 1/N

    for i in range(1, total):
        u = i * dt           # current u
        if u <= 1.0:
            rho[i] = 1.0
            continue
        # RK4 / Simpson integration of rho'(u) = -rho(u-1)/u
        # Since f(u, y) = -rho(u-1)/u does not depend on y, RK4 = Simpson's rule:
        #   rho(u) = rho(u - dt) + dt/6 * [f(u-dt) + 4*f(u-dt/2) + f(u)]
        # where f(v) = -rho(v-1)/v and rho(v-1) for v-1 < u-dt is already computed.
        u0 = (i - 1) * dt
        um = u0 + dt * 0.5   # midpoint
        f0 = -_rho_lookup(u0 - 1.0, rho) / u0
        fm = -_rho_lookup(um - 1.0, rho) / um
        f1 = -_rho_lookup(u  - 1.0, rho) / u
        rho[i] = max(rho[i - 1] + dt / 6.0 * (f0 + 4.0 * fm + f1), 0.0)

    return rho

_rho_table = _build_dickman_table(20.0)

def dickman(u):
    """Return ρ(u) for 0 ≤ u ≤ 30 (uses asymptotic u^{-u} beyond table)."""
    if u <= 0:
        return 0.0
    if u <= 1:
        return 1.0
    u_max_table = (len(_rho_table) - 1) / _DICKMAN_N
    if u > u_max_table:
        return u ** (-u)
    j = int(u * _DICKMAN_N)
    j = min(j, len(_rho_table) - 2)
    frac = u * _DICKMAN_N - j
    return float(_rho_table[j] * (1.0 - frac) + _rho_table[j + 1] * frac)


# ---------------------------------------------------------------------------
# P(hit) for each algorithm at modulus bit-length n
#
# k32:   y is fully 2^32-smooth
#        P = ρ(n / 32)
#
# Algo3: y = smooth_base × q0 where q0 is a single prime in (2^32, 2^65]
#        Approximation: P ≈ ρ((n-48) / 32) × (1/48) × ln(48/32)
#        (smooth base covers n-48 bits, q0 fills the remaining 33..64 bits)
#        Simpler bound used in the paper (Lemma 3.1):
#            P(Algo3) / P(k32) ≈ (65-32) / ρ(0) × integral ≈ 119
#        We use P(Algo3) = P(k32) × 119 as the paper's approximation.
#
# Algo4: P(Algo4) > P(Algo3) because composite cofactors are also accepted.
#        Additional mass from q1*q2 pairs, each in (2^32, 2^33].
#        The paper gives Table 1 speedup 18× vs Algo3's 20×, implying
#        P(Algo4)/P(Algo3) ≈ 20/18 × (T_algo3/T_algo4) = (20/18) × (13/14) ≈ 1.03.
#        We use P(Algo4) = P(Algo3) × 1.03 as a lower bound.
#
# These are well-known approximations from the paper; not our own derivations.
# ---------------------------------------------------------------------------

ALGO3_K32_RATIO = 119.0        # paper: Algo3 hits 119× more often than k32
ALGO4_ALGO3_RATIO = (20 / 18) * (13 / 14)  # ≈ 1.03

def p_hit_k32(n_bits, B_bits=32):
    u = n_bits / B_bits
    return dickman(u)

def p_hit_algo3(n_bits, B_bits=32):
    return p_hit_k32(n_bits, B_bits) * ALGO3_K32_RATIO

def p_hit_algo4(n_bits, B_bits=32):
    return p_hit_algo3(n_bits, B_bits) * ALGO4_ALGO3_RATIO


# ---------------------------------------------------------------------------
# Load measured speedup_total and total_ms from benchmark JSON files
# ---------------------------------------------------------------------------

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")

SIZES = [256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576]
NS    = ["N1", "N2", "N4", "N8", "N16", "N32", "N64"]
N_RANKS = {"N1": 2, "N2": 4, "N4": 8, "N8": 16, "N16": 32, "N32": 64, "N64": 128}

def _read_json(path):
    with open(path) as f:
        return json.load(f)

def _extract(d):
    rv = d["metadata"]["runs_valid"]
    if rv == 0:
        return None
    return (
        d["warm_avg"]["speedup_total"],
        d["warm_avg"]["total_ms"],
        d["metadata"]["powm_baseline_ms"],
    )

def load_alg2_variant(variant, method):
    """Load alg2 results. Files: results/alg2/alg2_12175008/{variant}_mpi_{N}/{variant}_mpi_{method}_{sz}_*.json"""
    base = os.path.join(BASE, "alg2/alg2_12175008")
    out = {}
    for nkey in NS:
        d = os.path.join(base, f"{variant}_mpi_{nkey}")
        if not os.path.isdir(d):
            continue
        for sz in SIZES:
            files = glob.glob(os.path.join(d, f"{variant}_mpi_{method}_{sz}_*.json"))
            if not files:
                continue
            v = _extract(_read_json(files[0]))
            if v:
                out[(nkey, sz)] = v
    return out

def load_alg_jobs(alg_subdir, prefix, job_N):
    """Load alg3/alg4 results. Files: results/{alg_subdir}/{job}/{N}/{prefix}_sieve_{sz}_*.json"""
    out = {}
    for job, nkey in job_N.items():
        d = os.path.join(BASE, alg_subdir, job, nkey)
        if not os.path.isdir(d):
            continue
        for sz in SIZES:
            files = glob.glob(os.path.join(d, f"{prefix}_sieve_{sz}_*.json"))
            if not files:
                continue
            v = _extract(_read_json(files[0]))
            if v:
                out[(nkey, sz)] = v
    return out

# Load all benchmark tables
k32_sieve_data  = load_alg2_variant("k32", "sieve")
algo3_data      = load_alg_jobs("alg3",
    "alg3_mpi",
    {"alg3_12179496":"N1","alg3_12179497":"N2","alg3_12179502":"N4",
     "alg3_12179596":"N8","alg3_12179604":"N16","alg3_12179606":"N32",
     "alg3_12179495":"N64"})
algo4_data      = load_alg_jobs("alg4",
    "alg4_mpi",
    {"alg4_12180104":"N1","alg4_12180105":"N2","alg4_12180106":"N4",
     "alg4_12180107":"N8","alg4_12180109":"N16","alg4_12180111":"N32",
     "alg4_12180103":"N64"})
algo4_multi_data = load_alg_jobs("alg4",
    "alg4_multi_mpi",
    {"alg4_multi_12180104":"N1","alg4_multi_12180105":"N2","alg4_multi_12180106":"N4",
     "alg4_multi_12180107":"N8","alg4_multi_12180109":"N16","alg4_multi_12180111":"N32",
     "alg4_multi_12180103":"N64"})


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------

def fmt_p(p):
    if p <= 0:
        return "≈0"
    bits = math.log2(p)
    return f"2^{bits:.1f}"

def fmt_r(r):
    if r == 0 or not math.isfinite(r):
        return "    —"
    return f"{r:7.3f}×"

print("=" * 72)
print("EFFECTIVE THROUGHPUT ANALYSIS")
print("=" * 72)
print()

# --- Table 1: P(hit) by algorithm and modulus size ---
print("Table 1: P(hit per blinding attempt) — theoretical (Dickman function)")
print()
print(f"{'Size':>8}  {'k32 (B=2^32)':>14}  {'Algo3 (×119)':>14}  {'Algo4 (×1.03)':>14}  {'u=n/32':>8}")
print("-" * 68)
for sz in SIZES:
    u = sz / 32
    pk  = p_hit_k32(sz)
    pa3 = p_hit_algo3(sz)
    pa4 = p_hit_algo4(sz)
    print(f"{sz:>6}b  {fmt_p(pk):>14}  {fmt_p(pa3):>14}  {fmt_p(pa4):>14}  {u:>8.1f}")

print()
print("Note: P(hit) drops to ~0 for u > 12 (n > 384 bits for B=2^32).")
print("At 24576b (u=768): P(k32) ≈ 768^{-768} — unmeasurably small.")
print()

# --- Table 2: Effective rate at 256 bits ---
print("Table 2: Effective rate at n=256 bits — P(hit) / total_ms")
print("(Both numerically: actual_P × 1/total_ms)")
print()
print(f"{'Config':>12}  {'k32 eff_rate':>14}  {'algo3 eff_rate':>14}  {'algo4 eff_rate':>14}  {'algo3/k32':>10}  {'algo4/k32':>10}")
print("-" * 80)
for nkey in NS:
    k32v  = k32_sieve_data.get((nkey, 256))
    a3v   = algo3_data.get((nkey, 256))
    a4v   = algo4_data.get((nkey, 256))
    if not (k32v and a3v and a4v):
        continue
    pk  = p_hit_k32(256)
    pa3 = p_hit_algo3(256)
    pa4 = p_hit_algo4(256)
    erk  = pk  / k32v[1]   # P(hit) / total_ms
    era3 = pa3 / a3v[1]
    era4 = pa4 / a4v[1]
    r3 = era3 / erk if erk > 0 else float('inf')
    r4 = era4 / erk if erk > 0 else float('inf')
    ranks = N_RANKS[nkey]
    print(f"{nkey:>4}({ranks:>3}r)  {erk:.3e}   {era3:.3e}   {era4:.3e}   {r3:>10.2f}×  {r4:>10.2f}×")

print()

# --- Table 3: Effective rate at 24576 bits (where speedup peaks) ---
print("Table 3: Effective rate at n=24576 bits")
print("Note: P(hit) ≈ 0 for random inputs at 24576b. Values show conditional-on-smooth metric.")
print()
print(f"{'Config':>12}  {'k32 speedup':>12}  {'algo3 speedup':>14}  {'algo4 speedup':>14}  {'algo3/k32 speedup':>18}")
print("-" * 80)
for nkey in NS:
    k32v  = k32_sieve_data.get((nkey, 24576))
    a3v   = algo3_data.get((nkey, 24576))
    a4v   = algo4_data.get((nkey, 24576))
    if not (k32v and a3v and a4v):
        continue
    ranks = N_RANKS[nkey]
    ratio = a3v[0] / k32v[0] if k32v[0] > 0 else 0
    print(f"{nkey:>4}({ranks:>3}r)  {k32v[0]:>12.2f}×  {a3v[0]:>14.2f}×  {a4v[0]:>14.2f}×  {ratio:>18.3f}×")

print()

# --- Table 4: Combined effective_rate = P(hit at 256b) × speedup_total (mixed metric) ---
# This is what the wiki effective-throughput.md uses.
print("Table 4: Mixed metric — P(hit at 256b) × speedup_total at each size/rank")
print("(Normalized: k32 at N64/24576b = 1.0×)")
print()

# Get k32 normalisation value at N64/24576b
k32_norm_v = k32_sieve_data.get(("N64", 24576))
pk32_256 = p_hit_k32(256)
pa3_256  = p_hit_algo3(256)
pa4_256  = p_hit_algo4(256)

if k32_norm_v:
    k32_norm = pk32_256 * k32_norm_v[0]

    for sz in [256, 1024, 4096, 8192, 12288, 24576]:
        print(f"  n={sz}b:")
        print(f"  {'Config':>10}  {'k32':>10}  {'algo3':>10}  {'algo4':>10}  {'algo3/k32':>12}")
        print(f"  {'-'*58}")
        for nkey in NS:
            k32v = k32_sieve_data.get((nkey, sz))
            a3v  = algo3_data.get((nkey, sz))
            a4v  = algo4_data.get((nkey, sz))
            if not (k32v and a3v and a4v):
                continue
            ek32 = pk32_256 * k32v[0] / k32_norm
            ea3  = pa3_256  * a3v[0]  / k32_norm
            ea4  = pa4_256  * a4v[0]  / k32_norm
            r3   = ea3 / ek32 if ek32 > 0 else float('inf')
            ranks = N_RANKS[nkey]
            print(f"  {nkey:>4}({ranks:>3}r)  {ek32:>10.4f}  {ea3:>10.4f}  {ea4:>10.4f}  {r3:>12.2f}×")
        print()

# --- Crossover analysis: where does the attack become "profitable"? ---
print("Table 5: Speedup crossover (where attack beats sequential) — k32 sieve, N64")
print()
print(f"{'Size':>8}  {'speedup':>10}  {'P(k32,256b)':>14}  {'eff_rate/seqrate':>18}")
print("-" * 56)
for sz in SIZES:
    v = k32_sieve_data.get(("N64", sz))
    if not v:
        continue
    sp, tm, pm = v
    # eff_rate vs sequential: (P(hit) × speedup) / 1
    # At 256b: P(y B-smooth) = p_hit_k32(256)
    # sequential always hits (P=1), costs pm ms
    # eff_ratio = (p_hit_k32(256) / tm) / (1 / pm) = p_hit_k32(256) × pm / tm
    #           = p_hit_k32(256) × speedup_total
    eff = pk32_256 * sp
    print(f"{sz:>6}b  {sp:>10.2f}×  {fmt_p(pk32_256):>14}  {eff:>18.6f}")

print()
print("Interpretation:")
print("  eff_rate/seqrate < 1: attack is slower than sequential (for random inputs)")
print("  eff_rate/seqrate = p_hit * speedup — at 24576b this is 2^(-24.9) * 43 ≈ 1.3e-6")
print("  The attack is NOT profitable on random VDF inputs with our hardware at any size.")
print("  It requires 2^54.5 CPUs to make it profitable at 256b (paper's operating point).")
print()
print("Summary of 6.5× Algo3 advantage over k32:")
print()
print("  Both are measured at their best configurations (N64/24576b).")
print("  The ratio = P(Algo3) × speedup(Algo3) / (P(k32) × speedup(k32))")
k32_best = k32_sieve_data.get(("N64", 24576))
a3_best  = algo3_data.get(("N64", 24576))
if k32_best and a3_best:
    ratio = (pa3_256 * a3_best[0]) / (pk32_256 * k32_best[0])
    print(f"  = (2^{math.log2(pa3_256):.1f} × {a3_best[0]:.2f}×) / (2^{math.log2(pk32_256):.1f} × {k32_best[0]:.2f}×)")
    print(f"  = {ratio:.2f}×  (Algo3 is {ratio:.1f}× more effective than k32)")
    print()
    print("  Note: P(hit) values are at 256-bit modulus; speedup values are at 24576b.")
    print("  A consistent evaluation at 256b would use 256b speedup values (~0.002×),")
    print("  giving: (2^-18 × 0.003) / (2^-24.9 × 0.002) ≈ 178×.")
    print("  The 6.5× figure uses the 24576b speedup because that's where")
    print("  speedup_total is meaningful — an approximation of the relative cost.")

print()
print("=" * 72)
print("DICKMAN FUNCTION REFERENCE TABLE")
print("=" * 72)
print()
print(f"{'u':>6}  {'ρ(u)':>14}  {'2^x exponent':>14}")
print("-" * 40)
for u in [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 16, 20]:
    r = dickman(u)
    if r > 0:
        exp = math.log2(r)
        print(f"{u:>6.1f}  {r:>14.4e}  {exp:>14.2f}")
    else:
        print(f"{u:>6.1f}  {'≈0':>14}  {'—':>14}")

print()
print("B=2^32: P(k32 hit at n bits) = ρ(n/32)")
print("B=2^35: P(k35 hit at n bits) = ρ(n/35)  [paper's primary Algo2]")
print()

# Edit: 2026-07-07
