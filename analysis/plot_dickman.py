#!/usr/bin/env python3
"""Plot the Dickman-de Bruijn function rho(u) for THESIS.md figure 2.4.

Reuses the same RK4/Simpson integration of rho'(u) = -rho(u-1)/u as
raw-sources/minroot_tp/analysis/effective_throughput.py, so the curve
matches the rho(u) values tabulated in section 2.4 exactly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Points per unit interval. effective_throughput.py uses 4000, which is accurate
# to ~u=6 but understates the tail (linear-interp lookup error accumulates: at
# 4000, rho(9) comes out ~3.6e-10 vs the true ~1.0e-9). 40000 converges the whole
# 1<=u<=10 range to Granville's tabulated values, which the thesis figure needs.
_DICKMAN_N = 40000


def _rho_lookup(u1, rho):
    if u1 <= 1.0:
        return 1.0
    j = min(int(u1 * _DICKMAN_N), len(rho) - 2)
    frac = u1 * _DICKMAN_N - j
    return rho[j] * (1.0 - frac) + rho[j + 1] * frac


def build_dickman_table(u_max):
    u_max_int = int(u_max)
    total = u_max_int * _DICKMAN_N + 1
    rho = [1.0] * total
    dt = 1.0 / _DICKMAN_N
    for i in range(1, total):
        u = i * dt
        if u <= 1.0:
            rho[i] = 1.0
            continue
        u0 = u - dt
        um = u - dt / 2.0
        f0 = -_rho_lookup(u0 - 1.0, rho) / u0
        fm = -_rho_lookup(um - 1.0, rho) / um
        f1 = -_rho_lookup(u - 1.0, rho) / u
        rho[i] = max(rho[i - 1] + dt / 6.0 * (f0 + 4.0 * fm + f1), 0.0)
    return rho


U_MAX = 10.0
rho = build_dickman_table(U_MAX)
us = [i / _DICKMAN_N for i in range(len(rho))]


def rho_at(u):
    j = min(int(u * _DICKMAN_N), len(rho) - 2)
    frac = u * _DICKMAN_N - j
    return rho[j] * (1.0 - frac) + rho[j + 1] * frac


fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.semilogy(us, rho, color="#1f4e79", lw=2)

# Mark the k32 / 256-bit operating point u = 8, rho(8) ~ 2^-24.9
u_op = 8.0
r_op = rho_at(u_op)
ax.scatter([u_op], [r_op], color="#c00000", zorder=6, s=45,
           edgecolor="white", linewidth=0.8)
# Label placed in the empty lower-left band (well below the curve) so the red
# text never sits on the plotted line; a white box keeps it readable over the
# grid, and the arrow points to the marked operating point at u = 8.
ax.annotate(
    r"$u=8,\ \rho(8)\approx 2^{-24.9}$" "\n(256-bit, $B=2^{32}$)",
    xy=(u_op, r_op), xytext=(1.7, 3e-9),
    fontsize=9, color="#c00000", ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c00000", alpha=0.95),
    arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.3,
                    connectionstyle="arc3,rad=-0.15"),
    zorder=7,
)

ax.set_xlabel(r"$u = \log x / \log B \;=\; |x| / |B|$")
ax.set_ylabel(r"$\rho(u)$  (probability of $B$-smoothness)")
ax.set_title(r"Dickman--de Bruijn smoothness probability $\rho(u)$")
ax.set_xlim(1, U_MAX)
ax.set_ylim(1e-10, 2)
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(f"dickman_rho.{ext}", dpi=200, bbox_inches="tight")
print("wrote dickman_rho.pdf and dickman_rho.png")
print(f"check: rho(2)={rho_at(2):.3f}  rho(4)={rho_at(4):.5f}  "
      f"rho(6)={rho_at(6):.3e}  rho(8)={rho_at(8):.3e}")

# Edit: 2026-07-07
