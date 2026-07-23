# src/iris/ — CUDA GPU solvers (IRIS cluster)

*Edit: 2026-06-30*

## What this is

A single-GPU, single-process CUDA port of **Phase 1** of Algorithm 7 (the
inverted sieve): for a fixed challenge/modulus pair, test `R` lifted integer
candidates `y_r = x + r*M` against every prime `q <= B` in parallel, marking
a `log_sum[R]` smoothness score. This is the one part of this project's
workload that maps very well onto a GPU's SIMT model — see
`wiki/results/gpu-feasibility.md` for the full reasoning on why this part
(and not, say, the modular exponentiations) was chosen.

Everything except that one kernel reuses the project's existing,
MPI-free `src/common/common.cpp` / `src/common/factorization.cpp` utilities:
file I/O (same binary format as `tools/challenges.c`), Phase 2 full trial
division, cbrt reconstruction, and verification.

**Status: written but not yet built or run on real hardware.** There was no
CUDA toolkit or GPU available while writing this. It's a careful first draft,
reasoned through by hand — treat every claim below about it working as
"should work," not "confirmed," until you've run it on **IRIS** (the GPU
cluster; AION is the CPU-only cluster and has no GPU nodes).

## Build

```bash
# On IRIS (the GPU cluster — not AION which is CPU-only):
module load system/CUDA/12.6.0          # confirmed 2026-06-25
# GCC and GMP module names on IRIS may differ from AION's env/development/2023b
# toolchain — check with `module spider GCC` and `module spider GMP` first.

cd src/iris
# IRIS GPU nodes have 4x Nvidia Tesla V100 SXM2 → sm_70 (confirmed 2026-06-25)
make ARCH=sm_70
```

If `nvcc` isn't on `PATH` after the module loads, try `make NVCC=$(which nvcc)` or
confirm with `module list` that `system/CUDA/12.6.0` loaded correctly.

## Run

Point it at any existing alg3-style challenge + modulus file pair (Algorithm
7 reuses alg3's challenge format):

```bash
# Run from project root:
./src/iris/alg7_cuda data/challenges/alg3/challenge_0_alg3_1024.txt \
                     data/challenges/alg3/modulo_alg3_1024.txt \
                     --R 20 --bitbound 24
```

Start with a small `--bitbound` (default 24 — a few million primes, sieves
in milliseconds) to confirm correctness before stepping up to `--bitbound 32`
(the full k32/Algo3/Algo4/Algo7 range, ~203M primes — needs ~537MB of host
RAM for the sieve and noticeably longer to build the prime list).

If it reports "no candidate cleared the threshold," raise `--R` (more
candidates = higher chance one is almost-smooth) before assuming something's
broken — this mirrors the real algorithm's own behavior, not a bug.

The program prints a `result^e mod M == challenge mod M` check at the end
(via the existing `verify_eth_root` from `common.cpp`) — this is the
project's standard correctness gate. Don't trust a speedup number without it
passing, exactly as elsewhere in this project.

## What's deliberately NOT done yet

- **No MPI / multi-GPU.** This is a single-process, single-GPU prototype, by
  design — get the kernel correct first. A real benchmark comparable to the
  rest of this project's results would need an MPI+CUDA hybrid (one GPU per
  rank, or one GPU shared across a node's ranks), analogous to how
  MPI+OpenMP work together today.
- **No per-block partial-sum reduction for `log_sum`.** The kernel writes
  directly via `atomicAdd` on global memory. Modern GPUs handle float atomics
  efficiently, and different primes touch different stride patterns, so true
  write collisions should be infrequent — but if profiling on real hardware
  shows atomic contention is a bottleneck, the fix is the same idea already
  used for the CPU OpenMP version (per-thread/per-block private partials,
  reduced at the end) — see `wiki/implementation/performance-review-2026-06-11.md`
  item C.
- **No top-K candidate retry.** Only the single highest-`log_sum` candidate
  is tried in Phase 2. `log_sum` is an *estimate* (it under-counts primes
  that divide `y_r` more than once — see the code comment in
  `inverted_sieve_kernel`), so the true best candidate could occasionally
  score lower than a runner-up. Worth adding if the single-candidate
  approach turns out to fail too often in practice.
- **No GPU-side big-number arithmetic beyond mod-by-small-prime.** The
  actual cbrt reconstruction (multi-thousand-bit modular exponentiation)
  still runs on CPU via the existing GMP-based `factorization.cpp` — this
  was a deliberate choice, not an oversight; see the wiki page above for why
  GPUs are a poor fit for that part specifically.

## Validating correctness once you have hardware

1. Build at the default small `--bitbound 24`, run against a known-smooth
   alg3/alg7 challenge, confirm `result^e mod M == challenge mod M` passes.
2. Cross-check the *factors found* (printed via Phase 2) against what the
   existing CPU `alg7`/`alg3` solver finds on the same challenge file — they
   should match exactly (same prime, same exponents), since both use the
   same trial-division logic (`kf_trial_divide_table`).
3. Only then move to `--bitbound 32` and to timing/profiling.

Report back what you find — this file and the wiki page should be updated
with real numbers (or real bugs) once you've run it.
