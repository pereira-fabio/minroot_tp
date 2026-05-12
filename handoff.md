# Project Handoff — MinRoot VDF Cryptanalysis

## Goal

Implement and benchmark **Algorithms 2 and 3** from Biryukov et al. (CRYPTO 2024)
for attacking the MinRoot VDF. Each algorithm computes cube roots modulo a large prime p
by finding smooth or almost-smooth pre-images.

| Algorithm | Variant | Success probability | Key constraint |
|-----------|---------|--------------------|----|
| 2 | k16, k32, k35 | ~2^-16 / 2^-24.9 / 2^-21 | All factors ≤ 2^B |
| 3 | alg3 | ~2^-18 | All factors ≤ 2^32, plus ONE prime q₀ ∈ (2^32, 2^65] |

Algorithm 3 has ~64× more frequent successes than k32. Its reconstruction is slower
(must compute `cbrt(q₀) = mpz_powm(q₀, e_inv, p)` on every solve), so
`speedup_total < 1` at all tested sizes — this is expected. The advantage is measured
in **successful inversions per hour**, not per-attempt speed (see Figure 13).

---

## Current State

### What works
- **k16/k32/k35** (Algorithm 2): fully implemented, benchmarked up to 24576 bits
  across N1–N16 nodes (5 array tasks). Results in `results/k*_mpi_N*/`.
- **alg3** (Algorithm 3): implemented, first HPC run (job 12171516–12171520) succeeded
  across all sizes 256–7000 bits, N1–N16. Results in `results/alg3_mpi_N*/`.
- **Hit rate tracking**: added in latest commits — JSON files now contain a
  `"hit_stats"` block with `smooth_count`, `almost_smooth_count`, `hit_type[]`,
  `cofactor_bits[]`. This is what distinguishes alg3 from k32 in the plotter.
- **Plotter**: generates Figures 1–13. Figure 12 (hit stats) and Figure 13
  (effective throughput vs k32) are new and require alg3 results with `hit_stats`.
- **SIZES extended**: alg3 now runs the same 11 sizes as k32/k35
  (256–24576 bits), enabling direct side-by-side comparison in Figure 11.

### What is NOT done yet
- alg3 N1 is missing `sieve_24576` — likely a timeout on the single-node job.
  All other node counts (N2, N4, N8, N16) have 24576-bit data.
- Nothing else outstanding: all figures are complete and validated.

---

## Files Actively Modified (this work sprint)

| File | What changed |
|------|-------------|
| `src/common/k_common.h` | Added `hit_type` and `cofactor_bits` vectors to `benchmark_results_t` |
| `src/common/k_common.cpp` | Writes `"hit_stats"` JSON section (guarded on non-empty — absent for k16/k32/k35) |
| `src/solvers/k_alg3_mpi.cpp` | Records hit_type + cofactor_bits per warm run; **bug fix**: method_name was hardcoded "sieve" even for powm runs |
| `scripts/alg3.slurm` | SIZES extended from `7000` → full set ending at `24576` |
| `scripts/challenge_gen.slurm` | SIZES_ALG3 same extension |
| `analysis/ploter.py` | `is_alg3()`/`algo_name()` helpers; alg3 added to Figure 11; new Figure 12 (hit stats); new Figure 13 (effective speedup); `7000` added to `NOMINAL_SIZES`; **bug fix** (commit `cdedd6e`): inject `_source_algo` from filename so old alg3 files without `hit_stats` are not misclassified as k32 (was corrupting Figure 13 baseline at large sizes) |

All changes are committed (see `git log`). Nothing is in a half-finished state.

---

## Things Tried That Failed / Bugs Found

### 1. Method name hardcoded "sieve" in alg3 (FIXED — commit `db6cdb5`)
`k_alg3_mpi.cpp` line 176 had `ksc_init_json_results(..., "sieve", ...)` hardcoded.
k32_mpi sets it dynamically: `(method == METHOD_TABLE) ? "table" : (method == METHOD_POWM) ? "powm" : "sieve"`.
Result: all alg3 `--method powm` JSON files had `"method": "sieve"` — wrong label.
Fix: added `const char *mname = (method == METHOD_POWM) ? "powm" : "sieve"` before the init call.

### 2. alg3 SIZES stopped at 7000 bits (FIXED)
Original `scripts/alg3.slurm` had `SIZES="256 512 1024 2048 4096 7000"` while k32
runs up to 24576. This prevented the 17.5× speedup regime from being reached by alg3.
Extended to match k32 exactly.

### 3. speedup_total < 1 for alg3 (NOT a bug — expected by design)
`recon_ms ≈ powm_baseline_ms` at all sizes because rank 0 always computes
`mpz_powm(q₀, e_inv, p)` for the medium prime. k32 achieves 17.5× at 24576 bits
because its `recon_ms` = cached table lookups (microseconds). alg3 cannot avoid
this extra powm — it's fundamental to the algorithm.
The correct comparison metric is Figure 13: effective throughput = `speedup_total × p_success`.

### 4. powm JSON files have empty warm_runs (NOT a bug — expected)
When `--method powm` is passed, the bench loop is skipped by design (same in k32_mpi).
The powm JSON files only record `powm_baseline_ms` as a reference. `runs_valid = N`
is a counting artifact (blinding_attempts vector is empty, so all runs count as valid).

### 5. cofactor_bits constant per run (NOT a bug — expected)
All 9 warm runs solve the same `challenge_0_alg3_${BITS}.txt`, so q₀ is identical
each time. Variation appears across sizes (different challenge files, different q₀).

### 6. alg3 files mis-classified as k32 in plotter (FIXED — commit `cdedd6e`)
Old alg3 runs (job 12171516/17) predate the `hit_stats` field. `is_alg3(d)` used
`"hit_stats" in d` as the only discriminator, so those files were identified as k32
and polluted the k32 speedup baseline. Worst case: at 24576 bits, alg3 has
speedup_total=0 while real k32 has 17.5×, so the k32 baseline in Figure 13 was
being replaced by 0 (since alg3_mpi_N* sorts before k32_mpi_N* alphabetically).
Fix: `load_file()` injects `_source_algo="alg3"` from the filename prefix;
`is_alg3()` checks this tag first; `dedup_datasets()` prefers datasets with
`hit_stats` when deduplicating so new alg3 runs survive.

---

## Table Method: Intentionally Excluded from alg3

The table method precomputes cbrt for all primes ≤ 2^32 (~800 MB file).
For k32/k35 it makes `recon_ms ≈ 0`, enabling 17.5× speedup at 24576 bits.
For alg3 it would NOT help: the bottleneck is `mpz_powm(q₀, e_inv, p)` (one per solve),
which costs the same as `powm_baseline_ms` regardless of table vs sieve for small primes.
Adding the table would cost 800 MB I/O with negligible gain. This matches the paper's
intent: Algorithm 3's advantage is success probability, not reconstruction speed.

---

## Current Status — All Benchmarking Complete

All HPC runs are done. Figures 1–13 regenerated and validated locally.

**Figure 13 summary** (effective throughput, alg3 vs k32, 32 ranks, sieve):
- At 1024 bits: alg3 ≈ 119× better (speedup_total ≈ equal, probability dominates)
- At 24576 bits: alg3 ≈ 6.5× better (k32's 17.5× reconstruction speedup partially offsets
  alg3's 119× probability advantage: 119 × 0.95/17.51 ≈ 6.5)

**Figure 12 validated**: all `cofactor_bits` in 33–65 range; `almost_smooth_count = 9`
(all warm runs) at every size — no B-smooth hits observed.

### Remaining item
- alg3 N1 missing `sieve_24576` (single-node job likely timed out). Not worth re-running
  unless strong-scaling plot for N1 is specifically needed.

### To regenerate figures from scratch
```bash
analysis/env/bin/python3 analysis/ploter.py 'results/**/*.json' --out figures/
```

### (Future) Consider adding a table mode to alg3
Only worth doing if someone wants to measure whether the table helps at all for
small-prime cbrt lookups in alg3. Expected improvement: < 5% of total time.

---

## Architecture Quick Reference

```
src/
  common/
    k_common.h/.cpp          — benchmark_results_t struct, write_benchmark_json()
    k_solver_common.h/.cpp   — ksc_alg3_broadcast_cofactor(), cbrt cache, MPI helpers
    k_factorization.h/.cpp   — trial division sieve, blinding tables
  solvers/
    k16_mpi.cpp / k32_mpi.cpp / k35_mpi.cpp   — Algorithm 2 variants (DO NOT MODIFY)
    k_alg3_mpi.cpp                              — Algorithm 3 (actively developed)
tools/
  challenges.c               — challenge generator; ./challenges N bits B [alg3]
scripts/
  challenge_gen.slurm        — generates challenges for all algorithms
  k16.slurm / k32.slurm / k35.slurm   — Algorithm 2 SLURM jobs
  alg3.slurm                 — Algorithm 3 SLURM job array (N1–N16)
analysis/
  ploter.py                  — all figures; run from project root
data/
  challenges/k16/ k32/ k35/ alg3/   — challenge + modulo + factor files
  cbrt/                              — precomputed cube root caches (.cbrt.<rank>, .cbrt_alg3.<rank>)
  results/                           — HPC output (rsynced to results/ locally)
```

**Isolation contract**: alg3 never touches `data/challenges/k*/`, `data/cbrt/*.cbrt.*`,
or `data/results/k*_mpi_N*/`. Algorithm 2 solver source files are never modified.
