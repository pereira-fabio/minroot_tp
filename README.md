# Modular Cube Root — VDF Cryptanalysis Project

Implementation of efficient cube root computation `challenge^(1/3) mod modulo`
as part of a parallel computing course project on Verifiable Delay Functions (VDFs),
based on the paper *"Cryptanalysis of Algebraic Verifiable Delay Functions"*
(Biryukov et al., CRYPTO 2024).

---

## Background

A Verifiable Delay Function (VDF) requires computing a function that cannot be
parallelized below a minimum time, even with many processors. A key primitive is
computing `x^(1/a) mod p` — the a-th root of x modulo a prime p.

This project implements the factorization-based approach: if the challenge is a
product of small primes, the cube root can be recovered using the homomorphism
`cbrt(a·b) = cbrt(a) · cbrt(b) mod M`, which is much cheaper than direct
modular exponentiation on the full challenge.

---

## Project Structure

```
src/
  common/                    shared infrastructure
    common.h/.cpp              file I/O, sieve, GMP helpers, timing, JSON output
    factorization.h/.cpp       sieve, trial division, Pollard's rho, cbrt cache, blinding
    solver_common.h/.cpp       MPI+OpenMP scaffolding (broadcast, gather, bench loop)
    sizes.h                    canonical challenge sizes
    gpu/                       shared CUDA code
      gpu_common.h               device helpers, JSON, per-run stats
      gpu_k32_kernels.cuh        k32 trial-division kernel, shared by k32/k35/alg3/alg4
      gpu_sieve.h                host-side segmented sieve for --method sieve
  aion/                      MPI+OpenMP solvers (CPU cluster)
    alg2_k16.cpp               Algorithm 2, B = 2^16
    alg2_k32.cpp               Algorithm 2, B = 2^32
    alg2_k35.cpp               Algorithm 2, B = 2^35 (paper's concrete parameters)
    alg3.cpp                   Algorithm 3, one medium prime cofactor
    alg4.cpp                   Algorithm 4, composite cofactor
    alg7.cpp                   Algorithm 7, inverted sieve
  iris/                      CUDA solvers (GPU cluster)
    alg2_k16_cuda.cu  alg2_k32_cuda.cu  alg2_k35_cuda.cu
    alg3_cuda.cu      alg4_cuda.cu      alg7_cuda.cu

experiments/                 research code that is NOT part of the attack
  delta/                     delta-encoded prime table (L2-residency experiment)
  ksweep/                    smoothness-bound sweep
  scale_sim/                 strong-scaling simulators (CPU and GPU)
  crt/                       CRT exponentiation (negative result)
  prefilter/                 B0 threshold pre-filter cost
  reconstruction/            Amdahl analysis: would a device port pay? (no)
                             each holds its own SLURM job script

tools/
  smoke_test.sh              the project's test (`make test`)
  challenges.c               challenge generator
  gen_prime_table.c          prime table for an arbitrary bound 2^k
  timelock/                  standalone time-lock puzzle demo, with its jobs

scripts/                     the sweeps that reproduce the results, nothing else
  aion/                        all_cpu, all_cpu_e5, challenge_gen, challenge_gen_e5
  iris/                        all_gpu, all_gpu_e5, all_gpu_l40, all_gpu_l40_e5

analysis/                    Python plotting and analysis

Makefile                     all builds, CPU and CUDA
REQUIREMENTS.md              prerequisites, install, smoke test
```

See `experiments/README.md` for what each experiment asked and what it found.

---

## Algorithms

### k16 — Precomputed Table Approach

Challenges with 16-bit prime factors (≤65535). There are 6,542 such primes.

**Precomputation (once per modulo file):**
- Compute `e = 3^-1 mod (M-1)` once
- For each prime q ≤ 2^16: store `cbrt_table[q] = q^e mod M`
- Save to `modulo_*.cbrt` file — reloaded on subsequent runs

**Per-challenge solve:**
1. Export challenge as raw bytes
2. Parallel trial division (OpenMP): for each prime, compute `challenge mod prime`
   using native byte arithmetic — GMP only called for the rare divisible cases
3. Reconstruction: multiply `cbrt_table[p]^exponent` for each found factor

**Key insight:** `cbrt(p1·p2·p3) = cbrt(p1)·cbrt(p2)·cbrt(p3) mod M`
— the table lookup avoids recomputing expensive modular exponentiations.

### k32 — On-the-fly Approach

Challenges with 32-bit prime factors (65537 to 2^32). ~203 million primes.

A precomputed table would require ~52GB — infeasible. Instead:

**Per-challenge solve:**
1. Segmented sieve over [65537, 2^32] in 512K-entry segments (fits in L2 cache)
2. For each prime found by sieve: check divisibility with `mpz_tdiv_ui`
3. For divisible primes: count exact exponent with `count_exponent`
4. Reconstruction: compute `powm(p, e, M)` on-the-fly per found factor
   — only ~challenge_bits/32 calls total (e.g. ~34 for 1024-bit challenge)

### alg2_k32 — MPI+OpenMP Hybrid

Distributes the sieve range `[65537, 2^32]` across MPI ranks. Each rank
factors its own slice and computes its partial cbrt product locally
(`∏ cbrt(pᵢ)^eᵢ mod M`). The per-rank partial products are combined by a
log-depth reduction tree (`MPI_Reduce` with a user-defined multiply-mod-M op),
so the combine takes `ceil(log2 N)` rounds instead of `N-1` serial
multiplications on rank 0 — both factoring and reconstruction scale with the
number of ranks.

**Communication pattern:**
- `MPI_Bcast`: challenge + modulo bytes → all ranks (once per challenge)
- `MPI_Barrier`: synchronise before timing each phase
- `MPI_Reduce (MAX)`: max local-cbrt time across ranks → rank 0 (timing only)
- `MPI_Reduce (custom op)`: one `~ceil(modulo_bits/8)` buffer per rank, multiplied
  up a binary tree → rank 0 (cbrt products, and the exact factor products used by
  the completeness/cofactor checks). Each buffer is one element of a contiguous
  derived datatype so MPI never hands the op a partial big integer
- No communication during the hot trial division loop

### alg2_k35 — Paper's Concrete Parameters (B = 2^35)

Extends the sieve range to `[65537, 2^35]`, matching the paper's Algorithm 2
parameters. ~1.1 billion primes in range.

The upper sub-range `(2^32, 2^35]` uses `uint64_t` prime storage since values
exceed `uint32_t`. Three methods are available:

**sieve** — on-the-fly segmented sieve over `[65537, 2^35]`, no precomputed files
needed. This is the default and recommended method.

**table** — two-slice preloaded prime files:
- `k32_primes.bin` for `[65537, 2^32]` (~813 MB, shared with alg2_k32)
- `k35_upper_primes.bin` for `(2^32, 2^35]` (~10 GB, generated on first table run)

**powm** — sequential `mpz_powm` baseline (single rank, no factoring), used only
for speedup comparison.

Breakeven vs sequential powm: ~17000 bits (sieve), ~12000 bits (table).
Below these sizes the factoring overhead exceeds the powm cost. The communication
pattern is identical to alg2_k32.

---

### Algorithm 3 — one medium prime cofactor

Sieves to 2^32 exactly as k32 does, but accepts a candidate whose unfactored
remainder is a single prime `q0` in (2^32, 2^65] rather than requiring it to be
1. Reconstruction folds in `cbrt(q0)` alongside the tabulated small-prime roots.

Sieving cost is unchanged, so `speedup_total` is indistinguishable from k32 --
both land near 43x. What changes is the **hit rate**: Algorithm 3 accepts roughly
119x more of the candidates it is handed, which is the only currency that matters
to an attacker paying for failures.

`cbrt(q0)` is a full-width modular exponentiation whose cost is set by |M|, not
by the size of q0 -- about 1800 ms at 24576 bits, the same as the sequential
baseline. It is computed in an untimed precompute so the benchmark measures the
parallel attack rather than one exponentiation. An attacker could not amortise
it: there are some 2^59 primes in the range, so no table can hold them, and each
successful candidate presents a fresh one.

### Algorithm 4 — composite cofactor

Accepts a cofactor that is a *product* of medium primes, split with Pollard's rho
during reconstruction, and screens candidates against a B0 = 2^20 threshold
pre-filter before sieving. On challenges that are smooth by construction the
pre-filter never fires and the cofactor path is a cached lookup, so it measures
almost exactly what k32 and Algorithm 3 measure -- which is the consistency check.

### Algorithm 7 — inverted sieve

Instead of sieving one candidate against every prime, it searches a range of
lifts `y = x + r*M` for one that is smooth, inverting the loop nesting. Reaches
26.76x at 128 ranks; its table method is not usable.

## Build

Prerequisites, install commands per platform and disk/RAM needs are in
[REQUIREMENTS.md](REQUIREMENTS.md). In short: an MPI compiler, GMP and OpenMP
for the CPU solvers; CUDA only for the GPU ones.

To check a machine end to end, run `make test` — see [Testing](#testing) below.

### CPU (AION)

```bash
make                  # the six solvers + the challenge generator
make experiments      # alg_crt, cpu_scale_sim, delta_convert, prefilter_bench
make tools            # challenges, gen_prime_table
make clean
```

Requires `mpicxx` (OpenMPI) and `libgmp`. On HPC load the modules first:

```bash
module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0
```

### GPU (any NVIDIA card, or IRIS)

```bash
make gpu                  # detects the local card; prints what it chose
make gpu-experiments      # alg_crt_cuda, alg2_ksweep_cuda, gpu_scale_sim_cuda
make gpu ARCH=sm_75       # force one architecture (2080/2080 Super)
make gpu ARCH=portable    # one fatbinary for any Volta-or-newer card

# on a cluster, load the toolkit first:
module load system/CUDA/12.6.0
```

### Manual compilation

Every CPU solver takes the same three shared sources plus the delta module, and
needs both include paths:

```bash
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common -I experiments/delta \
    src/aion/alg2_k32.cpp \
    src/common/common.cpp src/common/factorization.cpp src/common/solver_common.cpp \
    experiments/delta/delta_table.cpp \
    -lgmp -o alg2_k32
```

Substitute any of `alg2_k16`, `alg2_k32`, `alg2_k35`, `alg3`, `alg4`, `alg7`.

## Usage

### Generate challenges first

Nothing runs without them. No cluster needed:

```bash
./challenges 3 1024 16          # 3 challenges, 1024-bit modulus, B = 2^16
./challenges 3 1024 32          # B = 2^32   (k32)
./challenges 3 1024 32 alg3     # + one medium prime cofactor
./challenges 3 1024 32 alg4     # + a cofactor Algorithm 4 will accept
./challenges 3 1024 32 --exponent 5        # a modulus with gcd(5, M-1) = 1
./challenges 3 1024 32 alg3 --exponent 5  # the same for Algorithm 3
```

Files land in `data/challenges/<variant>/`. On a cluster, the SLURM equivalents
generate every canonical size at once:

```bash
sbatch scripts/aion/challenge_gen.slurm        # e = 3
sbatch scripts/aion/challenge_gen_e5.slurm     # e = 5
```

### CPU solvers

`alg2_k16`, `alg2_k32`, `alg2_k35`, `alg3`, `alg4`, `alg7` — MPI programs, so
they run under `mpirun` locally or `srun` on a cluster.

```bash
# one challenge, 2 ranks, on any machine
mpirun -n 2 ./alg2_k32 \
    data/challenges/k32/challenge_0_32_1024.txt \
    data/challenges/k32/modulo_32_1024.txt \
    --method sieve

# the fifth root, against the modulus generated for it
mpirun -n 2 ./alg3 \
    data/challenges/alg3/challenge_0_alg3_1024.txt \
    data/challenges/alg3/modulo_alg3_1024_e5.txt \
    --method sieve --exponent 5

# every canonical size in one execution — cluster only, srun is SLURM's
srun --mpi=pmix -N 64 --ntasks=128 ./alg2_k32 \
    --all --data data/challenges/k32 \
    --method table --skip-powm-baseline \
    --n-challenges 50 --cbrt-dir data/cbrt/alg2_k32_e3 \
    --json out/alg2_k32_table
```

| flag | meaning |
|------|---------|
| `--method sieve\|table\|powm` | factoring strategy; `powm` is the sequential baseline |
| `--all --data DIR` | run every canonical challenge size in one execution |
| `--n-challenges N` | challenges to solve; each is one timed run |
| `--exponent E` | root exponent (default 3) |
| `--modulo-suffix S` | e.g. `_e5`, to select the exponent-5 moduli |
| `--table-format flat\|delta\|auto` | prime table encoding (`auto` detects by file magic) |
| `--primes FILE` | prime table for `--method table` |
| `--cbrt-dir DIR` | where to keep the cube-root cache |
| `--json PREFIX` | write results as `PREFIX_<size>.json` |
| `--skip-powm-baseline` | skip the baseline (the `powm` pass already measured it) |

There is no `--bench` flag: one challenge is one timed run, and `--n-challenges`
sets how many are averaged.

### GPU solvers

`alg2_k16_cuda`, `alg2_k32_cuda`, `alg2_k35_cuda`, `alg3_cuda`, `alg4_cuda`,
`alg7_cuda` — **no MPI**, so no `mpirun`. One process drives every device.

```bash
# positional, one device
./alg2_k16_cuda \
    data/challenges/k16/challenge_0_16_1024.txt \
    data/challenges/k16/modulo_16_1024.txt \
    --method sieve

# by directory and size, splitting the prime range over 4 devices
# (--n-gpus above what you have warns and falls back)
./alg2_k32_cuda --data data/challenges/k32 --size 1024 \
    --method sieve --n-challenges 10 --n-gpus 4

# the fifth root
./alg3_cuda --data data/challenges/alg3 --size 1024 \
    --method sieve --exponent 5 --modulo-suffix _e5
```

Same flags as the CPU solvers, minus `--all`, `--cbrt-dir` and
`--skip-powm-baseline`, plus:

| flag | meaning |
|------|---------|
| `--size N` | challenge size, with `--data`, instead of naming both files |
| `--n-gpus N` | split the prime range across N devices |
| `--n-runs N` | warm runs to average |

Two things worth knowing before the first k32 or k35 run:

**`--method sieve` generates the primes every time.** For k32 that is 203
million primes and about 35 seconds before any work starts, and the solver says
so. `--method table` reads them from disk instead, but needs the 813 MB table
built once with `mpirun -n 1 ./alg2_k32 --precompute`. Sieve is the zero-setup
option; table is the one to use if you are running more than a couple of times.

**`--n-challenges` above the number you generated is not an error on the GPU.**
Missing files are skipped with a warning and the run continues on what exists.
The CPU solvers' `--all` does the opposite and aborts on the first missing size.

**Do not trust `speedup_total` from a GPU run.** Its `powm_baseline_ms` measures
the forward `y^e` with a small exponent and comes out a few hundred times too
small. Every GPU figure in this README divides by the AION baseline for the
matching size instead.

### The whole study

```bash
sbatch scripts/aion/all_cpu.slurm         # every CPU variant, N1..N64, e=3
sbatch scripts/aion/all_cpu_e5.slurm      # the same at e=5
sbatch scripts/iris/all_gpu.slurm         # every GPU variant, V100, e=3
sbatch scripts/iris/all_gpu_e5.slurm      # the same at e=5
sbatch scripts/iris/all_gpu_l40.slurm     # L40 partition, e=3
sbatch scripts/iris/all_gpu_l40_e5.slurm  # L40 partition, e=5
```

Each sweep is one exponent, so e=3 and e=5 submit and fail independently.
Algorithm 7 runs at e=3 only, on both clusters.

Experiment jobs live with their experiment, not here — e.g.
`sbatch experiments/ksweep/ksweep.slurm`. See `experiments/README.md`.
The time-lock demo is `tools/timelock/timelock_demo.slurm`.

### Per-algorithm notes

- **`alg2_k35 --method table`** generates `data/k35_upper_primes.bin` (~10 GB) on
  first use, and needs ~9.6 GB of VRAM on the GPU. Use `--method sieve` on a
  consumer card.
- **`alg7`** takes `--R <range>` (the study used `--R 1048576`) and reads the
  Algorithm 3 challenge set.
- **`alg3` / `alg4`** read `data/challenges/alg3` and `data/challenges/alg4`.
- **The first run** for a given modulus precomputes and saves a `.cbrt` cache;
  later runs load it from disk.
- **Delta tables** are built with the `delta_convert` tool, not by a solver flag:
  `./delta_convert data/k32_primes.bin`, or `--u64` for the k35 upper table.

---

## Testing

```bash
make test
```

Two minutes, no cluster. Builds what is missing, generates small challenges,
runs every solver once and prints a pass/fail table — using the GPU when `nvcc`
and a card are present, and skipping that half cleanly when they are not.

```
  PASS cpu k16   sieve    correctness gate passed
  PASS cpu alg3  sieve    correctness gate passed
  PASS cpu k32   e5       correctness gate passed
  PASS gpu alg4  sieve    correctness gate passed
  PASS gpu alg3  q0       warm cache; recon 0.0536 vs k32 0.0458 ms, powm 0.309 ms
  SKIP cpu k35   sieve    1.4 B primes per run — use --slow
  ...
  15 passed, 0 failed, 4 skipped
```

Fifteen checks run by default, four more behind `--slow`: the correctness gate
on k16, k32, Algorithm 3 and Algorithm 4 across both CPU and GPU, the same at
exponent 5 against moduli where `gcd(5, M-1) = 1`, the sequential baseline, and
that `cbrt(q0)` stays out of the timed region for the two cofactor algorithms.

Two of those are worth knowing about.

**The correctness gate is printed, not enforced.** `verify_eth_root` prints
`FAILURE` and continues — exit code 0, JSON still written. The test therefore
classifies on output, not exit status, so a wrong root fails even when the
solver believed it succeeded.

**The cofactor rows measure rather than look for a log line.** `cbrt(q0)` is a
full-width modular exponentiation whose cost is set by the modulus size, not by
q0, and it must stay outside the timed region. Algorithm 3 and k32 reconstruct a
comparable set of small primes and differ in exactly that fold, so the gap
between their `recon_ms` is the cofactor's cost — weighed against one measured
`powm` at the same size, which needs no hardcoded threshold and holds on any
hardware.

`--slow` adds k35 and Algorithm 7, both slow regardless of challenge size
because the sieve range follows the smoothness bound rather than the modulus.
`./tools/smoke_test.sh --help` lists the rest.

Verified on an RTX 2080 Super (sm_75) with GCC 15.2.0 and MPICH 4.3.2, and on
the AION and IRIS clusters with GCC 13.2.0 and Open MPI 4.1.6.

---

## Analysis and Plotting

Benchmark results (JSON files from SLURM runs) are visualised with a Python plotter.

```
analysis/
  ploter.py          generates all benchmark figures from JSON results
  env/               Python venv with numpy + matplotlib (run from project root)
```

Run from project root:

```bash
analysis/env/bin/python3 analysis/ploter.py 'results/**/*.json' --out figures/ --format png
```

Output goes to `figures/` (36 PNG files). See `docs/figures.md` for a description of
every figure and how to interpret it.

---

## Performance Results

All figures below are `speedup_total = powm_baseline_ms / (factor_ms + recon_ms)`,
measured on AION with e = 3 and 48 warm runs per point. Every reported result
passed the correctness gate: `result^3 mod M == challenge`.

Two caveats that matter for reading these numbers:

- **The challenges are smooth by construction.** What is measured is the cost of
  inverting a challenge that is *already* smooth, not the cost of finding one.
  See "Known Limitations" below.
- **Warm runs that took a cube-root cache miss are excluded.** A miss pays a
  full-width `mpz_powm` (~1.8 s at 24576 bits), so it measures cache population
  rather than the attack. This affects Algorithm 4's table method at 24576 bits,
  where 1 run in 49 is dropped; without the exclusion that column reads 16.24x
  instead of 49.39x.

### Two regimes

The attack is a large-modulus phenomenon. At MinRoot's own deployment size it
loses to the sequential baseline; at the design point it wins by ~50x.

| 128 MPI ranks, table method | 256-bit | 1024-bit | 2048-bit | 8192-bit | 24576-bit |
|-----------------------------|--------:|---------:|---------:|---------:|----------:|
| k16   (B = 2^16)            |  0.366x |   3.382x |   9.985x |  43.407x |  119.022x |
| k32   (B = 2^32)            |  0.017x |   0.156x |   0.789x |  11.145x |   51.839x |
| k35   (B = 2^35)            |  0.002x |   0.023x |   0.119x |   2.108x |   11.922x |
| Algorithm 3                 |  0.016x |   0.154x |   0.782x |  11.155x |   52.475x |
| Algorithm 4                 |  0.013x |   0.142x |   0.687x |  10.374x |   49.581x |
| Algorithm 7 (sieve)         |  0.002x |   0.028x |   0.165x |   4.143x |   26.760x |

k16 is the only variant that beats the baseline at cryptographic sizes, and it
pays for that in hit rate: a 256-bit challenge is 2^16-smooth with probability
2^-69.6, which makes the win unusable.

Sequential `mpz_powm` baseline, for reference: 0.026 ms at 256 bits, 3.04 ms at
2048, 117.6 ms at 8192, 1806.9 ms at 24576.

### CPU strong scaling at 24576 bits

Ranks are 2 per node (one per NUMA socket), 32 OpenMP threads each. 128 ranks
across 64 nodes is the hard account limit on AION.

| Variant | Method | 2r | 4r | 8r | 16r | 32r | 64r | 128r |
|---------|--------|---:|---:|---:|----:|----:|----:|-----:|
| k16     | table  | 20.99x | 40.35x | 75.47x | 131.12x | **180.58x** | 176.36x | 119.02x |
| k32     | sieve  |  1.66x |  2.87x |  5.15x |   9.46x |  17.57x |  30.27x |  42.82x |
| k32     | table  |  2.18x |  3.77x |  6.73x |  13.00x |  23.71x |  40.68x |  51.84x |
| k35     | sieve  |  0.23x |  0.40x |  0.72x |   1.40x |   2.62x |   4.95x |   8.85x |
| k35     | table  |  0.27x |  0.52x |  1.00x |   1.92x |   3.60x |   6.83x |  11.92x |
| Alg 3   | sieve  |  1.70x |  2.91x |  5.23x |   9.61x |  17.54x |  30.37x |  43.57x |
| Alg 3   | table  |  2.17x |  3.71x |  6.72x |  12.96x |  23.63x |  40.17x |  52.48x |
| Alg 4   | sieve  |  1.70x |  2.87x |  5.13x |   9.52x |  17.34x |  29.19x |  41.33x |
| Alg 4   | table  |  2.22x |  3.76x |  6.93x |  12.84x |  23.41x |  38.28x |  49.39x |
| Alg 7   | sieve  |  0.89x |  1.58x |  2.96x |   5.49x |  10.00x |  17.84x |  26.76x |

k32, Algorithm 3 and Algorithm 4 reach the paper's ~43x target on 128 ranks
rather than the 2^54.5 processors of its circuit model. That the three agree to
within a few percent is expected and is a consistency check: they differ only in
which cofactor they accept after sieving, and on a challenge that is smooth by
construction that branch never fires.

**k16 peaks at 32 ranks and then declines.** Beyond that the 6542-prime sieve is
divided too finely for the reconstruction and gather to be worth their cost.
These figures were measured with the old serial combine (gather to rank 0, then
`N-1` multiplications); the combine is now a log-depth tree, which
`analysis/ideal_combine.py` projects to remove most of this decline. Not yet
re-measured on the cluster.

**k35 does not reach the paper's figure**, and cannot here: it would need roughly
1024 ranks, eight times the maximum allocation. Any number quoted beyond 128
ranks in the thesis is extrapolation from this curve, not measurement.

### GPU, 24576 bits

Speedups use the AION N1 `powm_baseline` for the matching size. IRIS's own
baseline measures the forward `y^e` with a small exponent and is ~920x too
small; do not use it.

| Variant | 1x V100 | 4x V100 | 4x L40 | factor-only (1x V100) |
|---------|--------:|--------:|-------:|----------------------:|
| k16     |  11.69x |  12.17x | 16.71x |              43,097x  |
| k32     |   1.11x |   3.75x |  9.18x |                  1.2x |
| k35     |   0.12x |   0.36x |  1.23x |                  0.1x |
| Alg 3   |   0.57x |       - |  1.43x |                  1.1x |
| Alg 4   |   0.57x |   0.92x |  1.43x |                  1.1x |

The last column is the point: k16's trial division on a GPU is ~43,000x faster
than the sequential baseline, and that collapses to 11.69x end to end because
reconstruction stays on the host — no multi-precision arithmetic was written for
the device. Reconstruction, not sieving, is the binding constraint.

Algorithms 3 and 4 are understated here. Their reconstruction carries the
medium-prime cube root `cbrt(q0)`, which the CPU path takes from an untimed
precompute and the CUDA path did not, paying a full-width exponentiation on
every timed run. That is fixed in the current source but the numbers above
predate the fix.

---

## HPC Environment

- Cluster: AION (University of Luxembourg)
- Nodes: AMD EPYC 7452 (2 sockets × 32 cores, 64 cores/node)
- MPI: OpenMPI 4.1.6 via `mpi/OpenMPI/4.1.6-GCC-13.2.0`
- GMP: 6.3.0 via `math/GMP/6.3.0-GCCcore-13.2.0`
- Compiler: GCC 13.2.0
- Scheduler: SLURM
- Recommended SLURM config: `--ntasks-per-node=2 --cpus-per-task=32`
  (one MPI rank per NUMA node, 32 OpenMP threads per rank)

---

## Known Limitations and Future Work

- **The challenges are smooth by construction.** This is the limitation that
  bounds every number in this repository. `tools/challenges.c` builds each
  challenge as a product of small primes, so the benchmarks measure the cost of
  *inverting* a smooth challenge, not the cost of finding one. An attacker pays
  for the failures too, and the acceptance probability falls with the modulus far
  faster than the speedup rises: a 256-bit challenge is 2^32-smooth with
  probability about 2^-24.9, and a 24576-bit one with probability rho(768) — a
  quantity with no physical interpretation. The attack becomes profitable per
  attempt in exactly the regime where attempts stop being winnable.

- **128 MPI ranks is a hard ceiling**, and 4 GPUs. Both are account limits on the
  University of Luxembourg clusters, not choices. k35 would need roughly 1024
  ranks to reach the paper's figure and Algorithm 7 around 530 nodes; those
  numbers are extrapolated from the measured N1-N64 curve and are projections,
  not measurements.

- **GPU reconstruction runs on the host.** No multi-precision arithmetic was
  written for the device, so every variant gathers to the CPU to reconstruct.
  This is what turns k16's ~43,000x factoring advantage into 11.7x end to end.
  A device port would move the ceiling rather than remove it. The host product
  is now a parallel tree over all the job's cores (`kf_product_mod_tree`, also
  used by k16, which was fully serial); the GPU figures above predate it.

- **Algorithms 3 and 4 are understated on the GPU** in the results above. Their
  CUDA path computed the medium-prime cube root inside the timed region on every
  warm run, while the MPI path takes it from an untimed precompute. The source
  now does the same on both; the published numbers predate the fix and need a
  re-run.

- **k35 table file size**: `k35_upper_primes.bin` is ~10 GB, generated on the
  first `--method table` run. A one-time cost, but it needs scratch space.

- **Multiple `srun` calls hang**: several sequential `srun` invocations in one
  SLURM job cause MPI reinit issues. Use `--all` to run every challenge size in a
  single MPI execution.
