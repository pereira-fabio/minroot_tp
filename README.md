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
1. Segmented sieve over [65537, 2^32] in 512K-entry segments (fits in L2 cache);
   segments store odd numbers only, so each covers 1M integers
2. For each prime found by sieve: check divisibility with `mpz_tdiv_ui`. Two
   primes whose product is below 2^62 share one call (`y mod p1*p2`, then
   reduce by each) — GMP's remainder loop costs the same up to a 62-bit
   divisor, so this halves the passes over y for primes below ~2^31
3. For divisible primes: count exact exponent with `count_exponent`
4. Reconstruction: compute `powm(p, e, M)` on-the-fly per found factor
   — only ~challenge_bits/32 calls total (e.g. ~34 for 1024-bit challenge)

### alg2_k32 — MPI+OpenMP Hybrid

Distributes the sieve range `[65537, 2^32]` across MPI ranks, in pieces of
equal estimated trial-division cost rather than equal width (`kf_cost_split`:
low ranges hold more primes but test them in pairs, and the sieve adds a cost
per integer; at 24576 bits this cut the slowest rank by 1.13-1.18x in a local
8- and 32-rank simulation). Each rank
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
- `MPI_Allgather(v)` (cofactor sharing): seven checkpoints per trial division,
  each sending only the primes found since the last one

**Cofactor sharing.** Each test walks every limb of y, so every factor divided
out makes later tests cheaper. At 128 ranks x 32 threads a thread finds ~0.2 of
a challenge's ~768 factors, so its copy of y barely shrinks on its own. Trial
division therefore runs in eight phases; at each checkpoint, ranks exchange the
primes they found since the last one and divide them out (`ksc_share_found`). On a
smooth y this shrinks y for everyone as the scan advances: 2.2x in a
single-process simulation where every thread sees all finds, and 1.2-1.4x
measured with 2-4 local ranks, where each rank already holds a quarter to a
half of the factors. On AION at 128 ranks, k32 trial division went from 15.05
to 10.62 ms at 24576 bits (job 16123610, below). Two caveats:

- **It only helps a y that really is smooth.** A random candidate has fewer
  than one prime factor in [2^16, 2^32] on average, so in a real attack it
  speeds up the final, successful candidate and not the search for one. The
  benchmark challenges are smooth by construction, so it raises the measured
  numbers more than the real cost of an attack. Report it as part of "invert a
  smooth challenge".
- **It costs 3-5% when there is nothing to share** (the phase barriers).

Enabled in k32, alg3, alg4 and alg7's phase 2. k35 and k16 do not use it: some
k35 ranks skip a lower-range call, which would deadlock the collective, and k16's
slices are ~51 primes. With `--method table` it now only runs under `--td scan`;
the default gcd route below replaces it.

**gcd route (table method, default).** A prime table never changes during a
run, so each thread's chunk of it can be multiplied into one number P_c before
timing starts (`kf_table_products_prepare`, chunks of at most 65536 primes).
P_c is squarefree, so g_c = gcd(y, P_c mod y) is exactly the product of the
chunk's primes that divide y. Per challenge a thread pays one reduction of P_c,
whose cost is set by P_c's bits, and one gcd, instead of one walk over all of y
per prime pair. Only a chunk with g_c != 1 is split, against the small g_c, and
exponents are then read off y. Measured against the per-prime scan (with
cofactor sharing):

| Setup | scan | gcd route | |
|---|---:|---:|---:|
| one thread, 52K primes near 2^31 (an AION thread's slice), 24576-bit y, 20 numbers | 297 ms | 98 ms | 3.0x |
| k32 table, 2 ranks x 2 threads, 24576 bits, `Factoring` | 11.35 s | 6.85 s | 1.66x |
| k32 table, 2 ranks x 2 threads, 8192 bits, `Factoring` | 5.42 s | 3.80 s | 1.43x |
| alg3 table, same, 8192 bits | 5.80 s | 4.21 s | 1.38x |
| alg4 table, same, 8192 bits | 5.69 s | 3.83 s | 1.48x |

The local MPI runs give each thread ~50M primes, so the comparison there is
against a scan that cofactor sharing already speeds up 1.2-1.45x; an AION
thread holds ~50K primes, the first row's case. Not yet measured on AION.

- **It works on any y, smooth or not**: the cost does not depend on y having
  factors, so unlike cofactor sharing it also speeds up rejected candidates in
  a real attack.
- **The products are precomputation**, like the prime table itself: built once
  per rank, untimed, and reported at startup (`chunk products built in ...`).
  They take as many bits as the slice's primes, ~6 MB per rank at 128 ranks
  (800 MB per rank on a 2-rank local run, built in ~35 s on 2 threads).
- **The rank split is equal-width on this route.** A prime near x adds
  log2(x) bits to the products at density 1/ln x, a constant per integer, so
  `kf_cost_split`'s paired-scan model (which gives ranks below 2^31 about twice
  the primes) does not apply.
- **Small y gains less.** At 1024 bits on one thread the gcd route was ~1.2x
  slower than the scan; from 4096 bits up it was faster in every test.
- k32, alg3, alg4 and alg7's phase 2 use it with `--method table`; `--td scan`
  restores the per-prime scan. k35 always scans: its upper range is a 64-bit
  table this route does not cover. k16 always scans too: ~51 primes per rank
  are too few for a gcd to pay off (0.75 vs 0.59 ms factoring at 8192 bits). The sieve method is unchanged, since it has
  no table to precompute from.

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
make -B gpu GPU_DEFS=-DKF_TD_ILP=1   # one prime per thread in the k32 kernel (default: two)
make -B gpu GPU_DEFS=-DKF_TD_ILP=4   # four primes per thread

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
| `--td gcd\|scan` | table-method trial division: `gcd` (default) reduces precomputed chunk products of the primes mod y and takes one gcd per chunk; `scan` tests each prime pair against y, with cofactor sharing |
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
| `--recon auto\|host\|device` | where the final reconstruction product runs: a tree over the host cores, a tree of Montgomery-multiply kernels on device 0 (`gpu_recon.cuh`), or `auto` (default): the device once the product is big enough to pay for its launches |
| `--td-phases N` | k32, k35, Algorithms 3 and 4: run trial division as N launch rounds and divide the factors found so far out of y between them (cofactor sharing, as on the CPU; default 16, 1 = off; not with the delta table). RTX 2080 SUPER, k32 8192-bit: warm factor_ms 43.0 -> 28.8 ms. Only a smooth y shrinks, so this speeds up inverting a smooth challenge, not the search for one |

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
`N-1` multiplications). With the log-depth tree combine, k16 at 128 ranks
re-measures at 569x, well above the old 32-rank peak (next section).

**k35 does not reach the paper's figure**, and cannot here: it would need roughly
1024 ranks, eight times the maximum allocation. Any number quoted beyond 128
ranks in the thesis is extrapolation from this curve, not measurement.

### After the reduction trees: 24576 bits, 128 ranks

AION job 16119156: table method, 64 nodes x 2 ranks x 32 threads, 50 challenges
(k32: 49 valid, 1 with blinding exhausted). Results in
`data/results/aion/alg2/{k16,k32}_16119156_N64/`. The run skipped the powm
baseline, so speedups use the 1806.9 ms baseline above.

The build used for this job sits between the tree combine and the cost-balanced
rank split. It has the `MPI_Reduce` combine (`gather_ms` is 0), but ranks still
took equal-width ranges (every `tested_per_rank`, which counts integers, is
33,553,920), and the per-rank reconstruction looks serial. This table therefore
measures mainly the combine. The later changes need a rerun on the current
branch.

| Variant | before (table above) | now | factor_ms | local_cbrt_ms | combine_ms | total_ms |
|---------|-----:|-----:|-----:|-----:|-----:|-----:|
| k16     | 119.02x | **569.1x** |  0.82 | 1.46 | 0.89 |  3.17 |
| k32     |  51.84x | **105.2x** | 15.05 | 1.21 | 0.93 | 17.18 |

- **The combine went from ~12.8 ms to under 1 ms.** That is 7 tree rounds of
  about 130 us each, roughly one 24576-bit multiplication plus network latency.
  The serial version measured 12.80 ms at 128 ranks (k16 breakdown).
- **k16 no longer declines past 32 ranks.** Its 128-rank figure is 3.2x the old
  32-rank peak (180.58x). Only 128 ranks were measured, so the full curve still
  needs a sweep.
- **k32 is now factor-bound.** Trial division is 88% of the 17.2 ms.
- **These are warm-cache numbers.** In the k32 job, the first warm-up run of
  each new challenge took ~1.85 s. That is the cube roots of primes not yet in
  the cache, each a full-width exponentiation, computed before the timed runs
  hit the cache. A k32 challenge with a cold cache therefore costs ~1.87 s,
  slightly more than the baseline. k16's complete 6542-root table is a
  legitimate precomputation; k32's 203 million roots at 24576 bits cannot be
  stored, so its warm-cache speedup assumes a cache an attacker could not hold
  in full.

### After the cost split and cofactor sharing: k32, 24576 bits, 128 ranks

AION job 16123610: same setup (table method, 64 nodes x 2 ranks x 32 threads,
50 challenges, 49 valid, 1 with blinding exhausted), built from `main` with the
cost-balanced rank split and cofactor sharing. Results in
`data/results/aion/alg2/k32_16123610_N64/`. This run measured its own powm
baseline, 1802.6 ms.

| k32 | job 16119156 | job 16123610 |
|-----|-----:|-----:|
| factor_ms     | 15.05 | **10.62** |
| local_cbrt_ms |  1.21 |  1.23 |
| combine_ms    |  0.93 |  0.91 |
| total_ms      | 17.18 | **12.76** (median 12.56) |
| speedup       | 105.2x | **141.2x** |

- **Trial division is 1.42x faster** (15.05 -> 10.62 ms, std 0.47 ms over 49
  runs). This is the low end of the 1.2-2.2x range estimated for cofactor
  sharing: each rank holds only ~6 of a challenge's ~768 factors, so nearly all
  of the shrinking comes from the other ranks' finds at each checkpoint.
- **The cost split is active.** `tested_per_rank` is no longer constant: ranks
  0-41, whose primes lie below ~2^31 where pairing makes each test cheaper,
  take about twice as many as ranks 42-127.
- **Reconstruction is unchanged**, about 2.1 ms (17% of the total). The
  combine is already near one multiplication per tree round, so trial division
  remains the target.
- **The warm-cache and smooth-y caveats above still apply.** Cofactor sharing
  only helps a y that is smooth, so this improves the "invert a smooth
  challenge" figure, not the cost of finding one.
- The `powm` JSON in the same directory reports `runs_valid: 0,
  runs_failed: 50`. That is how the writer counts a baseline-only run (it
  records no per-run timings), not 50 failures; its `powm_baseline_ms` is the
  figure used here.

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
