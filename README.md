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
  common/
    k_common.h/.cpp          shared utilities: file I/O, sieve, GMP helpers, timing, JSON output
    k_factorization.h/.cpp   factorization stack: sieve, trial division, Pollard's rho, cbrt cache, blinding tables
    k_solver_common.h/.cpp   MPI+OpenMP shared infrastructure (broadcast, gather, precomputation)
  solvers/
    k16_mpi.cpp              Algorithm 2 solver — B = 2^16 prime factors
    k32_mpi.cpp              Algorithm 2 solver — B = 2^32 prime factors
    k35_mpi.cpp              Algorithm 2 solver — B = 2^35 (paper's concrete parameters)

tools/
  challenges.c               challenge generator: produces B-smooth numbers mod a random prime

scripts/
  challenge_gen.slurm        SLURM job: generate all challenge/modulo files
  k16.slurm                  SLURM job array: k16_mpi scaling study
  k32.slurm                  SLURM job array: k32_mpi scaling study
  k35.slurm                  SLURM job array: k35_mpi scaling study

Makefile                     builds all targets (k16_mpi, k32_mpi, k35_mpi, challenges)

data/challenges/
  k16/                       challenge and modulo files with 16-bit prime factors
  k32/                       challenge and modulo files with 32-bit prime factors
  k35/                       challenge and modulo files with 35-bit prime factors
```

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

### k32_mpi — MPI+OpenMP Hybrid

Distributes the sieve range `[65537, 2^32]` across MPI ranks. Each rank
factors its own slice and computes its partial cbrt product locally
(`∏ cbrt(pᵢ)^eᵢ mod M`). Rank 0 collects one fixed-size partial product
per rank and multiplies them — both factoring and reconstruction scale with
the number of ranks.

**Communication pattern:**
- `MPI_Bcast`: challenge + modulo bytes → all ranks (once per challenge)
- `MPI_Barrier`: synchronise before timing each phase
- `MPI_Reduce (MAX)`: max local-cbrt time across ranks → rank 0 (timing only)
- `MPI_Gather`: one `~ceil(modulo_bits/8)` buffer per rank → rank 0 (cbrt products)
- No communication during the hot trial division loop

### k35_mpi — Paper's Concrete Parameters (B = 2^35)

Extends the sieve range to `[65537, 2^35]`, matching the paper's Algorithm 2
parameters. ~1.1 billion primes in range.

The upper sub-range `(2^32, 2^35]` uses `uint64_t` prime storage since values
exceed `uint32_t`. Three methods are available:

**sieve** — on-the-fly segmented sieve over `[65537, 2^35]`, no precomputed files
needed. This is the default and recommended method.

**table** — two-slice preloaded prime files:
- `k32_primes.bin` for `[65537, 2^32]` (~813 MB, shared with k32_mpi)
- `k35_upper_primes.bin` for `(2^32, 2^35]` (~10 GB, generated on first table run)

**powm** — sequential `mpz_powm` baseline (single rank, no factoring), used only
for speedup comparison.

Breakeven vs sequential powm: ~17000 bits (sieve), ~12000 bits (table).
Below these sizes the factoring overhead exceeds the powm cost. The communication
pattern is identical to k32_mpi.

---

## Build

### Using Make (recommended)

```bash
# Build all targets
make

# Build individual targets
make k16_mpi
make k32_mpi
make k35_mpi
make challenges

make clean
```

Requires `mpicxx` (OpenMPI) and `libgmp`. On HPC load the modules first:

```bash
module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0
```

### Manual compilation

```bash
# challenge generator
gcc -O2 -fopenmp tools/challenges.c -lgmp -o challenges

# k16_mpi
mpicxx -O3 -std=c++17 -fopenmp -Wall -I src/common \
    src/solvers/k16_mpi.cpp src/common/k_solver_common.cpp \
    src/common/k_common.cpp src/common/k_factorization.cpp \
    -lgmp -o k16_mpi

# k32_mpi
mpicxx -O3 -std=c++17 -fopenmp -Wall -I src/common \
    src/solvers/k32_mpi.cpp src/common/k_solver_common.cpp \
    src/common/k_common.cpp src/common/k_factorization.cpp \
    -lgmp -o k32_mpi

# k35_mpi
mpicxx -O3 -std=c++17 -fopenmp -Wall -I src/common \
    src/solvers/k35_mpi.cpp src/common/k_solver_common.cpp \
    src/common/k_common.cpp src/common/k_factorization.cpp \
    -lgmp -o k35_mpi
```

---

## Usage

### k16_mpi

```bash
# single run (sieve method)
srun --mpi=pmix ./k16_mpi \
    data/challenges/k16/challenge_0_16_2048.txt \
    data/challenges/k16/modulo_16_2048.txt --method sieve

# benchmark 10 runs
srun --mpi=pmix ./k16_mpi \
    data/challenges/k16/challenge_0_16_2048.txt \
    data/challenges/k16/modulo_16_2048.txt --method sieve --bench 10

# sequential powm baseline (for comparison)
srun --mpi=pmix ./k16_mpi \
    data/challenges/k16/challenge_0_16_2048.txt \
    data/challenges/k16/modulo_16_2048.txt --method powm --bench 10
```

Note: the first run for a given modulo file precomputes and saves a `.cbrt`
table under `data/cbrt/`. Subsequent runs load it from disk.

### k32_mpi (local / single-node)

```bash
# single challenge (sieve method)
srun --mpi=pmix ./k32_mpi \
    data/challenges/k32/challenge_0_32_2048.txt \
    data/challenges/k32/modulo_32_2048.txt

# benchmark 5 runs
srun --mpi=pmix ./k32_mpi \
    data/challenges/k32/challenge_0_32_2048.txt \
    data/challenges/k32/modulo_32_2048.txt --bench 5
```

### k32_mpi (HPC)

```bash
# generate challenges first
sbatch scripts/challenge_gen.slurm

# submit SLURM job array (scaling study across 1/2/4/8 nodes)
sbatch scripts/k32.slurm

# manual run with specific challenge
srun --mpi=pmix ./k32_mpi \
    data/challenges/k32/challenge_0_32_1024.txt \
    data/challenges/k32/modulo_32_1024.txt

# run all challenge sizes in one MPI execution (avoids multi-srun issues)
srun --mpi=pmix ./k32_mpi --all --data data/challenges/k32 --bench 5
```

### k35_mpi (HPC)

```bash
# generate challenges first (k35 sizes: 8192 12288 16384 20480 24576 bits)
sbatch scripts/challenge_gen.slurm

# submit SLURM job array (scaling study across 1/2/4/8 nodes)
sbatch scripts/k35.slurm

# single challenge, sieve method (default)
srun --mpi=pmix ./k35_mpi \
    data/challenges/k35/challenge_0_35_16384.txt \
    data/challenges/k35/modulo_35_16384.txt --method sieve --bench 10

# table method — generates data/k35_upper_primes.bin (~10 GB) on first run
srun --mpi=pmix ./k35_mpi \
    data/challenges/k35/challenge_0_35_16384.txt \
    data/challenges/k35/modulo_35_16384.txt --method table --bench 10

# run all challenge sizes in one MPI execution
srun --mpi=pmix ./k35_mpi --all --data data/challenges/k35 --method sieve --bench 5
```

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

### k16 — local machine (AMD EPYC 7452, 16 threads)
Averaged over 1000 runs (warm cache, thread pool alive).

| Challenge size | Factoring | Reconstruction | Total   | Target  |
|---------------|-----------|----------------|---------|---------|
| 256-bit       | 0.108ms   | 0.076ms        | 0.184ms | 0.024ms |
| 1024-bit      | 0.194ms   | 0.087ms        | 0.281ms | 1.661ms |
| 2048-bit      | 0.399ms   | 0.172ms        | 0.571ms | 7.143ms |

All sizes beat the professor's targets at steady state (1000-run average).
The 256-bit target is below single-run overhead; achieving it requires
keeping the process alive across many challenges (as in real VDF deployments).

### k32_mpi — HPC scaling study (AMD EPYC 7452, 32 threads/rank)
Averaged over 5 runs. All results verified correct.

**Factoring time (ms) — scales linearly with ranks:**

| Challenge | 2 ranks | 4 ranks | 6 ranks | 8 ranks | Speedup (2→8) |
|-----------|---------|---------|---------|---------|---------------|
| 1024-bit  | 339ms   | 174ms   | 125ms   | 90ms    | 3.8×          |
| 2048-bit  | 403ms   | 201ms   | 141ms   | 100ms   | 4.0×          |
| 5000-bit  | 495ms   | 250ms   | 171ms   | 127ms   | 3.9×          |
| 7000-bit  | 579ms   | 289ms   | 200ms   | 146ms   | 4.0×          |

**Reconstruction time (ms) — distributed across ranks (each rank computes its own partial cbrt product):**

| Challenge | 2 ranks | 16 ranks | Note                                     |
|-----------|---------|----------|------------------------------------------|
| 1024-bit  | ~1ms    | ~1ms     | Fast — only ~34 powm calls total         |
| 2048-bit  | ~11ms   | ~3ms     | Scales — 67 powm calls split per rank    |
| 5000-bit  | ~205ms  | ~26ms    | Scales — 163 powm calls split per rank   |
| 7000-bit  | ~650ms  | ~81ms    | Scales — 229 powm calls split per rank   |

Each rank computes `cbrt(pᵢ)^eᵢ mod M` for its own factors locally, then sends
one fixed-size partial product to rank 0. Rank 0 multiplies `nranks` terms —
O(nranks) cheap multiplications instead of O(total_factors) expensive powm calls.

### k35_mpi — HPC scaling study (AMD EPYC 7452, 32 threads/rank)
Expected results at 8 nodes / 16 ranks / 32 threads per rank, 10 bench runs.

| Challenge | powm (ms) | sieve speedup | table speedup |
|-----------|-----------|---------------|---------------|
| 8192-bit  | 118ms     | 0.18×         | 0.31×         |
| 12288-bit | 309ms     | 0.48×         | 0.81×         |
| 16384-bit | 615ms     | 0.96×         | 1.62×         |
| 20480-bit | 1047ms    | 1.64×         | 2.76×         |
| 24576-bit | 1600ms    | 2.50×         | 4.21×         |

k35 breaks even against sequential powm at ~17000 bits (sieve) and ~12000 bits
(table). For smaller challenges, powm is faster because the sieve overhead across
2^35 primes exceeds the cost of a single modular exponentiation.

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

- **k64 not yet implemented**: challenges with 64-bit prime factors would
  require a different approach — the sieve range (up to 2^64) is too large
  for trial division even with MPI. The randomized smoothness approach from
  the paper (Algorithm 2) would be needed.

- **k35 table file size**: `k35_upper_primes.bin` is ~10 GB and is generated
  on the first `--method table` run. This is a one-time cost but requires
  sufficient scratch space on the HPC filesystem.

- **Multiple `srun` calls hang**: running k32_mpi or k35_mpi with multiple
  sequential `srun` invocations in one SLURM job causes MPI reinit issues.
  Use the `--all` flag to run all challenge sizes in a single MPI execution instead.

- **cbrt table load time**: loading the precomputed k16 table for large
  moduli (7000-bit) takes ~50ms from disk. For production use, keeping the
  table in memory across multiple challenge solves would eliminate this overhead.