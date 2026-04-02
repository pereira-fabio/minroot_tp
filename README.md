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
  k_common.h        shared declarations
  k_common.cpp      shared utilities: file I/O, sieve, cube root helpers
  k16.cpp           solver for challenges with 16-bit prime factors (single node)
  k32.cpp           solver for challenges with 32-bit prime factors (single node)
  k32_mpi.cpp       MPI+OpenMP hybrid solver for k32 (HPC, multi-node)

data/challenges/
  k16/              challenge and modulo files with 16-bit prime factors
  k32/              challenge and modulo files with 32-bit prime factors

job_k32.slurm       SLURM job script for k32_mpi on HPC
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

Distributes the sieve range `[65537, 2^32]` across MPI ranks. Each rank handles
a contiguous slice and finds factors independently. Rank 0 gathers and reconstructs.

**Communication pattern (minimal by design):**
- `MPI_Bcast`: challenge + modulo bytes → all ranks (once per challenge)
- `MPI_Gather`: factor counts → rank 0 (tiny)
- `MPI_Gatherv`: found factors → rank 0 (~34 uint64 pairs for 1024-bit)
- No communication during the hot trial division loop

---

## Build

### Local (single node)

```bash
# k16
g++ -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k16.cpp src/k_common.cpp -lgmp -o k16

# k32
g++ -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k32.cpp src/k_common.cpp -lgmp -o k32
```

### HPC (requires modules)

```bash
module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

# k32_mpi
mpicxx -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k32_mpi.cpp src/k_common.cpp -lgmp -o k32_mpi
```

---

## Usage

### k16

```bash
# single challenge, factor method
./k16 data/challenges/k16/challenge_0_16_2048.txt \
      data/challenges/k16/modulo_16_2048.txt --method factor

# benchmark 20 runs
./k16 data/challenges/k16/challenge_0_16_2048.txt \
      data/challenges/k16/modulo_16_2048.txt --method factor --bench 20

# naive Fermat baseline (for comparison)
./k16 data/challenges/k16/challenge_0_16_2048.txt \
      data/challenges/k16/modulo_16_2048.txt --method powm
```

Note: the first run for a given modulo file will compute and save a `.cbrt`
precomputed table. Subsequent runs load it from disk.

### k32

```bash
# single challenge
./k32 data/challenges/k32/challenge_32_2048.txt \
      data/challenges/k32/modulo_32_2048.txt

# benchmark 5 runs
./k32 data/challenges/k32/challenge_32_2048.txt \
      data/challenges/k32/modulo_32_2048.txt --bench 5
```

### k32_mpi (HPC)

```bash
# submit SLURM job (runs all 4 challenge sizes, 5 bench runs each)
sbatch job_k32.slurm

# manual run with specific challenge
srun --mpi=pmix ./k32_mpi \
    data/challenges/k32/challenge_32_1024.txt \
    data/challenges/k32/modulo_32_1024.txt

# run all challenge sizes in one MPI execution (avoids multi-srun issues)
srun --mpi=pmix ./k32_mpi --all --data data/challenges/k32 --bench 5
```

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

**Reconstruction time (ms) — rank 0 only, does not scale with MPI ranks:**

| Challenge | All configs | Note                              |
|-----------|-------------|-----------------------------------|
| 1024-bit  | ~1ms        | Fast — only 34 powm calls         |
| 2048-bit  | ~11ms       | Fast — only 67 powm calls         |
| 5000-bit  | ~205ms      | Bottleneck — 163 × 5000-bit powm  |
| 7000-bit  | ~650ms      | Bottleneck — 229 × 7000-bit powm  |

For 7000-bit challenges the reconstruction (650ms) dominates total time regardless
of how many nodes are used for factoring. Distributing reconstruction across ranks
is the natural next optimization.

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

- **Reconstruction not distributed**: for large moduli (5000+ bits), the
  `powm` calls on rank 0 dominate total time. Distributing these across ranks
  with a final `MPI_Reduce` product would improve scaling for large challenges.

- **k64 not yet implemented**: challenges with 64-bit prime factors would
  require a different approach — the sieve range (up to 2^64) is too large
  for trial division even with MPI. The randomized smoothness approach from
  the paper (Algorithm 2) would be needed.

- **Multiple `srun` calls hang**: running k32_mpi with multiple sequential
  `srun` invocations in one SLURM job causes MPI reinit issues. Use `--all`
  flag to run all challenge sizes in a single MPI execution instead.

- **cbrt table load time**: loading the precomputed k16 table for large
  moduli (7000-bit) takes ~50ms from disk. For production use, keeping the
  table in memory across multiple challenge solves would eliminate this overhead.