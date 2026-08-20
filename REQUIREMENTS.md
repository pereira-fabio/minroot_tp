# Requirements

What you need before `make`, and what you can realistically run once you have it.

The short version: **an MPI compiler, GMP, and OpenMP.** Everything else is
optional — CUDA only for the GPU targets, Python only for the plots, SLURM only
for the cluster sweeps. The CPU code builds and runs on an ordinary laptop.

---

## 1. Required — the CPU solvers

| | Used for | Minimum | Used in this study |
|---|---|---|---|
| C++17 compiler | all solvers | GCC 7 / Clang 5 | GCC 13.2.0 |
| MPI implementation | `mpicxx`, rank distribution | MPI 2.0 (`MPI_IN_PLACE`) | Open MPI 4.1.6; MPICH 4.3.2 also verified |
| GMP | all multi-precision arithmetic | 5.0 | 6.3.0 |
| OpenMP | per-rank threading | 3.1 (GCC 4.7) | OpenMP 5.2 |
| GNU make, `gcc` | build, and the C tools | any | — |

The code declares `-std=c++17` but uses no C++17 library features (no
`std::filesystem`, no structured bindings), and every GMP call is long-stable
API, so older toolchains than the ones above will very likely work. The versions
in the right-hand column are what the reported results were produced with; the
whole tree has also been built warning-free on GCC 15.2.0, three major versions
newer.

### A broken MPI install looks like a crash in this code

Observed on Ubuntu 25.10 with its Open MPI 5.0.10 packaging: every MPI program
dies with `Illegal instruction` (exit 132) inside `MPI_Init`, before any of its
own code runs. The symptom is easy to misread, because

- the process produces **no output at all**, so it looks like the solver failed
  at startup — and under `mpirun` stdout is a pipe, so even output that *was*
  written is lost when the process dies by signal (use `stdbuf -o0` to see it);
- `mpirun -n 2 hostname` still works, because `hostname` never calls `MPI_Init`;
- the error text is `prun:proc-aborted-strsignal` alongside a complaint about a
  missing help file, which names no cause.

The tell is that `mpirun --version` also fails, and that the help files are
missing from both the PMIx and PRRTE directories — Open MPI 5.x is built against
external PMIx and PRRTE on Debian and Ubuntu, and that combination is where this
went wrong.

**MPICH is a drop-in fix.** This code uses only MPI 2.0 calls, so nothing here
depends on which implementation you use:

```bash
sudo apt install mpich libmpich-dev
sudo update-alternatives --set mpi /usr/bin/mpicc.mpich
sudo update-alternatives --set mpirun /usr/bin/mpirun.mpich
make clean && make
```

Verified working with MPICH 4.3.2 and GCC 15.2.0. None of this affects the
clusters, where Open MPI 4.1.6 is loaded as a module and is what every published
result was measured with.

### Install

**Debian / Ubuntu**

```bash
sudo apt update
sudo apt install -y build-essential libgmp-dev libopenmpi-dev openmpi-bin
```

**Fedora / RHEL**

```bash
sudo dnf install -y gcc-c++ make gmp-devel openmpi-devel
module load mpi/openmpi-x86_64        # RHEL puts mpicxx behind a module
```

**macOS (Homebrew)**

```bash
brew install gmp open-mpi libomp
```

Apple Clang does not ship OpenMP. The simplest route is to have Open MPI's
wrapper call Homebrew GCC instead:

```bash
brew install gcc
OMPI_CXX=g++-14 make          # OMPI_CXX picks the compiler behind mpicxx
```

Alternatively keep Clang and add `-Xpreprocessor -fopenmp -lomp`, pointing at
Homebrew's `libomp`. Note that `make CXX=...` will not do it: the MPI targets
go through the wrapper. Use `make MPICXX=...` to substitute a different wrapper
entirely.

**HPC with environment modules** — the combination this study used:

```bash
module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0
```

### Verify

```bash
mpicxx --version                                  # must exist
echo '#include <gmp.h>' | cpp -o /dev/null        # GMP headers on the path
make && ls alg2_k16 alg2_k32 alg3 alg4 alg7
```

If GMP is somewhere non-standard, pass it through:

```bash
make INCLUDE="-I src/common -I experiments/delta -I /opt/gmp/include" \
     LIBS="-L /opt/gmp/lib -lgmp"
```

---

## 2. Optional — the GPU solvers

| | Minimum | Used in this study |
|---|---|---|
| CUDA toolkit (`nvcc`) | 11.0 (sm_70) · 11.8 for sm_89 | 12.6.0 |
| NVIDIA GPU | compute capability 7.0 | V100 (sm_70), L40 (sm_89) |

```bash
module load system/CUDA/12.6.0
make gpu                  # V100, sm_70 (the default)
make gpu ARCH=sm_89       # L40
make gpu ARCH=sm_80       # A100
```

`ARCH` must match your card or the binaries will not launch. `nvidia-smi
--query-gpu=compute_cap --format=csv` tells you what to pass.

Nothing in the CPU build depends on CUDA — `make` and `make gpu` are
independent, and you can ignore the GPU half entirely.

---

## 3. Optional — analysis and plots

Only the five plotting scripts need third-party packages; the other five in
`analysis/` are pure standard library.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
```

Python 3.9 or newer. Do not commit the virtualenv — `.gitignore` excludes
`analysis/venv/` and `.venv/`.

| Needs numpy + matplotlib | Standard library only |
|---|---|
| `plot_results.py` | `cache_residency.py` |
| `ploter.py` | `crt_best_alpha.py` |
| `thesis_figures.py` | `effective_throughput.py` |
| `plot_alg4_comparison.py` | `ideal_combine.py` |
| `plot_dickman.py` (matplotlib only) | `projection.py` |

---

## 4. Optional — the cluster sweeps

`scripts/` and the experiment job scripts are SLURM batch files. They assume
environment modules, `srun --mpi=pmix`, and are submitted **from the repository
root** (each one `cd`s to `$SLURM_SUBMIT_DIR` and uses paths relative to it).

Without SLURM you can still run every solver directly with `mpirun`; you just
drive the sweep yourself.

---

## 5. Disk and memory

Most of the footprint is precomputed prime tables, and they are all
regenerable — none is in the repository, and `.gitignore` excludes them.

| | Size | Needed for |
|---|---|---|
| `data/k32_primes.bin` | 813 MB | `--method table` at B = 2^32 |
| `data/k35_upper_primes.bin` | ~9.6 GB | `--method table` at B = 2^35 |
| `data/k32_primes.dlt` | ~204 MB | the delta experiment |
| `data/k35_upper_primes.dlt` | ~1.2 GB | the delta experiment |
| challenges + moduli | a few MB per size | everything |
| cbrt caches (`data/cbrt/`) | tens of MB | steady-state reconstruction |

**RAM.** The k32 table is loaded per rank as a slice, so a node needs roughly
813 MB divided by ranks-per-node, plus the challenge and working values. k35's
9.6 GB upper table is the one real constraint: `--method table` at B = 2^35 is
not a laptop workload. `--method sieve` needs no table at all and is the way to
try k35 on a small machine.

**A note on what is deliberately absent.** A full cube-root table for B = 2^32
would be ~52 GB and for B = 2^35 ~624 GB. That is why reconstruction computes
roots on the fly with a cache rather than looking them up, and it is a result in
its own right, not an oversight.

---

## 6. Smallest thing that proves the install works

No cluster, no GPU, about a minute:

```bash
make challenges alg2_k16

# 3 challenges, 1024-bit modulus, B = 2^16
# -> data/challenges/k16/{challenge,factors}_{0,1,2}_16_1024.txt + modulo_16_1024.txt
./challenges 3 1024 16

mpirun -n 2 ./alg2_k16 \
    data/challenges/k16/challenge_0_16_1024.txt \
    data/challenges/k16/modulo_16_1024.txt \
    --method sieve
```

Expect `SUCCESS: 3-th root is correct!`. That line is the correctness gate —
every timing in this project is only meaningful when it passes. Note that it is
printed, not enforced: a run that prints `FAILURE` still writes its JSON.

Paths are relative to the repository root — run from there, and pass the files
under `data/challenges/k16/`, not bare filenames.

### If it aborts

**Any missing input file aborts the job**, via `MPI_Abort`. Some MPI builds
report that as `prun:proc-aborted-strsignal`, often alongside a complaint about
a missing help file, which tells you nothing about the cause. Two things it is
usually not, and one it usually is:

1. **Check MPI itself first — and check it properly.** `mpirun -n 2 hostname`
   is not enough: `hostname` never calls `MPI_Init`, so it only proves the
   launcher runs programs. Compile a six-line MPI program instead:

   ```bash
   cat > /tmp/hello.c <<'EOF'
   #include <mpi.h>
   #include <stdio.h>
   int main(int argc, char **argv) {
       MPI_Init(&argc, &argv);
       int r; MPI_Comm_rank(MPI_COMM_WORLD, &r);
       printf("rank %d alive\n", r);
       MPI_Finalize();
       return 0;
   }
   EOF
   mpicc /tmp/hello.c -o /tmp/hello && mpirun -n 2 /tmp/hello
   ```

   If *that* aborts, no solver here can run and the fault is the MPI
   installation. See "A broken MPI install looks like a crash in this code"
   below.
2. **`--all` needs every canonical size.** It iterates
   {256, 384, 512, 768, 1024, 2048, 4096, 8192, 16384, 24576} and aborts on the
   first one it cannot find, so generating only 1024 bits is not enough:

   ```bash
   for S in 256 384 512 768 1024 2048; do ./challenges 3 $S 16; done
   mpirun -n 2 ./alg2_k16 --all --data data/challenges/k16 \
       --n-challenges 3 --method sieve
   ```

   The four largest sizes take substantially longer to generate; leave them out
   unless you want them.
3. **A stale `data/cbrt/` cache.** The cube-root cache is keyed by modulus
   filename, not by content, so regenerating a modulus at the same size leaves a
   cache that no longer matches. `rm -rf data/cbrt` if results look wrong.

The single-challenge form above has been run under AddressSanitizer at one rank
and is clean, so an abort there points at inputs or the MPI installation rather
than at the solver.

k16 is the right smoke test because its prime table is 6542 primes (~26 KB) and
needs no precomputation. To go further locally:

```bash
make alg2_k32
./challenges 3 1024 32
mpirun -n 2 ./alg2_k32 \
    data/challenges/k32/challenge_0_32_1024.txt \
    data/challenges/k32/modulo_32_1024.txt \
    --method sieve          # sieve needs no 813 MB table
```

Use `--method table` only after building the table once with
`mpirun -n 1 ./alg2_k32 --precompute`.

**Do not expect a speedup at these sizes.** At 1024 bits every large-bound
variant is far below 1×: the sieve cost cannot be amortised against a
sub-millisecond baseline. That is a result, not a broken build — see the two
regimes in `README.md`.

---

## 7. Reproducing the published numbers

The figures in `README.md` were measured on 64 nodes / 128 MPI ranks (AMD EPYC
7452) and on 4× V100 / 4× L40. You cannot reproduce them on one machine: they
are strong-scaling measurements, and the design point is 24576-bit moduli.

What you *can* reproduce locally is correctness at every size, and the shape of
the scaling curve up to your own core count. The two-regime result holds too,
though the crossover moves: the ~3000-bit figure quoted in `README.md` is for
k32 at 128 ranks, and with fewer ranks it sits higher.
