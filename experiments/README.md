# Experiments

Code in this directory is **not part of the attack**. It is the research
scaffolding: questions that came up while implementing Biryukov et al.
(CRYPTO 2024), the code written to answer them, and — in two cases — the
negative results that answered them.

It lives here rather than in `src/` so that `src/` contains only the six
algorithms the thesis measures. Everything here still builds and runs:

```bash
make experiments                # CPU: alg_crt, cpu_scale_sim, delta_convert, prefilter_bench
make gpu-experiments            # GPU: alg_crt_cuda, alg2_ksweep_cuda, gpu_scale_sim_cuda
```

Each directory holds its own SLURM job script, so an experiment is code, job and
write-up in one place. Submit from the repository root:

```bash
sbatch experiments/ksweep/ksweep.slurm
```

`scripts/` holds only the sweeps that reproduce the thesis results.

---

## `delta/` — delta-encoded prime table

**Question.** k16 wins on the GPU and k32 does not. The obvious explanation is
cache residency: k16's 26 KB prime table fits in L1, while k32's 813 MB table
misses every level. If that is the cause, shrinking the table below the L2
boundary should make k32 fast.

**Method.** Store gaps instead of values. Every prime gap below 2^35 is at most
456, so `gap/2 <= 228` fits in one byte: 1 byte per prime instead of 4, a 3.99x
reduction, which puts a 4-way GPU slice at ~51 MB — inside an L40's ~96 MB L2.
Blocks of 1024 primes re-anchor, so random block access and the per-rank range
split both survive. The GPU kernel decodes a block into shared memory with a
single thread, then trial-divides exactly as the flat kernel does.

**Result: negative.** The 3.99x smaller table buys **1.10x**. Trial division is
bound by big-integer arithmetic, not by memory traffic, so making the table
resident does almost nothing. A companion experiment — splitting one table
across N devices, which moves the footprint across the L2 boundary while
holding the prime count fixed — found total GPU work flat to 6.7% with no
feature at the crossing. What actually sets the cost is the *number of primes*:
~2.8 ns each, flat from B=2^26 to 2^32.

Still exercised by `scripts/iris/all_gpu.slurm`, and reported in the thesis.

| file | role |
|---|---|
| `delta_table.h/.cpp` | format, converters, slice loader, CPU trial division |
| `delta_table_gpu.cuh` | CUDA decode kernels (4-byte and 8-byte anchors) |
| `delta_convert.cpp` | `delta_convert [--u64] <flat.bin> [out.dlt]` |

`delta_convert` replaces the old `--convert-delta` mode that four MPI solvers
each carried. It is a file-format conversion and needs neither MPI nor an
allocation.

---

## `ksweep/` — where the cache boundary actually sits

**Question.** If residency is not what separates k16 from k32, what does the
cost curve look like as the bound B moves between them?

**Method.** Build prime tables at intermediate k with `tools/gen_prime_table`,
then sweep B and measure. `analysis/cache_residency.py` predicts the crossover
from prime counts alone.

**Result.** The curve tracks the prime count, not the footprint — which is what
sank the residency hypothesis above. k16 is not fast because it is resident; it
is fast because 6542 primes is almost no work, and it pays for that in hit rate
(2^-69.6 at 256 bits).

---

## `scale_sim/` — strong-scaling simulators

64 nodes (128 MPI ranks) is a hard account limit on AION, and 4 GPUs on IRIS.
Several questions in the thesis need rank counts beyond that — k35 would need
roughly 1024 ranks to reach the paper's figure, Algorithm 7 around 530 nodes.
These simulators extrapolate from the measured N1–N64 curve.

Anything they produce is **projection, not measurement**, and the thesis labels
it as such.

---

## `crt/` — CRT exponentiation

**Question.** Can the sequential exponentiation itself be parallelised, rather
than the sieve — splitting the modulus by CRT and exponentiating each residue
independently?

**Result: negative.** Correct at every size and alpha tested, and never close to
the baseline; it peaks at **0.167x**. The failure is structural, not an
implementation problem: a rank-invariant term of about 1800 ms — work done once
per round on *every* rank instead of divided among them — equals the sequential
baseline on its own. Free reconstruction and free residues would still cap the
method at 0.999x. The alpha-ary loop parallelises the squaring half of
square-and-multiply and replicates the multiplying half, and a method that
parallelises half an algorithm is bounded by the half it kept.

---

## `prefilter/` — is the pre-filter worth its cost?

**Question.** Before sieving a candidate, you can spend a little time choosing a
*better* candidate: strip its small factors up to some B0 and discard it if the
residue is too large. That costs `C_pf` per candidate and saves sieve work by
raising the chance the candidate is smooth. When does it pay?

**Method.** `prefilter_bench` measures `C_pf` directly across moduli and rank
counts, against a sieve cost `C_sieve` imported from the Algorithm 2 k32 batch —
a different job, so the two are not measured together.

**Result.** It pays when `C_pf < (2^b - 1) * C_sieve`, where `b` is the bit
saving the filter buys. The counter-intuitive part is the direction: **larger
moduli benefit more**, not less. The more expensive the sieve, the more you can
afford to spend choosing what to feed it.

`C_pf` turns out not to depend on which sieve it is paired with, which is what
makes importing `C_sieve` from another batch legitimate.

**Note on status.** The selection primitive here (`prefilter_simple_select`) is
called by **no solver** — Algorithm 4 carries its own inline pre-filter
(`ALG4_B0`, `b0_residue_bits`), and there is no `--prefilter` flag. This code
exists because it produced the thesis figures; wiring it into the bench loop was
never done.

---

## `reconstruction/` — could a device port of reconstruction pay?

**Question.** k16 factors ~43,000x faster on a GPU than the sequential baseline
and still only reaches 11.7x end to end, because reconstruction runs on the host.
Would writing multi-precision arithmetic for the device fix that?

**Method.** `k32_instrumented.slurm` runs one job that answers three things at
once, on k32 only — deliberately, because k32 has the *cheapest* recon share
(30.8% of total at 24576 bits, an Amdahl ceiling of 1.45x), so a negative answer
there generalises downward to every variant with a cheaper sieve and thus a
higher recon share.

| output | what it measures |
|---|---|
| `CBRTSTAT\|` | cbrt cache hits/misses/probes per thread |
| `RECONPROF\|` | reconstruction split into per-factor cbrt and combine (product mod M; `serial_ms`, now a parallel tree) |
| `XFERPROF\|` | H2D/D2H transfer cost, which the `factor_ms` event bracket excludes |

**Result: no.** In the warm steady state the cache hits ~100%, so reconstruction
performs no exponentiation at all — it is cache lookups plus a product of ~800
residues mod M, and a device port accelerates neither.

**Follow-up: the product parallelises on the host.** The original write-up
called the serial column unparallelisable; it is not. Multiplication mod M is
associative, so `kf_product_mod_tree` splits it into per-thread chunk products
combined pairwise — n/T + log2 T dependent mulmods instead of n-1 — on the
host cores the GPU jobs already reserve (28 on IRIS). Measured off-cluster on 4
cores at 24576 bits, warm cache: 62 -> 19 ms for 802 factors, 124 -> 40 ms for
1536, identical results. On a desktop RTX 2080 SUPER (8 host cores) the host
tree brought k32 reconstruction to 1.09 ms at 8192 bits; not yet re-measured on
IRIS.

**Follow-up: and on the device.** `--recon device` runs the same tree on the
GPU (`src/common/gpu/gpu_recon.cuh`): one thread block per Montgomery
multiplication (`gpu_mont.h`), one kernel launch per tree level, ceil(log2 n)+2
launches in all. The block arithmetic is unit-tested against GMP by running the
identical code as host threads behind a barrier. Timed on an RTX 2080 SUPER
against the host tree on its 8 cores (warm recon_ms): slower at 2048 bits
(0.18 vs 0.075 ms, 66 terms: launch cost dominates), faster at 8192 bits
(k32 0.86 vs 1.09 ms, 268 terms; k16 1.57 vs 4.10 ms, 528 terms).
`--recon auto` (the default) picks between them by product size. Against the
28-core IRIS host it is still unmeasured. The instrumentation itself lives in
`src/common/factorization.cpp` and `src/common/gpu/gpu_common.h`, compiled into
every solver; this directory holds only the job that exercises it.

The cache lookup is also where the O(n^2) shows up: `kf_cbrt_cache_find` is a
linear scan, so ~802 factors against an ~802-entry cache is ~322,000 comparisons
per reconstruction.
