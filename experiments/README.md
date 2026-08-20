# Experiments

Code in this directory is **not part of the attack**. It is the research
scaffolding: questions that came up while implementing Biryukov et al.
(CRYPTO 2024), the code written to answer them, and — in two cases — the
negative results that answered them.

It lives here rather than in `src/` so that `src/` contains only the six
algorithms the thesis measures. Everything here still builds and runs:

```bash
make experiments                # CPU: alg_crt, cpu_scale_sim, delta_convert
make -C src/iris experiments    # GPU: alg_crt_cuda, alg2_ksweep_cuda, gpu_scale_sim_cuda
```

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
