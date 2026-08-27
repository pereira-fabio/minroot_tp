# Live time-lock demo for the defence

```bash
cd tools/timelock/demo
./setup.sh          # builds, calibrates on this machine, generates both puzzles
./compare.sh        # THE DEMO: same search, serial vs batched on the GPU
```

No code needs writing — `tools/timelock` already has `generate` / `solve` /
`attack`, plus the CUDA batched search in `attack_cuda.cu`. This is calibration
and a runner.

---

## The centrepiece: `./compare.sh`

The same smoothness search against the same 64-bit puzzle, twice:

| | what it does | parallelism |
|---|---|---|
| `minroot-timelock attack` | draw `r`, blind, trial-divide, retry — one candidate at a time | **none**, single thread |
| `timelock_gpu` | generate a whole batch, test them concurrently, take the first hit | **CUDA**, `--batch` candidates per launch |

Both must decrypt the same plaintext, so it is a like-for-like wall-clock
comparison. The script prints a table and the speedup.

For reference, the thesis measures **70.3 s serial against 9.9 s batched** at
`T = 5000` — about **7×** — on an L40S. `setup.sh` sizes the puzzle so the
serial half lands near 30 s on *your* machine (`ATTACK_SECS=60 ./setup.sh` for
a longer run).

### What this actually demonstrates

**Parallelised — the search (phase 1).** Blinding candidates are mutually
independent, so the loop inverts cleanly onto thousands of device threads. This
is the one phase the rest of the project never measures, because the benchmark
solvers are handed a challenge that is smooth by construction and never enter
the retry loop at all.

**Not touched — the chain.** The `T` rounds stay strictly sequential in both
runs. Neither shortens the dependency chain, because nothing can. That is
exactly what MinRoot claims, and it holds.

That contrast is the point of running both. The attack parallelises the lottery,
not the delay.

### Batch size is a real trade-off, and it is visible

Each round needs exactly **one** smooth candidate, but a batch tests `N` at
once, so the search overshoots by roughly `N·ρ(u)`. The tool reports that
overshoot as a measured column. The optimum is the smallest batch that
reliably hits on the first launch — `b·ρ ≳ 5`, so `b ≳ 880` here, and 1024
sits almost exactly on it. Worth showing live:

```bash
BATCH=256  ./compare.sh     # too small: a second launch on ~23% of rounds
BATCH=4096 ./compare.sh     # too big: ~93 hits found per round, 1 consumed
```

Raw throughput scales essentially perfectly with batch size — 61.7× from 256 to
16384 — and buys **1.24×** of wall clock, after which the curve turns *upward*.
The run is latency-bound on the `T` dependent rounds, not throughput-bound.

### It is also the only measured hit rate in the project

The kernel counts *every* smooth candidate in a batch rather than stopping at
the first, so each launch is an unbiased sample of the density. Every other hit
rate in the thesis is Dickman's ρ(u); this one is observed:

| | 1 in |
|---|---|
| asymptotic ρ(4) | 204 |
| measured, GPU, four batch sizes | 176.4 / 175.2 / 175.7 / 175.9 |
| measured, serial CPU | 176.6 |

Two independent measurements agreeing to 0.4%. Both sit above the asymptotic
because at finite size the smooth density exceeds ρ(u) by about
`(1 + γu/ln B) ≈ 1.21`.

---

## The other half: `puzzle_delay.mrt`

Start this before you begin speaking, in a visible terminal:

```bash
./build/minroot-timelock solve puzzle_delay.mrt
```

1024-bit, `T` calibrated so the honest solve takes the length of the talk. A
progress line every 5% with elapsed and ETA.

**It is single-threaded, and that is the message.** The chain is `T` dependent
roots; if it could be parallelised, MinRoot would not be a VDF. Say so while it
runs — a serial progress bar grinding through the talk is the strongest
statement of what the construction claims.

The generator, by contrast, builds the same puzzle in seconds by walking the
chain *backwards* — one `e`-th power per round instead of one `e`-th root. At
1024 bits that is a ~400× per-round asymmetry, with **no trapdoor**: MinRoot has
no setup secret that lets even the setter skip ahead.

| bits | forward (root) | backward (power) | ratio |
|-----:|---------------:|-----------------:|------:|
| 256  | 68.6 µs        | 0.71 µs          | 97×   |
| 512  | 362 µs         | 1.65 µs          | 220×  |
| **1024** | **2.13 ms** | **5.32 µs**      | **400×** |
| 2048 | 14.4 ms        | 18.7 µs          | 768×  |

(Indicative, measured in Python; GMP is several times faster, which is why
`setup.sh` probes your machine rather than trusting this table.)

---

## Why two puzzles and not one

I looked for a single puzzle where the attack beats the honest solver live.
**There isn't one, and that is the thesis result rather than a limitation.**

The smoothness search only terminates when `u = |p| / log2(B)` is small — at
`B = 2^16` that caps the modulus at 64 bits. But at 64 bits an honest round is
microseconds, so the honest solver wins outright: the thesis measures 0.002 s
honest against 9.9 s for the best GPU attack, **~4960× slower**. Push the
modulus up until the delay means something and `u` explodes: at 1024 bits a
candidate is smooth with probability `ρ(64) = 2^-437`, and no search ever ends.

> Puzzle A has to be 1024 bits for the delay to mean anything.
> Puzzle B has to be 64 bits for the search to terminate.
> They cannot be the same puzzle — and that gap is the result of this thesis.

Say the honest solver is faster on Puzzle B **before** anyone spots it. It is
Gap 1 in miniature and it is the most honest slide in the talk.

---

## Which algorithm, and at what rate

`attack.c` accepts only when the residual reaches 1 (or is itself a tabled
prime ≤ B, which is still fully B-smooth). No medium-prime cofactor, no
composite cofactor, no Pollard's rho, no pre-filter — so this is **Algorithm 2
at B = 2^16**, the k16 variant. Algorithms 3 and 4 are not implemented here.

At `|p| = 64`, `u = 4` and the acceptance rate is `ρ(4) ≈ 2^-7.7`, about 1 in
204. k16 is the variant with the worst hit rate everywhere else in the thesis
(`2^-69.6` at 256 bits); it is usable here purely because dropping to 64 bits
takes `u` from 16 down to 4.

---

## Practical notes

- **Calibrate on the machine you will present from.** Laptop power management
  can halve throughput on battery. Re-run `setup.sh` there.
- **`ARCH` defaults to `sm_75`** (Turing — 2080 Super). The IRIS scripts use
  `sm_70` for the V100s. Override with `ARCH=sm_70 ./setup.sh` if needed.
- **The GPU tool is 64-bit-modulus only** by design (no multi-limb device
  mulmod), so it refuses `puzzle_delay.mrt` rather than silently truncating.
  That refusal is a fine thing to point at if asked why reconstruction stays on
  the host.
- **`make` cannot handle a path containing a colon**, which the NAS mount has,
  so `setup.sh` copies sources to `build/` and compiles directly.
- Needs `libgmp-dev`, `libssl-dev`, and `nvcc` for the GPU half.
- If Puzzle A finishes early the ETA line still shows; if it runs late, say "it
  is still running, which is rather the point". Neither is a failure.

## Tuning

| variable | default | effect |
|---|---|---|
| `MINUTES` | 20 | target wall time for Puzzle A |
| `ATTACK_SECS` | 30 | target wall time for the serial half of `compare.sh` |
| `BITS` | 1024 | modulus for Puzzle A |
| `BATCH` | 1024 | candidates per GPU launch (`compare.sh`) |
| `ARCH` | sm_75 | CUDA target architecture |
| `SRC` | `..` | where the timelock sources live |

---
*Edit: 2026-08-27*
