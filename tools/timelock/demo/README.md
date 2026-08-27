# Live time-lock demo for the defence

**Short answer: yes, and no code needs writing.** `tools/timelock` already has
`generate` / `solve` / `attack`, and `solve` already prints a progress line
every 5% with elapsed time and ETA. It is purely a calibration job.

```bash
cd Thesis_defence/demo/timelock
./setup.sh                 # measures this machine, generates both puzzles
```

---

## The constraint you cannot design around

I looked for a single puzzle where the attack beats the honest solver live, on
one machine. **There isn't one, and that is your thesis result rather than a
limitation of the demo.**

The attack's smoothness search only terminates when `u = |p| / log2(B)` is
small — at `B = 2^16` that caps the modulus at 64 bits. But at 64 bits one
honest round is microseconds, so the honest solver wins by orders of magnitude.
Push the modulus up until the delay is meaningful and `u` explodes: at 1024
bits with the same bound, a candidate is smooth with probability `rho(64) =
2^-437`, and no search ever ends.

Your thesis measures this exactly: at 64 bits the honest solver finishes in
0.002 s while the best GPU attack takes 9.9 s — **the attack is ~4960× slower
than simply doing the work it is trying to avoid.**

So the demo is **two puzzles**, and the fact that they must be two is the
punchline.

---

## Puzzle A — the delay is real

1024-bit prime, `T` calibrated so the honest solve takes the length of the talk.
Start it in a visible terminal as you begin.

```bash
build/minroot-timelock solve puzzle_delay.mrt
```

**What to say when it finishes:** the generator built this puzzle in a few
seconds by walking the chain *backwards* — one `e`-th power per round, a
handful of multiplications. Solving it means walking *forwards* — one `e`-th
root per round, a full modular exponentiation. That asymmetry is the whole
construction, and there is no trapdoor: MinRoot has no setup secret that lets
even the setter skip ahead.

Indicative round costs (measured in Python, so GMP will be several times
faster — `setup.sh` measures your actual machine rather than trusting these):

| bits | forward (root) | backward (power) | ratio | T for 20 min | generation |
|-----:|---------------:|-----------------:|------:|-------------:|-----------:|
| 256  | 68.6 µs        | 0.71 µs          | 97×   | 17,502,528   | 12.4 s |
| 512  | 362 µs         | 1.65 µs          | 220×  | 3,315,383    | 5.5 s |
| **1024** | **2.13 ms** | **5.32 µs**      | **400×** | **564,107** | **3.0 s** |
| 2048 | 14.4 ms        | 18.7 µs          | 768×  | 83,617       | 1.6 s |

1024 bits is the sweet spot: generation in seconds, a T large enough that the
progress bar moves visibly, and a modulus that isn't a toy.

## Puzzle B — the attack is real

64-bit prime, `T = 100`. Runs in seconds at the demo slide.

```bash
build/minroot-timelock attack puzzle_attack.mrt   # blind, factor, reconstruct
build/minroot-timelock solve  puzzle_attack.mrt   # honest, for comparison
```

**What this establishes that the benchmarks cannot.** The benchmark chapter
inverts one root supplied as a challenge file and checks it algebraically. Here
the recovered root derives an AES key that decrypts a real ciphertext — so the
correctness gate is a working key recovery, not an identity. It is also the only
place in the whole project where the smoothness lottery is actually played:
the retry loop runs on genuinely random challenges and terminates, ~176 attempts
per round, matching `rho(4)`.

**Say the comparison out loud.** The honest solver is faster. Do not let anyone
catch that before you say it — it is Gap 1 in miniature, and it is the most
honest slide in the talk.

---

## The line that ties them together

> Puzzle A has to be 1024 bits for the delay to mean anything.
> Puzzle B has to be 64 bits for the search to terminate.
> They cannot be the same puzzle — and that gap is the result of this thesis.

---

## Practical notes

- **Start Puzzle A before you start speaking.** If it finishes early the ETA
  line still shows; if it finishes late, say "it is still running, which is
  rather the point" and move on. Neither is a failure.
- **Calibrate on the machine you will present from**, not your dev box. Laptop
  power management can halve throughput on battery. Re-run `setup.sh` there.
- **`make` cannot handle the NAS path** (it contains a colon), which is why
  `setup.sh` copies the sources to `build/` and compiles directly. The upstream
  `Makefile` is fine on a normal checkout.
- **Have a fallback.** If the live run misbehaves, `Thesis_defence/demo/` has a
  complete pre-recorded run in `fallback_4thread.{json,log}`.
- Needs `libgmp-dev` and `libssl-dev`. Not available on every machine — check
  before the day.

## Tuning

| variable | default | effect |
|---|---|---|
| `MINUTES` | 20 | target wall time for Puzzle A |
| `BITS` | 1024 | modulus for Puzzle A |
| `PROBE_T` | 300 | calibration probe length |
| `MSG_A` / `MSG_B` | see script | the revealed plaintexts |

```bash
MINUTES=15 BITS=2048 ./setup.sh
```

---
*Edit: 2026-08-27*
