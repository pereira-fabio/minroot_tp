#pragma once

/* Solve a puzzle using the smoothness-based attack (single-threaded).
 *
 * For each of the T iterations the attack:
 *   1. Blinds the current challenge: y = x * r^e mod p
 *   2. Trial-divides y against a precomputed prime table up to B
 *   3. If y is B-smooth, reconstructs y^(1/e) from precomputed cbrt table
 *   4. Unblinds to recover x_{i+1} = y^(1/e) * r^{-1} mod p
 *   5. Verifies result^e mod p == x_i; retries with a new r if not smooth
 *
 * B (smoothness bound) controls the tradeoff:
 *   - Larger B  -> higher hit rate  -> fewer blinding attempts per step
 *                  but larger cbrt table and more trial division work
 *   - Smaller B -> lower hit rate   -> more retries per step
 *
 * Recommended values:
 *   64-bit  prime : B = 65536  (u=4, ~1 in 200 attempts — fast demo)
 *   128-bit prime : B = 65536  (u=8, ~1 in 31M attempts — slow but works)
 *   256-bit prime : needs B >= 2^32 for practical hit rate (not in this tool)
 *
 * Returns 0 on success. */
int cmd_attack(const char *puzzle_path, unsigned long B);
