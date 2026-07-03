#pragma once

/* Solve a puzzle by computing T sequential e-th roots.
 * Prints progress every 5% and the decrypted message on success.
 * Returns 0 on success. */
int cmd_solve(const char *puzzle_path);
