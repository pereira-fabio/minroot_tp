#!/usr/bin/env python3
# merge_cbrt_cache.py — one-time merge of the split cbrt caches into one full
# cache PER VARIANT, per size.  (Advisor M1, 2026-07-16.)
#
# Why per-variant (not one cache for everyone): the prime generator
# find_prime_parallel() (tools/challenges.c) is a RACE — the first thread to a
# valid prime wins, non-deterministically — so each variant's independently
# generated modulo file is a DIFFERENT prime, even at the same size.  cbrt(p) =
# p^{e^-1} mod M therefore differs across variants and CANNOT be shared between
# them.  Within a variant, though, every node-count and rank reads the same
# cached modulo file (same M), so those split caches ARE mergeable — and that
# per-node-count / per-rank duplication (rebuilt from scratch every job via
# `rm -rf $CBRT_DIR`) is the real waste this collapses.
#
#   data/cbrt/<variant>_e<exp>/N<n>/<file>.cbrt*.<rank>   (existing split)
#        -> data/cbrt/<variant>_e<exp>/cbrt_<size>.cbrt   (one cache per variant+size)
#
# k16 is skipped (its dense table is a different, binary format).  A cbrt
# conflict WITHIN a variant (same prime, two values => that variant's modulo was
# regenerated) skips just that (variant,size) and exits non-zero, leaving the
# clean ones written.
#
# Cache file format (text, from kf_cbrt_cache_save):
#   line 1 : count
#   line i : <prime_decimal> <cbrt_decimal>
#
# Usage:   python3 analysis/merge_cbrt_cache.py [CBRT_ROOT]   (default data/cbrt)
#          python3 analysis/merge_cbrt_cache.py --dry-run
# Edit: 2026-07-16

import os, re, sys, glob

def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry = "--dry-run" in sys.argv
    root = args[0] if args else "data/cbrt"

    split = glob.glob(os.path.join(root, "*_e*", "N*", "*"))
    if not split:
        print(f"No split caches found under {root}/*_e*/N*/ — nothing to merge.")
        return

    groups = {}            # (vardir, size) -> { prime(str): cbrt(str) }
    gconflicts = {}        # (vardir, size) -> conflict count
    files_read = skipped_k16 = 0

    for path in split:
        if not os.path.isfile(path):
            continue
        fname = os.path.basename(path)
        vardir = os.path.basename(os.path.dirname(os.path.dirname(path)))  # <variant>_e<exp>
        if "k16" in vardir:      # dense binary table — different format, separate primes
            skipped_k16 += 1
            continue
        if not re.search(r'_e\d+$', vardir):
            print(f"  [skip] not a <variant>_e<exp> dir: {path}")
            continue
        ms = re.search(r'(\d+)(?:_e\d+)?\.cbrt', fname)   # size right before .cbrt
        if not ms:
            print(f"  [skip] cannot parse size from {path}")
            continue
        size = int(ms.group(1))
        key = (vardir, size)
        try:
            # errors="ignore": a stray binary file decodes to garbage rather than
            # raising; its first line then fails the isdigit() guard and is skipped.
            with open(path, encoding="utf-8", errors="ignore") as f:
                first = f.readline().strip()
                if not first.isdigit():   # not a text cbrt cache (count on line 1)
                    print(f"  [skip] {path}: first line is not a count (binary/foreign?)")
                    continue
                d = groups.setdefault(key, {})
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) != 2 or not parts[0].isdigit():
                        continue
                    prime, cbrt = parts[0], parts[1].strip()
                    if prime in d and d[prime] != cbrt:
                        gconflicts[key] = gconflicts.get(key, 0) + 1
                        if gconflicts[key] <= 3:
                            print(f"  [WARN] {vardir} {size}b: prime {prime} has two cbrt "
                                  f"values — this variant's modulo was regenerated?  {path}")
                    d[prime] = cbrt
            files_read += 1
        except OSError as e:
            print(f"  [skip] {path}: {e}")

    print(f"\nRead {files_read} split files across {len(groups)} (variant,size) groups"
          f"  (skipped {skipped_k16} k16 dense-table files).")

    written = skipped = 0
    for (vardir, size) in sorted(groups):
        d = groups[(vardir, size)]
        c = gconflicts.get((vardir, size), 0)
        out = os.path.join(root, vardir, f"cbrt_{size}.cbrt")
        if c:
            print(f"  [{vardir:16} {size:>6}b]  {c} CONFLICTS — SKIPPED (mixed moduli), not written")
            skipped += 1
            continue
        print(f"  [{vardir:16} {size:>6}b]  {len(d):>7} entries -> {out}"
              + ("   (dry-run)" if dry else ""))
        if not dry:
            with open(out, "w") as f:
                f.write(f"{len(d)}\n")
                for prime in sorted(d, key=int):
                    f.write(f"{prime} {d[prime]}\n")
        written += 1

    print(f"\n{'Would write' if dry else 'Wrote'} {written} per-variant caches"
          f"{f'; SKIPPED {skipped} with conflicts' if skipped else ''}.")
    if not dry and skipped == 0:
        print("Point each solver at data/cbrt/<variant>_e<exp>/cbrt_<size>.cbrt "
              "and stop the per-job `rm -rf` of the N<n> split dirs.")
    if skipped:
        sys.exit(1)   # flag that some (variant,size) had mixed moduli

if __name__ == "__main__":
    main()
