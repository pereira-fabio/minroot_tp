#!/usr/bin/env python3
# merge_cbrt_cache.py — one-time merge of the split cbrt caches into one full
# cache per (size, exponent).  (Advisor M1, 2026-07-16.)
#
# The modulus is deterministic per (size, exponent) — find_prime_parallel(bits,
# e) in tools/challenges.c — so k32/k35/alg3/alg4 at a given (size, e) share the
# IDENTICAL M, and cbrt(p) = p^{e^-1} mod M is therefore interchangeable across
# variants.  This lets us collapse the per-variant / per-node-count / per-rank
# split caches into ONE full cache that every solver can read:
#
#   data/cbrt/<variant>_e<exp>/N<n>/<file>.cbrt*.<rank>   (existing split)
#        -> data/cbrt/cbrt_<size>_e<exp>.cbrt             (one merged full cache)
#
# The union of all the split files for a (size, e) is the complete prime set any
# variant needs (k35 contributes its 2^32..2^35 primes; alg3/alg4 their q0
# mediums; the small-prime overlap is stored once).  No cbrt is recomputed — this
# is pure file I/O over caches you already built.
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

    # Split caches live at:  <root>/<variant>_e<exp>/N<n>/<cachefile>
    split = glob.glob(os.path.join(root, "*_e*", "N*", "*"))
    if not split:
        print(f"No split caches found under {root}/*_e*/N*/ — nothing to merge.")
        return

    # (size, exp) -> { prime(str) : cbrt(str) }
    groups = {}
    conflicts = 0
    files_read = 0
    skipped_k16 = 0
    for path in split:
        if not os.path.isfile(path):
            continue
        fname = os.path.basename(path)
        vardir = os.path.basename(os.path.dirname(os.path.dirname(path)))  # <variant>_e<exp>
        # k16 uses a DIFFERENT format — a binary, prime-INDEXED dense table
        # (fwrite + mpz_out_raw, no prime values), and its primes (<=65536) do
        # not overlap the sieve solvers (which start at 65537).  Never merge it.
        if "k16" in vardir:
            skipped_k16 += 1
            continue
        me = re.search(r'_e(\d+)$', vardir)
        ms = re.search(r'(\d+)(?:_e\d+)?\.cbrt', fname)   # size right before .cbrt
        if not me or not ms:
            print(f"  [skip] cannot parse (size,exp) from {path}")
            continue
        exp = int(me.group(1)); size = int(ms.group(1))
        try:
            # errors="ignore": a stray binary file (e.g. a k16 dense table that
            # slipped the name check) decodes to garbage rather than raising —
            # its first line then fails the isdigit() guard below and is skipped.
            with open(path, encoding="utf-8", errors="ignore") as f:
                first = f.readline().strip()
                if not first.isdigit():   # not a text cbrt cache (count on line 1)
                    print(f"  [skip] {path}: first line is not a count (binary/foreign format?)")
                    continue
                d = groups.setdefault((size, exp), {})
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ', 1)
                    if len(parts) != 2 or not parts[0].isdigit():
                        continue                       # skip any non "<prime> <cbrt>" line
                    prime, cbrt = parts[0], parts[1].strip()
                    if prime in d and d[prime] != cbrt:
                        conflicts += 1     # would mean a MODULUS mismatch — must be 0
                        if conflicts <= 5:
                            print(f"  [WARN] prime {prime} has two cbrt values at "
                                  f"({size}b,e{exp}) — modulus mismatch?  file={path}")
                    d[prime] = cbrt
            files_read += 1
        except (OSError, UnicodeDecodeError) as e:
            print(f"  [skip] {path}: {e}")

    print(f"\nRead {files_read} split files across {len(groups)} (size,exp) groups"
          f"  (skipped {skipped_k16} k16 dense-table files — separate format).")
    if conflicts:
        print(f"  *** {conflicts} cbrt CONFLICTS — the caches came from different moduli. "
              f"Refusing to write. Investigate before using. ***")
        sys.exit(1)   # never write a merged cache built from mismatched moduli

    for (size, exp) in sorted(groups):
        d = groups[(size, exp)]
        out = os.path.join(root, f"cbrt_{size}_e{exp}.cbrt")
        print(f"  [{size:>6}b e{exp}]  {len(d):>7} unique entries -> {out}"
              + ("   (dry-run, not written)" if dry else ""))
        if dry:
            continue
        with open(out, "w") as f:
            f.write(f"{len(d)}\n")
            for prime in sorted(d, key=int):
                f.write(f"{prime} {d[prime]}\n")

    if not dry:
        print("\nDone. Point the solvers at these cbrt_<size>_e<exp>.cbrt files "
              "and stop the per-job `rm -rf` of the split dirs.")

if __name__ == "__main__":
    main()
