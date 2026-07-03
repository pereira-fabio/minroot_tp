#include "generate.h"
#include "solve.h"
#include "attack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
"MinRoot Time-Lock Puzzle Tool\n"
"\n"
"USAGE\n"
"  minroot-timelock generate [options] <output.mrt>\n"
"  minroot-timelock solve    <puzzle.mrt>\n"
"  minroot-timelock attack   [options] <puzzle.mrt>\n"
"\n"
"GENERATE OPTIONS\n"
"  -m <message>   Plaintext to encrypt          (default: \"Hello from MinRoot!\")\n"
"  -T <N>         Number of sequential roots    (default: 1000)\n"
"  -e <e>         Root exponent, coprime to p-1 (default: 5)\n"
"  -b <bits>      Prime modulus bit-length      (default: 128)\n"
"\n"
"ATTACK OPTIONS\n"
"  -B <bound>     Smoothness bound B            (default: 65536)\n"
"\n"
"QUICK DEMO (fast end-to-end on any machine)\n"
"  minroot-timelock generate -b 64 -T 20 -m \"secret\" puzzle.mrt\n"
"  minroot-timelock solve  puzzle.mrt\n"
"  minroot-timelock attack puzzle.mrt\n"
"\n"
"SMOOTHNESS GUIDE\n"
"  The hit rate per blinding attempt is rho(u) where u = |p| / log2(B).\n"
"  For a practical attack keep u <= 4 (rho(4) ~ 1 in 200):\n"
"    64-bit prime,  B=65536 : u=4  -> ~200 attempts/step  (fast demo)\n"
"   128-bit prime,  B=65536 : u=8  -> ~31M attempts/step  (slow)\n"
"  Security note: use -b 256 or larger for a real puzzle.\n"
"                 -b 64 is for demonstration only.\n"
"\n"
"FILE FORMAT\n"
"  Plain-text .mrt file with labeled hex fields (MINROOT-TIMELOCK-V1).\n"
    );
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }

    /* ---- generate ---- */
    if (strcmp(argv[1], "generate") == 0) {
        int           bits    = 128;
        unsigned long e       = 5;
        unsigned long T       = 1000;
        const char   *message = "Hello from MinRoot!";
        const char   *outpath = NULL;

        for (int i = 2; i < argc; i++) {
            if      (!strcmp(argv[i], "-m") && i+1 < argc) { message = argv[++i]; }
            else if (!strcmp(argv[i], "-T") && i+1 < argc) { T = strtoul(argv[++i], NULL, 10); }
            else if (!strcmp(argv[i], "-e") && i+1 < argc) { e = strtoul(argv[++i], NULL, 10); }
            else if (!strcmp(argv[i], "-b") && i+1 < argc) { bits = atoi(argv[++i]); }
            else if (argv[i][0] != '-')                     { outpath = argv[i]; }
            else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
        }
        if (!outpath) {
            fprintf(stderr, "Error: output file path required.\n");
            return 1;
        }
        return cmd_generate(bits, e, T, message, outpath);
    }

    /* ---- solve ---- */
    if (strcmp(argv[1], "solve") == 0) {
        if (argc < 3) { fprintf(stderr, "Error: puzzle file required.\n"); return 1; }
        return cmd_solve(argv[2]);
    }

    /* ---- attack ---- */
    if (strcmp(argv[1], "attack") == 0) {
        unsigned long B      = 65536UL;
        const char   *inpath = NULL;

        for (int i = 2; i < argc; i++) {
            if      (!strcmp(argv[i], "-B") && i+1 < argc) { B = strtoul(argv[++i], NULL, 10); }
            else if (argv[i][0] != '-')                     { inpath = argv[i]; }
            else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
        }
        if (!inpath) {
            fprintf(stderr, "Error: puzzle file required.\n");
            return 1;
        }
        return cmd_attack(inpath, B);
    }

    fprintf(stderr, "Unknown subcommand: %s\n\n", argv[1]);
    usage();
    return 1;
}
