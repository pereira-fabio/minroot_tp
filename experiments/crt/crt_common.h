#pragma once
// =============================================================================
// crt_common.h — CRT-based modular exponentiation (advisor todo.md "CRT method")
//
// Goal: compute the inverse root  x = c^b mod M   (b = e^-1 mod (M-1), the same
// exponent run_powm_baseline uses) WITHOUT a single big mpz_powm mod M. Instead
// the exponentiation is done in small steps: each step raises the running value
// to a SMALL power alpha via the CRT over small primes.
//
//   c^alpha mod M   is computed as:
//     for each small prime p_i:  r_i = (c mod p_i)^alpha mod p_i      (tiny, fast)
//     reconstruct the EXACT integer c^alpha = (sum_i r_i * K_i) mod prod
//                                                    (needs prod > (M-1)^alpha)
//     reduce that exact integer mod M.
//
// The per-prime residues are split across MPI ranks (the parallel win the advisor
// wants to benchmark); reconstruction + reduction happen once per step on rank 0
// and the result is broadcast.
//
// The full exponent b is reached by alpha-ary (base-alpha) exponentiation built
// on top of crt_step — see alg_crt.cpp. (The pseudocode's literal "for b/alpha
// times: c <- c^alpha" does not compose to c^b for a general b; alpha-ary
// exponentiation is the correct composition and reduces to the same per-step
// primitive. Flagged for the advisor.)
//
// Setup (crt_ctx_init) is deterministic (no RNG), so every rank builds an
// identical context independently — no broadcast of the prime set / K_i needed.
// =============================================================================

#include <gmp.h>
#include <stdint.h>

typedef struct
{
    uint32_t *primes;    // p_0 .. p_{k-1}  (distinct, ~31-bit)
    int k;               // number of primes
    mpz_t *K;            // K_i = M_i * P_i, where P_i = prod/p_i and
                         //       M_i * P_i = 1 (mod p_i)  (CRT basis element)
    mpz_t prod;          // prod = prod_i p_i   (the CRT modulus, > (M-1)^alpha)
    unsigned long alpha; // per-step exponent
} crt_ctx_t;

// Build the CRT context for modulus M and step exponent alpha. Chooses just
// enough ~31-bit primes so that prod > M^alpha (>= (M-1)^alpha >= any acc^alpha
// with acc < M). Aborts via MPI_Abort if the prime set would exceed 2^32.
void crt_ctx_init(crt_ctx_t *ctx, const mpz_t modulo, unsigned long alpha);

void crt_ctx_free(crt_ctx_t *ctx);

// -----------------------------------------------------------------------------
// The phases below partition crt_step exactly -- their sum is the step time.
// Only `local` is parallel work; everything from `gather` onward is serial cost
// that GROWS with rank count, which is why the scaling curve is U-shaped.
// -----------------------------------------------------------------------------
typedef struct
{
    double local_ms;    // per-prime  c mod p_i, r_i = c_i^alpha, local += r_i*K_i
    double localmod_ms; // mpz_mod(local, prod)  -- one |prod|-bit reduction
    double pack_ms;     // export_fixed / mpz_import marshalling
    double gather_ms;   // MPI_Gather of nranks x |prod| bytes to rank 0
    double combine_ms;  // rank 0: nranks x (import + add), then mod prod, mod M
    double bcast_ms;    // MPI_Bcast of the |M|-byte step result
    long steps;         // number of crt_step calls accumulated
} crt_profile_t;

static inline void crt_profile_reset(crt_profile_t *p)
{
    p->local_ms = p->localmod_ms = p->pack_ms = 0.0;
    p->gather_ms = p->combine_ms = p->bcast_ms = 0.0;
    p->steps = 0;
}

static inline double crt_profile_total(const crt_profile_t *p)
{
    return p->local_ms + p->localmod_ms + p->pack_ms + p->gather_ms + p->combine_ms + p->bcast_ms;
}

// One CRT step: c_inout <- c_inout^alpha mod M.
// This rank owns prime indices [lo, hi) of ctx->primes; rank 0 reconstructs and
// broadcasts, so on return c_inout is identical on every rank.
// `prof` may be NULL (no instrumentation); when non-NULL its fields are ADDED to.
void crt_step(mpz_t c_inout, const crt_ctx_t *ctx, const mpz_t modulo,
              int lo, int hi, int rank, int nranks,
              crt_profile_t *prof = nullptr);
