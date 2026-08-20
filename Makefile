CXXFLAGS = -O3 -std=c++17 -fopenmp -Wall
CFLAGS   = -O2 -fopenmp
INCLUDE  = -I src/common -I experiments/delta
LIBS     = -lgmp

COMMON_SRC = src/common/common.cpp \
             src/common/factorization.cpp \
             src/common/solver_common.cpp \
             experiments/delta/delta_table.cpp

.PHONY: all experiments tools clean

# The four algorithms the thesis measures.
all: alg2_k16 alg2_k32 alg2_k35 alg3 alg4 alg7 challenges

# Research code that is not part of the attack — see experiments/README.md.
experiments: alg_crt cpu_scale_sim delta_convert prefilter_bench

tools: challenges gen_prime_table

# --- algorithms --------------------------------------------------------------

alg2_k16: src/aion/alg2_k16.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg2_k32: src/aion/alg2_k32.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg2_k35: src/aion/alg2_k35.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg3: src/aion/alg3.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg4: src/aion/alg4.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg7: src/aion/alg7.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

# --- experiments -------------------------------------------------------------

# CRT exponentiation — negative result, never approaches the baseline.
alg_crt: experiments/crt/alg_crt.cpp experiments/crt/crt_common.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) -I experiments/crt $^ $(LIBS) -o $@

# Strong-scaling simulator used for the beyond-128-rank projections.
cpu_scale_sim: experiments/scale_sim/cpu_scale_sim.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

# B0 threshold pre-filter cost. No solver calls this primitive; Algorithm 4 has
# its own inline pre-filter. Kept because it produced the thesis C_pf figures.
prefilter_bench: experiments/prefilter/prefilter_bench.cpp \
                 experiments/prefilter/prefilter.cpp \
                 src/common/common.cpp src/common/factorization.cpp
	mpicxx $(CXXFLAGS) $(INCLUDE) -I experiments/prefilter $^ $(LIBS) -o $@

# Flat prime table -> delta table. Replaces the old `--convert-delta` mode.
delta_convert: experiments/delta/delta_convert.cpp experiments/delta/delta_table.cpp \
               src/common/common.cpp src/common/factorization.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

# --- tools -------------------------------------------------------------------

challenges: tools/challenges.c
	gcc $(CFLAGS) $^ $(LIBS) -o $@

gen_prime_table: tools/gen_prime_table.c
	gcc $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f alg2_k16 alg2_k32 alg2_k35 alg3 alg4 alg7 \
	      alg_crt cpu_scale_sim delta_convert \
	      challenges gen_prime_table prefilter_bench
