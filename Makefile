# =============================================================================
# minroot_tp — CPU (MPI) and GPU (CUDA) builds
#
#   make                  the six algorithms, MPI      (needs mpicxx, libgmp)
#   make gpu              the six algorithms, CUDA     (needs nvcc)
#   make experiments      research code, CPU           — see experiments/README.md
#   make gpu-experiments  research code, CUDA
#   make tools            challenge and prime-table generators
#
# On the GPU cluster, load CUDA first and pick the architecture:
#   module load system/CUDA/12.6.0
#   make gpu              # V100, sm_70 (default)
#   make gpu ARCH=sm_89   # L40
#   make gpu ARCH=sm_80   # A100
# =============================================================================

CXXFLAGS = -O3 -std=c++17 -fopenmp -Wall
CFLAGS   = -O2 -fopenmp
INCLUDE  = -I src/common -I experiments/delta
LIBS     = -lgmp

NVCC       ?= nvcc
ARCH       ?= sm_70
NVCCFLAGS   = -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -fopenmp
GPU_INCLUDE = -I src/common -I src/common/gpu -I experiments/delta
GPU_LIBS    = -lgmp -lgomp

COMMON_SRC = src/common/common.cpp \
             src/common/factorization.cpp \
             src/common/solver_common.cpp \
             experiments/delta/delta_table.cpp

# The CUDA solvers are MPI-free: no solver_common.cpp.
GPU_COMMON_SRC = src/common/common.cpp \
                 src/common/factorization.cpp \
                 experiments/delta/delta_table.cpp

.PHONY: all gpu experiments gpu-experiments tools clean

# The six algorithms the thesis measures.
all: alg2_k16 alg2_k32 alg2_k35 alg3 alg4 alg7 challenges

gpu: alg2_k16_cuda alg2_k32_cuda alg2_k35_cuda alg3_cuda alg4_cuda alg7_cuda

# Research code that is not part of the attack — see experiments/README.md.
experiments: alg_crt cpu_scale_sim delta_convert prefilter_bench

gpu-experiments: alg_crt_cuda alg2_ksweep_cuda gpu_scale_sim_cuda

tools: challenges gen_prime_table

# --- algorithms, CPU ---------------------------------------------------------

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

# --- algorithms, GPU ---------------------------------------------------------

alg2_k16_cuda: src/iris/alg2_k16_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg2_k32_cuda: src/iris/alg2_k32_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg2_k35_cuda: src/iris/alg2_k35_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg3_cuda: src/iris/alg3_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg4_cuda: src/iris/alg4_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg7_cuda: src/iris/alg7_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

# --- experiments, CPU --------------------------------------------------------

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

# --- experiments, GPU --------------------------------------------------------

# Self-contained CRT context (no crt_common.cpp -> no MPI).
alg_crt_cuda: experiments/crt/alg_crt_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

# Smoothness-bound sweep: where the GPU cache boundary actually sits.
alg2_ksweep_cuda: experiments/ksweep/alg2_ksweep_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

# Strong-scaling simulator used for the beyond-4-GPU projections.
gpu_scale_sim_cuda: experiments/scale_sim/gpu_scale_sim_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

# --- tools -------------------------------------------------------------------

challenges: tools/challenges.c
	gcc $(CFLAGS) $^ $(LIBS) -o $@

gen_prime_table: tools/gen_prime_table.c
	gcc $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f alg2_k16 alg2_k32 alg2_k35 alg3 alg4 alg7 \
	      alg2_k16_cuda alg2_k32_cuda alg2_k35_cuda alg3_cuda alg4_cuda alg7_cuda \
	      alg_crt cpu_scale_sim delta_convert prefilter_bench \
	      alg_crt_cuda alg2_ksweep_cuda gpu_scale_sim_cuda \
	      challenges gen_prime_table
