# =============================================================================
# minroot_tp — CPU (MPI) and GPU (CUDA) builds
#
#   make                  the six algorithms, MPI      (needs mpicxx, libgmp)
#   make gpu              the six algorithms, CUDA     (needs nvcc)
#   make experiments      research code, CPU           — see experiments/README.md
#   make gpu-experiments  research code, CUDA
#   make tools            challenge and prime-table generators
#   make test             build, run every solver, check every result
#
# GPU builds detect the card in this machine, so `make gpu` normally needs no
# argument. Override or force portability when the build host has no GPU:
#   make gpu                  detect the local card (nvidia-smi)
#   make gpu ARCH=sm_75       force one architecture (2080/2080 Super = sm_75)
#   make gpu ARCH=portable    one fatbinary that runs on any Volta-or-newer card
# On a cluster, load CUDA first: module load system/CUDA/12.6.0
# =============================================================================

MPICXX   ?= mpicxx
CXXFLAGS  = -O3 -std=c++17 -fopenmp -Wall
CFLAGS    = -O2 -fopenmp
INCLUDE   = -I src/common -I experiments/delta
LIBS      = -lgmp

NVCC       ?= nvcc
ARCH       ?= auto
# Extra -D switches for the CUDA builds, e.g. GPU_DEFS=-DKF_TD_ILP=2
GPU_DEFS   ?=
NVCCFLAGS   = -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -fopenmp $(GPU_DEFS)
GPU_INCLUDE = -I src/common -I src/common/gpu -I experiments/delta
GPU_LIBS    = -lgmp -lgomp

# Compute capability of the first visible GPU: "7.5" -> "75". Empty when there
# is no nvidia-smi or no card, which is the normal case on a login node.
GPU_CC     := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
NVCC_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' | head -1)

# Volta through Ada, plus PTX so anything newer JITs. Nothing in the kernels is
# architecture-specific -- no warp intrinsics, no tensor cores, no half or
# double atomics -- so the only constraint is what this nvcc still accepts.
GENCODE_PORTABLE = -gencode arch=compute_70,code=sm_70 \
                   -gencode arch=compute_75,code=sm_75 \
                   -gencode arch=compute_80,code=sm_80 \
                   -gencode arch=compute_86,code=sm_86 \
                   -gencode arch=compute_89,code=sm_89
ifeq ($(shell test "$(NVCC_MAJOR)" -ge 12 2>/dev/null && echo y),y)
GENCODE_PORTABLE += -gencode arch=compute_90,code=sm_90 \
                    -gencode arch=compute_90,code=compute_90
else
GENCODE_PORTABLE += -gencode arch=compute_89,code=compute_89
endif

ifeq ($(ARCH),auto)
  ifeq ($(GPU_CC),)
    NVARCH = $(GENCODE_PORTABLE)
    ARCH_NOTE = no GPU detected -> portable fatbinary
  else
    NVARCH = -arch=sm_$(GPU_CC)
    ARCH_NOTE = detected sm_$(GPU_CC)
  endif
else ifeq ($(ARCH),portable)
  NVARCH = $(GENCODE_PORTABLE)
  ARCH_NOTE = portable fatbinary
else
  NVARCH = -arch=$(ARCH)
  ARCH_NOTE = forced $(ARCH)
endif

COMMON_SRC = src/common/common.cpp \
             src/common/factorization.cpp \
             src/common/solver_common.cpp \
             experiments/delta/delta_table.cpp

# The CUDA solvers are MPI-free: no solver_common.cpp.
GPU_COMMON_SRC = src/common/common.cpp \
                 src/common/factorization.cpp \
                 experiments/delta/delta_table.cpp

.PHONY: all gpu gpu-arch experiments gpu-experiments tools test clean

# The six algorithms the thesis measures.
all: alg2_k16 alg2_k32 alg2_k35 alg3 alg4 alg7 challenges

gpu: gpu-arch alg2_k16_cuda alg2_k32_cuda alg2_k35_cuda alg3_cuda alg4_cuda alg7_cuda

# Report which architecture the build resolved to, so a wrong guess is visible
# before the binaries fail to launch.
gpu-arch:
	@echo "CUDA architecture: $(ARCH_NOTE)"

# Research code that is not part of the attack — see experiments/README.md.
experiments: alg_crt cpu_scale_sim delta_convert prefilter_bench

gpu-experiments: alg_crt_cuda alg2_ksweep_cuda gpu_scale_sim_cuda

tools: challenges gen_prime_table

# The project's test: builds what is missing, runs every solver once at a small
# size and checks each result against the correctness gate. Adds the GPU half
# when nvcc and a card are present. See tools/smoke_test.sh --help.
test:
	@./tools/smoke_test.sh

# --- algorithms, CPU ---------------------------------------------------------

alg2_k16: src/aion/alg2_k16.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg2_k32: src/aion/alg2_k32.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg2_k35: src/aion/alg2_k35.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg3: src/aion/alg3.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg4: src/aion/alg4.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg7: src/aion/alg7.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

# --- algorithms, GPU ---------------------------------------------------------

alg2_k16_cuda: src/iris/alg2_k16_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg2_k32_cuda: src/iris/alg2_k32_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg2_k35_cuda: src/iris/alg2_k35_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg3_cuda: src/iris/alg3_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg4_cuda: src/iris/alg4_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

alg7_cuda: src/iris/alg7_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

# --- experiments, CPU --------------------------------------------------------

# CRT exponentiation — negative result, never approaches the baseline.
alg_crt: experiments/crt/alg_crt.cpp experiments/crt/crt_common.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) -I experiments/crt $^ $(LIBS) -o $@

# Strong-scaling simulator used for the beyond-128-rank projections.
cpu_scale_sim: experiments/scale_sim/cpu_scale_sim.cpp $(COMMON_SRC)
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

# B0 threshold pre-filter cost. No solver calls this primitive; Algorithm 4 has
# its own inline pre-filter. Kept because it produced the thesis C_pf figures.
prefilter_bench: experiments/prefilter/prefilter_bench.cpp \
                 experiments/prefilter/prefilter.cpp \
                 src/common/common.cpp src/common/factorization.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDE) -I experiments/prefilter $^ $(LIBS) -o $@

# Flat prime table -> delta table. Replaces the old `--convert-delta` mode.
delta_convert: experiments/delta/delta_convert.cpp experiments/delta/delta_table.cpp \
               src/common/common.cpp src/common/factorization.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

# --- experiments, GPU --------------------------------------------------------

# Self-contained CRT context (no crt_common.cpp -> no MPI).
alg_crt_cuda: experiments/crt/alg_crt_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

# Smoothness-bound sweep: where the GPU cache boundary actually sits.
alg2_ksweep_cuda: experiments/ksweep/alg2_ksweep_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

# Strong-scaling simulator used for the beyond-4-GPU projections.
gpu_scale_sim_cuda: experiments/scale_sim/gpu_scale_sim_cuda.cu $(GPU_COMMON_SRC)
	$(NVCC) $(NVARCH) $(NVCCFLAGS) $(GPU_INCLUDE) $^ $(GPU_LIBS) -o $@

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
