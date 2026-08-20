# =============================================================================
# Makefile — GPU builds (CUDA)
#
# IRIS GPU cluster — module setup (run once per shell session, before `make`):
#   module load system/CUDA/12.6.0
#   # Note: GCC/GMP module names on IRIS may differ from AION — confirm with
#   # `module spider GCC` / `module spider GMP` if the link step fails.
#
# ARCH: IRIS GPU nodes have 4x Nvidia Tesla V100 SXM2 → sm_70 (confirmed).
# Override only if running on a different GPU:
#   make ARCH=sm_80    # A100
#   make ARCH=sm_89    # L40
#   make ARCH=sm_90    # H100
# =============================================================================

NVCC     ?= nvcc
ARCH     ?= sm_70
CXXFLAGS  = -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -fopenmp
INCLUDE   = -I ../common -I ../../experiments/delta
LIBS      = -lgmp -lgomp

COMMON_SRC = ../common/common.cpp ../common/factorization.cpp \
             ../../experiments/delta/delta_table.cpp

EXP = ../../experiments

.PHONY: all experiments clean

# The four algorithms the thesis measures.
all: alg2_k16_cuda alg2_k32_cuda alg2_k35_cuda alg3_cuda alg4_cuda alg7_cuda

# Research code that is not part of the attack — see experiments/README.md.
experiments: alg_crt_cuda alg2_ksweep_cuda gpu_scale_sim_cuda

# --- algorithms --------------------------------------------------------------

alg2_k16_cuda: alg2_k16_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg2_k16_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg2_k32_cuda: alg2_k32_cuda.cu gpu_common.h gpu_k32_kernels.cuh gpu_sieve.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg2_k32_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg2_k35_cuda: alg2_k35_cuda.cu gpu_common.h gpu_k32_kernels.cuh gpu_sieve.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg2_k35_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg3_cuda: alg3_cuda.cu gpu_common.h gpu_k32_kernels.cuh gpu_sieve.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg3_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg4_cuda: alg4_cuda.cu gpu_common.h gpu_k32_kernels.cuh gpu_sieve.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg4_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg7_cuda: alg7_cuda.cu $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

# --- experiments -------------------------------------------------------------

# GPU CRT (Option A: GPU residues + host reconstruction). Self-contained CRT
# context (no crt_common.cpp -> no MPI); links only common + factorization.
alg_crt_cuda: $(EXP)/crt/alg_crt_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) -I. -I $(EXP)/crt \
	    $(EXP)/crt/alg_crt_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

# Smoothness-bound sweep: where the GPU cache boundary actually sits.
alg2_ksweep_cuda: $(EXP)/ksweep/alg2_ksweep_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) -I. \
	    $(EXP)/ksweep/alg2_ksweep_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

# Strong-scaling simulator used for the beyond-4-GPU projections.
gpu_scale_sim_cuda: $(EXP)/scale_sim/gpu_scale_sim_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) -I. \
	    $(EXP)/scale_sim/gpu_scale_sim_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

clean:
	rm -f alg2_k16_cuda alg2_k32_cuda alg2_k35_cuda alg3_cuda alg4_cuda alg7_cuda \
	      alg_crt_cuda alg2_ksweep_cuda gpu_scale_sim_cuda
