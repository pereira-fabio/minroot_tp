# =============================================================================
# Makefile — GPU build for alg7_cuda (Algorithm 7 inverted sieve on CUDA)
#
# IRIS GPU cluster — module setup (run once per shell session, before `make`):
#   module load system/CUDA/12.6.0   # confirmed on IRIS 2026-06-25
#   # Note: GCC/GMP module names on IRIS may differ from AION — confirm with
#   # `module spider GCC` / `module spider GMP` if the link step fails.
#
# ARCH: IRIS GPU nodes have 4x Nvidia Tesla V100 SXM2 → sm_70 (confirmed).
# Override only if running on a different GPU:
#   make ARCH=sm_80    # A100
#   make ARCH=sm_90    # H100
# =============================================================================

NVCC     ?= nvcc
ARCH     ?= sm_70
CXXFLAGS  = -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -fopenmp
INCLUDE   = -I ../common
LIBS      = -lgmp -lgomp

COMMON_SRC = ../common/common.cpp ../common/factorization.cpp

.PHONY: all clean

all: alg7_cuda alg2_k16_cuda alg2_k32_cuda alg2_k35_cuda alg3_cuda alg4_cuda

alg7_cuda: alg7_cuda.cu $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg2_k16_cuda: alg2_k16_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg2_k16_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg2_k32_cuda: alg2_k32_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg2_k32_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg2_k35_cuda: alg2_k35_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg2_k35_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg3_cuda: alg3_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg3_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

alg4_cuda: alg4_cuda.cu gpu_common.h $(COMMON_SRC)
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(INCLUDE) alg4_cuda.cu $(COMMON_SRC) $(LIBS) -o $@

clean:
	rm -f alg7_cuda alg2_k16_cuda alg2_k32_cuda alg2_k35_cuda alg3_cuda alg4_cuda

# Edit: 2026-06-30
