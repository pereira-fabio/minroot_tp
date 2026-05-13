CXXFLAGS = -O3 -std=c++17 -fopenmp -Wall
CFLAGS   = -O2 -fopenmp
INCLUDE  = -I src/common
LIBS     = -lgmp

COMMON_SRC = src/common/k_common.cpp \
             src/common/k_factorization.cpp \
             src/common/k_solver_common.cpp

.PHONY: all clean challenges

all: k16_mpi k32_mpi k35_mpi alg3_mpi alg4_mpi challenges

k16_mpi: src/solvers/k16_mpi.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

k32_mpi: src/solvers/k32_mpi.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

k35_mpi: src/solvers/k35_mpi.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg3_mpi: src/solvers/k_alg3_mpi.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

alg4_mpi: src/solvers/k_alg4_mpi.cpp $(COMMON_SRC)
	mpicxx $(CXXFLAGS) $(INCLUDE) $^ $(LIBS) -o $@

challenges: tools/challenges.c
	gcc $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f k16_mpi k32_mpi k35_mpi alg3_mpi alg4_mpi challenges
