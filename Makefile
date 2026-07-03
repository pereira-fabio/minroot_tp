CXXFLAGS = -O3 -std=c++17 -fopenmp -Wall
CFLAGS   = -O2 -fopenmp
INCLUDE  = -I src/common
LIBS     = -lgmp

COMMON_SRC = src/common/common.cpp \
             src/common/factorization.cpp \
             src/common/solver_common.cpp

.PHONY: all clean challenges

all: alg2_k16 alg2_k32 alg2_k35 alg3 alg4 alg7 challenges

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

challenges: tools/challenges.c
	gcc $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm -f alg2_k16 alg2_k32 alg2_k35 alg3 alg4 alg7 challenges
