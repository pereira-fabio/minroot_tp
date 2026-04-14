#!/bin/bash -l
#SBATCH --job-name=minroot_k16_bench
#SBATCH --output=minroot_k16_%j.out
#SBATCH --error=minroot_k16_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00

module load compiler/GCC/13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile the single-node version
g++ -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k16.cpp src/k_common.cpp \
    -lgmp -o k16

CHALLENGE="data/challenge_16_2048.txt"
MODULO="data/modulo_16_2048.txt"

echo "=== K16 COMPARISON: 2048-bit ==="

echo "1. Optimized Factorization + Table Method:"
./k16 $CHALLENGE $MODULO --method factor --bench 10

echo -e "\n2. Sequential powm (Fermat Baseline):"
./k16 $CHALLENGE $MODULO --method powm --bench 10

echo -e "\n3. Factorization + Sequential powm Reconstruction (Isolation Test):"
./k16 $CHALLENGE $MODULO --method powm-recon --bench 10