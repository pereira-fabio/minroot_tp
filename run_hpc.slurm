#!/bin/bash -l
#SBATCH --job-name=minroot_k16
#SBATCH --output=minroot_k16_%j.out
#SBATCH --error=minroot_k16_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64      # Using all 64 physical cores
#SBATCH --exclusive             # Request exclusive access to the node
#SBATCH --time=48:00:00        # Max execution time (adjust as necessary)

# Load required modules
module load math/GMP/6.3.0-GCCcore-13.2.0

echo "Compiling ginza..."
g++ -O3 -std=c++17 -fopenmp -Wall -Wextra src/k16.cpp src/k_common.cpp -lgmp -o k16 

if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Compilation successful"

# Ensure OpenMP uses the allocated number of CPUs
# echo "Running with $OMP_NUM_THREADS threads"

./k16 data/challenges/k16/challenge_0_16_1024.txt data/challenges/k16/modulo_0_16_1024.txt --method powm
./k16 data/challenges/k16/challenge_0_16_1024.txt data/challenges/k16/modulo_0_16_1024.txt --method factor

echo "Job finished"
