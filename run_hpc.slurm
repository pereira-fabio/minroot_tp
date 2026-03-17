#!/bin/bash -l
#SBATCH --job-name=minroot_k64
#SBATCH --output=minroot_k64_%j.out
#SBATCH --error=minroot_k64_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64      # Adjust this to match the number of cores you want to use
#SBATCH --time=48:00:00        # Max execution time (adjust as necessary)

# Load required modules (uncomment and adjust according to your HPC environment)
# module load gcc
# module load gmp
module load math/GMP/6.3.0-GCCcore-13.2.0

echo "Compiling ginza..."
g++ -O3 -std=c++17 -fopenmp -Wall -Wextra src/ginza1.cpp -lgmp -o ginza_run

if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Compilation successful"

# Ensure OpenMP uses the allocated number of CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running with $OMP_NUM_THREADS threads"

# Run the K=64 challenge 
# Adjust --bound and --attempts based on what you need
./ginza_run data/challenges/challenge_64_1024.txt data/challenges/modulo_64_1024.txt --bound 2000000 --attempts 10000000

echo "Job finished"

