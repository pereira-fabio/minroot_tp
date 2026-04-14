#!/bin/bash -l
#SBATCH --job-name=minroot_k32_bench
#SBATCH --output=minroot_k32_%j.out
#SBATCH --error=minroot_k32_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Compile the MPI version
mpicxx -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k32_mpi.cpp src/k_common.cpp \
    -lgmp -o k32_mpi

echo "=== BENCHMARK: SIEVE METHOD (On-the-fly) ==="
srun ./k32_mpi --run-all --method sieve --bench 5

echo -e "\n=== BENCHMARK: TABLE METHOD (Precomputed) ==="
# Rank 0 will generate the table if it doesn't exist, then all ranks load it
srun ./k32_mpi --run-all --method table --bench 5