#!/bin/bash -l
#SBATCH --job-name=minroot_k32
#SBATCH --output=minroot_k32_%j.out
#SBATCH --error=minroot_k32_%j.err
#SBATCH --nodes=4                   # 4 nodes
#SBATCH --ntasks-per-node=2         # 2 MPI ranks per node (one per NUMA node)
#SBATCH --cpus-per-task=32          # 32 OpenMP threads per rank
#SBATCH --exclusive
#SBATCH --time=01:00:00

# --- modules ---
module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

# --- OpenMP: bind each rank to its NUMA node ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # 32 threads per rank
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

echo "=== Job info ==="
echo "Nodes:          $SLURM_JOB_NUM_NODES"
echo "Tasks (ranks):  $SLURM_NTASKS"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "Total cores:    $(( SLURM_NTASKS * SLURM_CPUS_PER_TASK ))"
echo ""

# --- compile ---
echo "Compiling k32_mpi..."
mpicxx -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k32_mpi.cpp src/k_common.cpp \
    -lgmp -o k32_mpi

if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi
echo "Compilation successful"
echo ""

# --- run all four challenge sizes ---
for size in 1024 2048 5000 7000; do
    echo "--- challenge_32_${size} ---"
    srun --mpi=pmix k32_mpi \
        data/challenges/k32/challenge_32_${size}.txt \
        data/challenges/k32/modulo_32_${size}.txt \
        --bench 5
    echo ""
done

echo "Job finished"