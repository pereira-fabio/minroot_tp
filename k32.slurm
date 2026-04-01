#!/bin/bash -l
#SBATCH --job-name=minroot_k32
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
export OMP_WAIT_POLICY=active

echo "=== Job info ==="
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "Tasks (ranks): $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total cores:   $(( SLURM_NTASKS * SLURM_CPUS_PER_TASK ))"
echo ""

cd $SLURM_SUBMIT_DIR

mpicxx -O3 -std=c++17 -fopenmp -Wall -Wextra \
    src/k32_mpi.cpp src/k_common.cpp \
    -lgmp -o k32_mpi

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

# single srun call — all four challenge sizes run inside one MPI execution
# --all loops through 1024, 2048, 5000, 7000 internally
# --bench 5 runs each challenge 5 times and reports averages
srun --mpi=pmix ./k32_mpi \
    --all \
    --data data/challenges/k32 \
    --bench 5

echo ""
echo "Job finished"