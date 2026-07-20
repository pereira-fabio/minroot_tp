#!/bin/bash -l
#SBATCH --job-name=minroot_alg4
#SBATCH --output=minroot_alg4_%A_%a.out
#SBATCH --error=minroot_alg4_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=08:00:00

# ---------------------------------------------------------------------------
# Job array — Algorithm 4 scaling study
#
# Algorithm 4 extends Algorithm 3 by tolerating composite cofactors ≤ 2^64.
# After trial division (primes ≤ 2^32), the cofactor is factored by
# Pollard's rho if composite. All medium primes have their cbrts folded in.
#
# Each array task runs at a different node count:
#   task 0  ->  1 node   (2  ranks,  64 workers)
#   task 1  ->  2 nodes  (4  ranks, 128 workers)
#   task 2  ->  4 nodes  (8  ranks, 256 workers)
#   task 3  ->  8 nodes  (16 ranks, 512 workers)
#   task 4  -> 16 nodes  (32 ranks, 1024 workers)
#   task 5  -> 32 nodes  (64 ranks, 2048 workers)
#   task 6  -> 64 nodes  (128 ranks, 4096 workers)
#
# Two challenge sets are run (Pass 1 then Pass 2):
#   --all        data/challenges/alg4/       -> exercises hit_type=2
#   --all --multi data/challenges/alg4_multi/ -> exercises hit_type=3 (Pollard's rho)
#
# Results landing: data/results/aion/alg4/alg4_<jobid>_N1/ .. alg4_<jobid>_N64/
#                  data/results/aion/alg4/alg4_multi_<jobid>_N1/ .. alg4_multi_<jobid>_N64/
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

# ---------------------------------------------------------------------------
# Map array index -> node count
# ---------------------------------------------------------------------------
NODE_COUNTS=(1 2 4 8 16 32 64)
N_NODES=${NODE_COUNTS[$SLURM_ARRAY_TASK_ID]}
N_CHALLENGE_COUNTS=(10 15 20 30 40 50 50)
N_CHALLENGES=${N_CHALLENGE_COUNTS[$SLURM_ARRAY_TASK_ID]}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/alg4
DATA_DIR_MULTI=data/challenges/alg4_multi

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/aion/alg4/alg4_${SLURM_JOB_ID}_N${N_NODES}
JSON_DIR_MULTI=$SLURM_SUBMIT_DIR/data/results/aion/alg4/alg4_multi_${SLURM_JOB_ID}_N${N_NODES}
mkdir -p $JSON_DIR $JSON_DIR_MULTI

# M1 (2026-07-16): per-variant dirs (not per-N), never wiped — the solver reads
# the merged cbrt_<size>.cbrt built by merge_cbrt.slurm (alg4 single vs alg4_multi).
CBRT_DIR_P1=$SLURM_SUBMIT_DIR/data/cbrt/alg4_e3
CBRT_DIR_P2=$SLURM_SUBMIT_DIR/data/cbrt/alg4_multi_e3
mkdir -p $CBRT_DIR_P1 $CBRT_DIR_P2

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES"
echo "Tasks (ranks): $(( N_NODES * 2 ))"
echo "Total workers: $(( N_NODES * 2 * SLURM_CPUS_PER_TASK ))"
echo "Challenges:    $N_CHALLENGES"
echo "Sizes:         256 384 512 768 1024 2048"
echo "Results:       $JSON_DIR"
echo "Results multi: $JSON_DIR_MULTI"
echo ""

cd $SLURM_SUBMIT_DIR
BINARY=./alg4_N${N_NODES}
trap 'rm -f $BINARY' EXIT

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common \
    src/aion/alg4.cpp src/common/solver_common.cpp \
    src/common/common.cpp src/common/factorization.cpp \
    -lgmp -o alg4_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (alg4_N${N_NODES})"
echo ""

# ---------------------------------------------------------------------------
# Pass 1: single medium prime cofactor (hit_type=2)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "alg4 Pass 1: single-prime cofactor  (N=${N_NODES}, $N_CHALLENGES challenges/method)"
echo "  methods: powm (baseline, once) | table | sieve (--skip-powm-baseline)"
echo "  JSON  -> $JSON_DIR/"
echo "============================================================"
echo ""

echo "  [powm — baseline ($N_CHALLENGES challenges)]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg4_N${N_NODES} \
    --all --data $DATA_DIR \
    --method powm \
    --n-challenges $N_CHALLENGES \
    --cbrt-dir $CBRT_DIR_P1 \
    --json $JSON_DIR/alg4_powm_${SLURM_JOB_ID}
echo ""

echo "  [table]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg4_N${N_NODES} \
    --all --data $DATA_DIR \
    --n-challenges $N_CHALLENGES \
    --method table \
    --skip-powm-baseline \
    --cbrt-dir $CBRT_DIR_P1 \
    --json $JSON_DIR/alg4_table_${SLURM_JOB_ID}
echo ""

echo "  [sieve]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg4_N${N_NODES} \
    --all --data $DATA_DIR \
    --n-challenges $N_CHALLENGES \
    --method sieve \
    --skip-powm-baseline \
    --cbrt-dir $CBRT_DIR_P1 \
    --json $JSON_DIR/alg4_sieve_${SLURM_JOB_ID}
echo ""

# ---------------------------------------------------------------------------
# Pass 2: composite cofactor (hit_type=3, Pollard's rho)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "alg4 Pass 2: composite cofactor  (N=${N_NODES}, $N_CHALLENGES challenges/method)"
echo "  methods: powm (baseline, once) | table | sieve (--skip-powm-baseline)"
echo "  JSON  -> $JSON_DIR_MULTI/"
echo "============================================================"
echo ""

echo "  [powm — baseline ($N_CHALLENGES challenges)]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg4_N${N_NODES} \
    --all --multi --data $DATA_DIR_MULTI \
    --method powm \
    --n-challenges $N_CHALLENGES \
    --cbrt-dir $CBRT_DIR_P2 \
    --json $JSON_DIR_MULTI/alg4_multi_mpi_powm_${SLURM_JOB_ID}
echo ""

echo "  [table]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg4_N${N_NODES} \
    --all --multi --data $DATA_DIR_MULTI \
    --n-challenges $N_CHALLENGES \
    --method table \
    --skip-powm-baseline \
    --cbrt-dir $CBRT_DIR_P2 \
    --json $JSON_DIR_MULTI/alg4_multi_mpi_table_${SLURM_JOB_ID}
echo ""

echo "  [sieve]"
srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg4_N${N_NODES} \
    --all --multi --data $DATA_DIR_MULTI \
    --n-challenges $N_CHALLENGES \
    --method sieve \
    --skip-powm-baseline \
    --cbrt-dir $CBRT_DIR_P2 \
    --json $JSON_DIR_MULTI/alg4_multi_mpi_sieve_${SLURM_JOB_ID}
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}) complete."
echo ""
echo "Pass 1 JSON (single-prime cofactor):"
ls $JSON_DIR/*_${SLURM_JOB_ID}_*.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Pass 2 JSON (composite cofactor):"
ls $JSON_DIR_MULTI/*_${SLURM_JOB_ID}_*.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Use warm_avg.* (runs_valid runs) for all analysis."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"

# Edit: 2026-07-16
