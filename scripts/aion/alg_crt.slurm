#!/bin/bash -l
#SBATCH --job-name=minroot_alg_crt
#SBATCH --output=minroot_alg_crt_%A_%a.out
#SBATCH --error=minroot_alg_crt_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=06:00:00

# ---------------------------------------------------------------------------
# Job array — CRT-method scaling study (advisor todo.md "CRT method")
#
# alg_crt computes the inverse root x = c^b mod M (b = e^-1 mod (M-1)) via
# alpha-ary exponentiation whose per-step "raise to alpha" is done in parallel
# over small primes with the CRT. No factorization / smoothness — works for any
# challenge (reuses the alg3 challenge/modulo files for convenience).
#
# Each array task runs a different node count (2 MPI ranks/node):
#   task 0->1 node ... task 6->64 nodes (128 ranks). Same N1..N64 sweep as alg7.
#
# ALPHA is the tunable per-step exponent (advisor: "use #cores to pick #primes,
# then explore"). Larger alpha -> fewer steps but larger prod (comm/reconstruct);
# smaller alpha -> more steps but cheaper per step. This job SWEEPS a list of
# alphas (ALPHAS below) in one run so the best value per size can be read off.
#   Override the list:   sbatch --export=ALL,ALPHAS="4 16" ...
#   WALL TIME: runtime scales ~linearly with the number of alphas. The slow
#   corner is large moduli at small alpha (~|M|/log2(alpha) steps, each a big
#   reconstruction). If it times out: trim ALPHAS, lower --n-challenges, or
#   raise #SBATCH --time.
#
# Results: data/results/aion/alg_crt/alg_crt_<JOBID>_N<nodes>/alg_crt_a<ALPHA>_crt_<JOBID>_<size>.json
# The CRT JSON is self-contained: powm_baseline_ms is recorded in it, so
# speedup_total = powm_baseline_ms / total_ms directly (no _powm_ sibling needed).
# The JSON does NOT record alpha internally, so it is encoded in the filename (a<ALPHA>).
# ---------------------------------------------------------------------------

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

NODE_COUNTS=(1 2 4 8 16 32 64)
N_NODES=${NODE_COUNTS[$SLURM_ARRAY_TASK_ID]}
N_CHALLENGE_COUNTS=(10 15 20 30 40 50 50)
N_CHALLENGES=${N_CHALLENGE_COUNTS[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=data/challenges/alg3
ALPHAS=${ALPHAS:-"2 4 8 16 32"}   # per-step exponents to sweep (override: --export=ALL,ALPHAS="4 16")
EXPONENT=${EXPONENT:-3}           # 3 = cube root; set 5 for the MinRoot 5th root

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/aion/alg_crt/alg_crt_${SLURM_JOB_ID}_N${N_NODES}
mkdir -p $JSON_DIR

echo "=== Job info ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES"
echo "Tasks (ranks): $(( N_NODES * 2 ))"
echo "alphas:        $ALPHAS"
echo "exponent:      $EXPONENT"
echo "Challenges:    $N_CHALLENGES  (challenge_0=cold warmup; 1..$(( N_CHALLENGES - 1 )) merged)"
echo "Sizes:         256 384 512 768 1024 2048 4096 8192 16384 24576"
echo "Results:       $JSON_DIR"
echo ""

cd $SLURM_SUBMIT_DIR
BINARY=./alg_crt_N${N_NODES}
trap 'rm -f $BINARY' EXIT

mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common -I experiments/delta \
    experiments/crt/alg_crt.cpp src/common/solver_common.cpp \
    src/common/common.cpp src/common/factorization.cpp \
    experiments/delta/delta_table.cpp \
    src/common/prefilter.cpp experiments/crt/crt_common.cpp \
    -lgmp -o alg_crt_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (alg_crt_N${N_NODES})"
echo ""

echo "============================================================"
echo "alg_crt alpha-sweep  (N=${N_NODES} nodes, alphas: $ALPHAS, $N_CHALLENGES challenges)"
echo "  JSON -> $JSON_DIR/"
echo "============================================================"
echo ""

# One full size-sweep per alpha; results distinguished by a<ALPHA> in the filename.
for A in $ALPHAS; do
    echo "  ---- alpha=$A ----"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) $BINARY \
        --all --data $DATA_DIR \
        --alpha $A \
        --exponent $EXPONENT \
        --n-challenges $N_CHALLENGES \
        --json $JSON_DIR/alg_crt_a${A}_crt_${SLURM_JOB_ID}
    echo ""
done

echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}) complete."
ls $JSON_DIR/*_${SLURM_JOB_ID}_*.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo "speedup_total = powm_baseline_ms / total_ms   (both in the same JSON)"
echo "============================================================"
