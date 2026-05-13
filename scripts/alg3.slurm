#!/bin/bash -l
#SBATCH --job-name=minroot_alg3_mpi
#SBATCH --output=minroot_alg3_mpi_%A_%a.out
#SBATCH --error=minroot_alg3_mpi_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=04:00:00

# ---------------------------------------------------------------------------
# Job array — Algorithm 3 scaling study
#
# Algorithm 3 extends Algorithm 2 by tolerating one medium prime q0 ∈ (2^32,2^65].
# Success probability: ~2^(-18) vs ~2^(-24.9) for k32 (64× more successes).
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
# Isolation from Algorithm 2:
#   Challenges:  data/challenges/alg3/
#   cbrt caches: data/cbrt/<stem>.cbrt_alg3.<rank>
#   Results:     data/results/alg3_mpi_N<nodes>/
#
# Results landing: data/results/alg3_mpi_N1/ .. data/results/alg3_mpi_N64/
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR=data/challenges/alg3
BENCH=50

SIZES="256 512 1024 2048 4096 6144 8192 12288 16384 20480 24576"

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/alg3_mpi_N${N_NODES}
mkdir -p $JSON_DIR

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Array job:     $SLURM_ARRAY_JOB_ID  task $SLURM_ARRAY_TASK_ID"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $N_NODES"
echo "Tasks (ranks): $(( N_NODES * 2 ))"
echo "Total workers: $(( N_NODES * 2 * SLURM_CPUS_PER_TASK ))"
echo "Bench runs:    $BENCH"
echo "Sizes:         $SIZES"
echo "Results:       $JSON_DIR"
echo ""

cd $SLURM_SUBMIT_DIR

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
mpicxx -O3 -std=c++17 -fopenmp -Wall \
    -I src/common \
    src/solvers/k_alg3_mpi.cpp src/common/k_solver_common.cpp \
    src/common/k_common.cpp src/common/k_factorization.cpp \
    -lgmp -o alg3_mpi_N${N_NODES}

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful (alg3_mpi_N${N_NODES})"
echo ""

# ---------------------------------------------------------------------------
# Helper: benchmark one challenge size
# ---------------------------------------------------------------------------
run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_alg3_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_alg3_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found -- run challenge_gen.slurm first"
        return
    fi

    # ALG3: cache suffix is .cbrt_alg3.0
    CACHE="data/cbrt/$(basename ${MODULO%.txt}).cbrt_alg3.0"
    if [ ! -f "$CACHE" ]; then
        echo "  NOTE: cbrt cache missing -- first run will do an untimed precompute pass"
    else
        echo "  cbrt cache: found ($(wc -c < "$CACHE") bytes)"
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits  |  Nodes: ${N_NODES}  |  Algorithm 3"
    echo "------------------------------------------------------------"

    echo "  [powm] baseline"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg3_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/alg3_mpi_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [sieve]"
    srun --mpi=pmix -N ${N_NODES} --ntasks=$(( N_NODES * 2 )) ./alg3_mpi_N${N_NODES} \
        $CHALLENGE $MODULO \
        --method sieve \
        --bench $BENCH \
        --json $JSON_DIR/alg3_mpi_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""
}

# ---------------------------------------------------------------------------
# Run all sizes
# ---------------------------------------------------------------------------
echo "============================================================"
echo "alg3_mpi Scaling Study  (N=${N_NODES} nodes, $BENCH runs/method)"
echo "  methods: powm | sieve"
echo "  JSON  -> $JSON_DIR/"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    run_size $SIZE
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID (N=${N_NODES}) complete."
echo "JSON files in: $JSON_DIR/"
ls $JSON_DIR/*_${SLURM_JOB_ID}.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Use warm_avg.* (runs_valid runs) for all analysis."
echo "Job finished  (Array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID)"
echo "============================================================"
