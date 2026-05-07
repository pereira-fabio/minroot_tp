#!/bin/bash -l
#SBATCH --job-name=minroot_k35_mpi
#SBATCH --output=minroot_k35_mpi_%j.out
#SBATCH --error=minroot_k35_mpi_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=04:00:00

module load compiler/GCC/13.2.0
module load mpi/OpenMPI/4.1.6-GCC-13.2.0
module load math/GMP/6.3.0-GCCcore-13.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=active

# ---------------------------------------------------------------------------
# k35_mpi: paper's Algorithm 2 with B = 2^35
#
# Sieve range per rank: [65537, 2^35] / 16 ≈ 2.15 billion candidates
# Expected factor_ms:   ~640 ms per run (8x k32 sieve cost)
# Breakeven modulus:    ~17000 bits (powm > factor cost above this)
#
# We therefore benchmark at LARGER challenge sizes than k32/k16:
#   8192  bits -> powm ~118ms  -> speedup ~0.18x (below breakeven, expected)
#   12288 bits -> powm ~309ms  -> speedup ~0.48x
#   16384 bits -> powm ~615ms  -> speedup ~0.96x
#   20480 bits -> powm ~1047ms -> speedup ~1.6x  (first real speedup)
#   24576 bits -> powm ~1600ms -> speedup ~2.5x
# ---------------------------------------------------------------------------

DATA_DIR=data/challenges/k35
BENCH=10

# Sizes must be generated first with: ./challenges 2 <bits> 35
SIZES="8192 12288 16384 20480 24576"

JSON_DIR=$SLURM_SUBMIT_DIR/data/results/k35_mpi_N${SLURM_JOB_NUM_NODES}
mkdir -p $JSON_DIR

echo "=== Job info ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "Tasks (ranks): $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total workers: $(( SLURM_NTASKS * SLURM_CPUS_PER_TASK ))"
echo "Bench runs:    $BENCH"
echo "Sizes:         $SIZES"
echo "Results:       $JSON_DIR"
echo "Algorithm:     B = 2^35 (paper Algorithm 2)"
echo "Breakeven:     ~17000 bits (larger sizes needed vs k32)"
echo ""

cd $SLURM_SUBMIT_DIR

mpicxx -O3 -std=c++17 -fopenmp -Wall \
    src/k35_mpi.cpp src/k_solver_common.cpp \
    src/k_common.cpp src/k_factorization.cpp \
    -lgmp -o k35_mpi

if [ $? -ne 0 ]; then echo "Compilation failed"; exit 1; fi
echo "Compilation successful"
echo ""

run_size() {
    local BITS=$1
    local CHALLENGE=$DATA_DIR/challenge_0_35_${BITS}.txt
    local MODULO=$DATA_DIR/modulo_35_${BITS}.txt

    if [ ! -f "$CHALLENGE" ] || [ ! -f "$MODULO" ]; then
        echo "  [SKIP] ${BITS}-bit files not found"
        echo "         Generate with: ./challenges 2 ${BITS} 35"
        return
    fi

    echo "------------------------------------------------------------"
    echo "  Size: ${BITS} bits  (k35: B=2^35)"
    echo "------------------------------------------------------------"

    echo "  [powm]"
    srun --mpi=pmix ./k35_mpi \
        $CHALLENGE $MODULO \
        --method powm \
        --bench $BENCH \
        --json $JSON_DIR/k35_mpi_powm_${BITS}_${SLURM_JOB_ID}.json
    echo ""

    echo "  [sieve]"
    srun --mpi=pmix ./k35_mpi \
        $CHALLENGE $MODULO \
        --method sieve \
        --bench $BENCH \
        --json $JSON_DIR/k35_mpi_sieve_${BITS}_${SLURM_JOB_ID}.json
    echo ""
}

echo "============================================================"
echo "k35_mpi Benchmark  (N=${SLURM_JOB_NUM_NODES} nodes, $BENCH runs)"
echo "  methods: powm | sieve"
echo "  JSON  -> $JSON_DIR/"
echo "============================================================"
echo ""

for SIZE in $SIZES; do
    run_size $SIZE
done

echo "============================================================"
echo "All sizes complete."
echo "JSON files in: $JSON_DIR/"
ls $JSON_DIR/*_${SLURM_JOB_ID}.json 2>/dev/null | sed 's|.*/||' | sed 's/^/  /'
echo ""
echo "Note: speedup visible only at >= ~17000 bits (breakeven for B=2^35)."
echo "Job finished  (Job ID: $SLURM_JOB_ID)"
echo "============================================================"