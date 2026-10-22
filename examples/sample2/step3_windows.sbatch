#!/bin/bash
# One array task solves WINDOWS_PER_TASK consecutive windows.
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$TRACEL_ENV"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=false XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
start=$(( SLURM_ARRAY_TASK_ID * WINDOWS_PER_TASK )); end=$(( start + WINDOWS_PER_TASK - 1 ))
tracel ot2-run --run-dir "$WORK/ot2" --soft "$WORK/ot1/ot1_soft_expression.h5ad" --windows "${start}-${end}"
