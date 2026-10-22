#!/bin/bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$TRACEL_ENV"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT="$WORK/inputs" python "$EXAMPLE_DIR/make_inputs.py"   # Slurm copies this script, so $0 is not the example dir
extra=(); [ "${NO_STRUCTURE:-0}" = 1 ] && extra=(--no-structure)
tracel ot1 --spatial "$WORK/inputs/spatial_counts.h5ad" --reference "$WORK/inputs/scrna_counts.h5ad" \
  --spatial-label celltype --reference-label celltype --out "$WORK/ot1" "${extra[@]}"
