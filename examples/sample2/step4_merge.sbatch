#!/bin/bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$TRACEL_ENV"
tracel ot2-merge --run-dir "$WORK/ot2" --soft "$WORK/ot1/ot1_soft_expression.h5ad" --threshold "${THRESHOLD:-0.1}"
