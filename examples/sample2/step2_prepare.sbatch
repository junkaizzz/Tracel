#!/bin/bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$TRACEL_ENV"
tracel ot2-prepare --transcripts "$XENIUM_DIR/transcripts.parquet" --cells "$XENIUM_DIR/cells.parquet" --run-dir "$WORK/ot2"
