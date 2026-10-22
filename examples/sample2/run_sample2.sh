#!/bin/bash
# Submit the full Sample2 pipeline as dependent Slurm jobs. Edit the variables below for your cluster.
set -euo pipefail
: "${TRACEL_ENV:?set TRACEL_ENV to the conda env with tracel installed}"
: "${XENIUM_DIR:?set XENIUM_DIR to the Xenium Sample2 output folder}"
: "${SCRNA_DIR:?set SCRNA_DIR to the SCP1039 folder (expression/, metadata/)}"
: "${ACCOUNT:?set ACCOUNT to your Slurm account}"
export TRACEL_ENV XENIUM_DIR SCRNA_DIR
export WORK=${WORK:-$PWD/sample2_run}                            # all outputs go here
PARTITION=${PARTITION:-cpu}
WINDOWS_PER_TASK=${WINDOWS_PER_TASK:-8}; CONCURRENCY=${CONCURRENCY:-40}
NO_STRUCTURE=${NO_STRUCTURE:-0}                                  # 1 = OT1 without the structure term
HERE=$(cd "$(dirname "$0")" && pwd); export EXAMPLE_DIR=$HERE
mkdir -p "$WORK/logs"
common=(--account="$ACCOUNT" --partition="$PARTITION" --export=ALL --parsable)

ot1=$(sbatch "${common[@]}" --job-name=tracel_ot1 --cpus-per-task=16 --mem=64G --time=08:00:00 \
      --output="$WORK/logs/ot1_%j.out" "$HERE/step1_ot1.sbatch")
prep=$(sbatch "${common[@]}" --job-name=tracel_prep --cpus-per-task=2 --mem=32G --time=01:00:00 \
       --output="$WORK/logs/prep_%j.out" "$HERE/step2_prepare.sbatch")
echo "OT1 job $ot1, OT2 prepare job $prep"
echo "When both finish, run:  bash $HERE/submit_ot2.sh   (needs the number of windows from prepare)"
