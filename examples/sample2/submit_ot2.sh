#!/bin/bash
# After OT1 and ot2-prepare finished: submit the window array and the merge.
set -euo pipefail
: "${WORK:?set WORK as in run_sample2.sh}"
: "${ACCOUNT:?set ACCOUNT to your Slurm account}"; PARTITION=${PARTITION:-cpu}
export WINDOWS_PER_TASK=${WINDOWS_PER_TASK:-8}; CONCURRENCY=${CONCURRENCY:-40}
HERE=$(cd "$(dirname "$0")" && pwd)
n_windows=$(( $(wc -l < "$WORK/ot2/windows.csv") - 1 ))
n_tasks=$(( (n_windows + WINDOWS_PER_TASK - 1) / WINDOWS_PER_TASK ))
common=(--account="$ACCOUNT" --partition="$PARTITION" --export=ALL --parsable)
arr=$(sbatch "${common[@]}" --job-name=tracel_ot2 --array=0-$((n_tasks - 1))%"$CONCURRENCY" --cpus-per-task=2 \
      --mem=16G --time=24:00:00 --output="$WORK/logs/ot2_%A_%a.out" "$HERE/step3_windows.sbatch")
merge=$(sbatch "${common[@]}" --job-name=tracel_merge --dependency=afterany:"$arr" --cpus-per-task=4 --mem=64G \
        --time=04:00:00 --output="$WORK/logs/merge_%j.out" "$HERE/step4_merge.sbatch")
echo "OT2 array $arr ($n_windows windows in $n_tasks tasks), merge $merge"
echo "If some windows failed, rerun this script: finished windows are skipped."
