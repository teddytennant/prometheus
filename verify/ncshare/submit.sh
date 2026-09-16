#!/usr/bin/env bash
# Submit V stages to NCShare H200s. Packs V0-V10 into one 2-GPU interactive-gpu
# job so we do not wait a day on the gpu partition. Idempotent under flock.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NCSHARE="${PROMETHEUS_NCSHARE:-$HOME/prometheus-build/ncshare.sh}"
REMOTE_ROOT="${PROMETHEUS_ROOT:-/work/ttennant1/prometheus}"
STATE="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}"
RUN_ID="${PROMETHEUS_RUN_ID:-v-h200}"
LOCK="$STATE/submit.lock"
mkdir -p "$STATE"

exec 9>"$LOCK"
flock -n 9 || {
  echo "submit already in progress" >&2
  exit 0
}

echo "rsync $ROOT -> ncshare:$REMOTE_ROOT"
rsync -az \
  --exclude .git --exclude target --exclude .venv --exclude __pycache__ \
  --exclude .ruff_cache --exclude .pytest_cache --exclude runs \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  "$ROOT/" "ncshare:$REMOTE_ROOT/"

mkdir -p "$STATE/runs/$RUN_ID"
export AIL_PARTITION="${AIL_PARTITION:-interactive-gpu}"
# 2 GPUs: V0 NCCL + V2 mesh. 50 min < interactive-gpu 1h cap.
SCRIPT="${SUBMIT_SCRIPT:-$HERE/run_all.sh}"
WALL="${SUBMIT_WALL:-00:50:00}"
GPUS="${SUBMIT_GPUS:-2}"
jobid="$("$NCSHARE" submit "$RUN_ID" all "$SCRIPT" "$WALL" "$GPUS")"
echo "submitted $jobid partition=$AIL_PARTITION"
echo "$jobid" >"$STATE/runs/$RUN_ID/last-job"
echo "$jobid"
