#!/usr/bin/env bash
# Two 8-GPU jobs: train, then resume. Spec 16.2 V5. gpu partition, not interactive-gpu.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NCSHARE="${PROMETHEUS_NCSHARE:-$HOME/prometheus-build/ncshare.sh}"
REMOTE="${AIL_REMOTE_ROOT:-/work/ttennant1/prometheus}"
RUN_ID="${1:-v5-$(date +%Y%m%d-%H%M%S)}"

rsync -az \
  --exclude .git --exclude target --exclude .venv --exclude __pycache__ \
  --exclude .ruff_cache --exclude .pytest_cache --exclude runs \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  "$ROOT/" "ncshare:$REMOTE/"

JOB="$HERE/V5.sh"
chmod +x "$JOB"
id1=$("$NCSHARE" submit "$RUN_ID" "v5a" "$JOB" "04:00:00" 8)
echo "V5 job1 $id1"
id2=$("$NCSHARE" submit "$RUN_ID-b" "v5b" "$JOB" "04:00:00" 8 "--dependency=afterany:$id1")
echo "V5 job2 $id2 afterany:$id1"
