#!/usr/bin/env bash
# 48h + 24h GPU soak. Spec 16.2 V10. gpu partition max is 2 days.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
NCSHARE="${PROMETHEUS_NCSHARE:-$HOME/prometheus-build/ncshare.sh}"
REMOTE="${AIL_REMOTE_ROOT:-/work/ttennant1/prometheus}"
RUN_ID="${1:-v10-$(date +%Y%m%d-%H%M%S)}"

rsync -az \
  --exclude .git --exclude target --exclude .venv --exclude __pycache__ \
  --exclude .ruff_cache --exclude .pytest_cache --exclude runs \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  "$ROOT/" "ncshare:$REMOTE/"

JOB="$HERE/V10.sh"
chmod +x "$JOB"
id1=$("$NCSHARE" submit "$RUN_ID" "v10a" "$JOB" "2-00:00:00" 2)
echo "V10 job1 $id1"
id2=$("$NCSHARE" submit "$RUN_ID-b" "v10b" "$JOB" "1-02:00:00" 2 "--dependency=afterany:$id1")
echo "V10 job2 $id2 afterany:$id1"
