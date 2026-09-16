#!/usr/bin/env bash
# V9 job template (spec 16.2). Exit criterion: planted positive found and replicated; planted negative recorded negative
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V9
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 9 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 9 --result "$OUT/result.json"
