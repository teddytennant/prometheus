#!/usr/bin/env bash
# V7 job template (spec 16.2). Exit criterion: reward rises on tiny task; log-prob drift halt; planted test-file write flagged
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V7
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 7 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 7 --result "$OUT/result.json"
