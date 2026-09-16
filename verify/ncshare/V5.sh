#!/usr/bin/env bash
# V5 job template (spec 16.2). Exit criterion: rung-0 loss curve matches ladder fit; ckpt/resume across jobs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V5
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 5 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 5 --result "$OUT/result.json"
