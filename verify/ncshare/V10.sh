#!/usr/bin/env bash
# V10 job template (spec 16.2). Exit criterion: 72h soak: no lost task, no duplicated output, no dead token
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V10
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 10 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 10 --result "$OUT/result.json"
