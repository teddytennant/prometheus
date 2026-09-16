#!/usr/bin/env bash
# V2 job template (spec 16.2). Exit criterion: loss matches single-device to 1e-6 relative over steps; routing identical
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V2
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 2 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 2 --result "$OUT/result.json"
