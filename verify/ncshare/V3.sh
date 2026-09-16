#!/usr/bin/env bash
# V3 job template (spec 16.2). Exit criterion: FP8 loss within 0.5% of BF16; NVFP4 numerics only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V3
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 3 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 3 --result "$OUT/result.json"
