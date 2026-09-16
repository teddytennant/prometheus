#!/usr/bin/env bash
# V1 job template (spec 16.2). Exit criterion: JAX vs reference logits 1e-5; grad checks; overfits one batch
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V1
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 1 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 1 --result "$OUT/result.json"
