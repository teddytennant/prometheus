#!/usr/bin/env bash
# V8 job template (spec 16.2). Exit criterion: SGLang vs JAX log-probs within threshold; tiered restore matches
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V8
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 8 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 8 --result "$OUT/result.json"
