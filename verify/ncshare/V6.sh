#!/usr/bin/env bash
# V6 job template (spec 16.2). Exit criterion: curriculum trains; held-out accuracy rises with latent budget; thoughts decode
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V6
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 6 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 6 --result "$OUT/result.json"
