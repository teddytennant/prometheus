#!/usr/bin/env bash
# V0 job template (spec 16.2). Exit criterion: measured bus bandwidth recorded; kvm presence known
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V0
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 0 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 0 --result "$OUT/result.json"
