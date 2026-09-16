#!/usr/bin/env bash
# V4 job template (spec 16.2). Exit criterion: resume bitwise-equal; SDC hash catches flip; spike rollback skips shard
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE=V4
OUT="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}/runs/${STAGE}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
uv run python "$ROOT/verify/ncshare/run_stage.py" --stage 4 --out "$OUT/result.json"
uv run python "$ROOT/verify/ncshare/checkers.py" --stage 4 --result "$OUT/result.json"
