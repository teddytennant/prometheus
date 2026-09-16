#!/usr/bin/env bash
# V5 job template (spec 16.2). This file is the sbatch body.
# Off-cluster it submits; on-cluster it runs the stage on the H200 allocation.
set -euo pipefail
STAGE_N=5
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  exec "$HERE/submit.sh" "$STAGE_N"
fi
ROOT="${PROMETHEUS_ROOT:-/work/ttennant1/prometheus}"
PY="${JAX_VENV:-/hpc/home/ttennant1/arxiv-jax/venv}/bin/python"
OUT="${RESULT_JSON:-$PWD/result.json}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CKPT_DIR="${CKPT_DIR:-$ROOT/runs/v5/ckpt}"
export TOKENS_TARGET="${TOKENS_TARGET:-20000000000}"
if [ -z "${MAX_SECONDS:-}" ]; then
  MAX_SECONDS=13800
fi
export MAX_SECONDS
mkdir -p "$CKPT_DIR"
cd "$ROOT"
"$PY" "$ROOT/verify/ncshare/run_stage.py" --stage "$STAGE_N" --out "$OUT"
mkdir -p "$ROOT/runs/v5"
cp "$OUT" "$ROOT/runs/v5/result.json"
"$PY" "$ROOT/verify/ncshare/checkers.py" --stage "$STAGE_N" --result "$OUT" || true
