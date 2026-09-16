#!/usr/bin/env bash
# V10 job template (spec 16.2). This file is the sbatch body.
# Off-cluster it submits; on-cluster it runs the stage on the H200 allocation.
set -euo pipefail
STAGE_N=10
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
export SOAK_STATE="${SOAK_STATE:-$ROOT/runs/v10/soak_state.json}"
mkdir -p "$(dirname "$SOAK_STATE")"
if [ -z "${SOAK_SECONDS:-}" ]; then
  left=""
  if [[ "${SLURM_JOB_END_TIME:-}" =~ ^[0-9]+$ ]]; then
    left=$(( SLURM_JOB_END_TIME - $(date +%s) - 120 )) || left=""
  elif [[ "${SLURM_TIMELIMIT:-}" =~ ^[0-9]+$ ]]; then
    left=$(( SLURM_TIMELIMIT * 60 - 120 )) || left=""
  fi
  if [ -n "${left}" ] && [ "$left" -gt 60 ]; then
    SOAK_SECONDS=$left
  else
    SOAK_SECONDS=169200
  fi
fi
export SOAK_SECONDS
cd "$ROOT"
"$PY" "$ROOT/verify/ncshare/run_stage.py" --stage "$STAGE_N" --out "$OUT"
mkdir -p "$ROOT/runs/v10"
cp "$OUT" "$ROOT/runs/v10/result.json"
"$PY" "$ROOT/verify/ncshare/checkers.py" --stage "$STAGE_N" --result "$OUT" || true
