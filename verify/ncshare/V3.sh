#!/usr/bin/env bash
# V3 job template (spec 16.2). This file is the sbatch body.
# Off-cluster it submits; on-cluster it runs the stage on the H200 allocation.
set -euo pipefail
STAGE_N=3
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
cd "$ROOT"
"$PY" "$ROOT/verify/ncshare/run_stage.py" --stage "$STAGE_N" --out "$OUT"
"$PY" "$ROOT/verify/ncshare/checkers.py" --stage "$STAGE_N" --result "$OUT"
