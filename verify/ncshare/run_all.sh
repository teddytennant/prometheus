#!/usr/bin/env bash
# Packed V0-V10 sbatch body. Runs on the allocation this script was submitted to.
set -euo pipefail
ROOT="${PROMETHEUS_ROOT:-/work/ttennant1/prometheus}"
PY="${JAX_VENV:-/hpc/home/ttennant1/arxiv-jax/venv}/bin/python"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export NVIDIA_TF32_OVERRIDE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
cd "$ROOT"
RUNS="${PROMETHEUS_RUNS:-$ROOT/runs}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
fail=0
# shellcheck disable=SC2086
for i in ${STAGES:-0 1 2 3 4 5 6 7 8 9 10}; do
  mkdir -p "$RUNS/V$i"
  if [ "$i" = 10 ]; then
    export SOAK_SECONDS="${SOAK_SECONDS:-180}"
  fi
  echo "=== V$i ==="
  if ! "$PY" "$ROOT/verify/ncshare/run_stage.py" --stage "$i" --out "$RUNS/V$i/result.json"; then
    echo "V$i run failed" >&2
    fail=1
    continue
  fi
  if ! "$PY" "$ROOT/verify/ncshare/checkers.py" --stage "$i" --result "$RUNS/V$i/result.json"; then
    echo "V$i checker failed" >&2
    if [ "$i" != 10 ]; then
      fail=1
    fi
  fi
done
exit "$fail"
