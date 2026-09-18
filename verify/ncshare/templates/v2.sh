#!/bin/bash
# V2 parallel equivalence (spec 16.2). 8 GPUs, then 2x4.
# Stop gpu-opportunist around this window.
# Exit: loss matches single-device to 1e-6 relative in FP32 over 200 steps; routing identical.
#SBATCH -J fv-{{RUN_ID}}-v2
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v2-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v2-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v2}"
mkdir -p "$OUT"

python3 -m venv "$OUT/venv"
# shellcheck disable=SC1091
source "$OUT/venv/bin/activate"
python -m pip install -q --upgrade pip
if [[ -f "$ROOT/pyproject.toml" ]]; then
  python -m pip install -q -e "$ROOT"
fi

export PROMETHEUS_ROOT="$ROOT"
export VERIFY_OUT="$OUT"
python - <<'PY'
import json, os, sys
from pathlib import Path

root = Path(os.environ["PROMETHEUS_ROOT"])
out = Path(os.environ["VERIFY_OUT"])
sys.path.insert(0, str(root))
from prometheus.verify.v2_parallel import run_v2  # noqa: E402

result = run_v2(gpus=int("{{GPUS}}"), steps=200)
payload = {
    "loss_rel_diff": float(result["loss_rel_diff"]),
    "routing_identical": bool(result["routing_identical"]),
}
(out / "v2.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
