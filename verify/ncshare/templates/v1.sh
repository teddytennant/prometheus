#!/bin/bash
# V1 reference parity (spec 16.2). 1 GPU.
# Exit: JAX vs PyTorch reference logits match to 1e-5 in FP32; grad checks pass; overfits one batch.
#SBATCH -J fv-{{RUN_ID}}-v1
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v1-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v1-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v1}"
mkdir -p "$OUT"

python3 -m venv "$OUT/venv"
# shellcheck disable=SC1091
source "$OUT/venv/bin/activate"
python -m pip install -q --upgrade pip
cd "$ROOT"
python -m pip install -q -e ".[cuda]"

export PROMETHEUS_ROOT="$ROOT"
export VERIFY_OUT="$OUT"
python - <<'PY'
import json, os, sys
from pathlib import Path

import jax

assert any(d.platform == "gpu" for d in jax.devices()), "JAX is not using a GPU"

root = Path(os.environ["PROMETHEUS_ROOT"])
out = Path(os.environ["VERIFY_OUT"])
sys.path.insert(0, str(root))
from prometheus.verify.v1_parity import run_v1  # noqa: E402

result = run_v1(gpus=int("{{GPUS}}"))
payload = {
    "logits_max_diff": float(result["logits_max_diff"]),
    "grad_ok": bool(result["grad_ok"]),
    "overfit_ok": bool(result["overfit_ok"]),
}
(out / "v1.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
