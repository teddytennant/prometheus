#!/bin/bash
# V3 precision (spec 16.2). 8 GPUs.
# Exit: FP8 loss within 0.5% of BF16; NVFP4 numerics only, never speed.
#SBATCH -J fv-{{RUN_ID}}-v3
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v3-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v3-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v3}"
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
from prometheus.verify.v3_precision import run_v3  # noqa: E402

result = run_v3(gpus=int("{{GPUS}}"))
payload = {
    "fp8_loss_rel_diff": float(result["fp8_loss_rel_diff"]),
    "nvfp4_numerics_ok": bool(result["nvfp4_numerics_ok"]),
}
(out / "v3.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
