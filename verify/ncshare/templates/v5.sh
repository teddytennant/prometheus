#!/bin/bash
# V5 rung 0 (spec 16.2). 8 GPUs. Stop gpu-opportunist around this window.
# Exit: loss curve matches the ladder's small-scale fit; checkpoint/resume across jobs works.
#SBATCH -J fv-{{RUN_ID}}-v5
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v5-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v5-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v5}"
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
from prometheus.verify.v5_rung0 import run_v5  # noqa: E402

result = run_v5(gpus=int("{{GPUS}}"))
payload = {
    "loss_curve_matches_ladder": bool(result["loss_curve_matches_ladder"]),
    "checkpoint_resume_ok": bool(result["checkpoint_resume_ok"]),
}
(out / "v5.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
