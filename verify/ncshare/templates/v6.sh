#!/bin/bash
# V6 latent (spec 16.2). 4 to 8 GPUs.
# Exit: curriculum trains without collapse; accuracy rises with latent budget; thoughts decode.
#SBATCH -J fv-{{RUN_ID}}-v6
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v6-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v6-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v6}"
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
from prometheus.verify.v6_latent import run_v6  # noqa: E402

result = run_v6(gpus=int("{{GPUS}}"))
payload = {
    "curriculum_no_collapse": bool(result["curriculum_no_collapse"]),
    "accuracy_rises_with_latent_budget": bool(result["accuracy_rises_with_latent_budget"]),
    "thoughts_decode": bool(result["thoughts_decode"]),
}
(out / "v6.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
