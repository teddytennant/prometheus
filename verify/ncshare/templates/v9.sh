#!/bin/bash
# V9 lab dry run (spec 16.2). 1 to 4 GPUs.
# Exit: planted known-positive found and replicated; planted negative recorded as negative.
#SBATCH -J fv-{{RUN_ID}}-v9
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v9-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v9-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v9}"
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
from prometheus.verify.v9_lab import run_v9  # noqa: E402

result = run_v9(gpus=int("{{GPUS}}"))
payload = {
    "planted_positive_found": bool(result["planted_positive_found"]),
    "planted_positive_replicated": bool(result["planted_positive_replicated"]),
    "planted_negative_recorded": bool(result["planted_negative_recorded"]),
}
(out / "v9.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
