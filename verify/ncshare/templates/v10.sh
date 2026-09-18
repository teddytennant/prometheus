#!/bin/bash
# V10 soak (spec 16.2). Shared. Chaos suite while V5 to V9 jobs run.
# Exit: 72h with no lost task, no duplicated output, no dead token.
#SBATCH -J fv-{{RUN_ID}}-v10
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --output=fv-{{RUN_ID}}-v10-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v10-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v10}"
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
from prometheus.verify.v10_soak import run_v10  # noqa: E402

result = run_v10(hours=72)
payload = {
    "hours": float(result["hours"]),
    "lost_tasks": float(result["lost_tasks"]),
    "duplicated_outputs": float(result["duplicated_outputs"]),
    "dead_tokens": float(result["dead_tokens"]),
}
(out / "v10.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
