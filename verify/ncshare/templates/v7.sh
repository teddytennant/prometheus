#!/bin/bash
# V7 RL end to end (spec 16.2). 8 GPUs (4 SGLang + 4 JAX).
# Exit: reward rises on a tiny verifiable task; log-prob drift halts RL; planted write flagged.
#SBATCH -J fv-{{RUN_ID}}-v7
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v7-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v7-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v7}"
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
from prometheus.verify.v7_rl import run_v7  # noqa: E402

result = run_v7(gpus=int("{{GPUS}}"))
payload = {
    "reward_rises": bool(result["reward_rises"]),
    "logprob_drift_halted": bool(result["logprob_drift_halted"]),
    "planted_write_flagged": bool(result["planted_write_flagged"]),
}
(out / "v7.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
