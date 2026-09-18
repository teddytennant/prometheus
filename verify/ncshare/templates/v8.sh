#!/bin/bash
# V8 serving (spec 16.2). 1 to 4 GPUs.
# Exit: SGLang vs JAX log-probs within threshold; tiered session restore matches.
#SBATCH -J fv-{{RUN_ID}}-v8
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v8-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v8-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v8}"
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
from prometheus.verify.v8_serving import run_v8  # noqa: E402

result = run_v8(gpus=int("{{GPUS}}"))
payload = {
    "logprob_within_threshold": bool(result["logprob_within_threshold"]),
    "tiered_restore_matches": bool(result["tiered_restore_matches"]),
}
(out / "v8.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
