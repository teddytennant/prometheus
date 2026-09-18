#!/bin/bash
# V4 fault tolerance (spec 16.2). 4 to 8 GPUs.
# Exit: resumed run bitwise-equal; SDC hash catches the flip; spike rollback skips the shard.
#SBATCH -J fv-{{RUN_ID}}-v4
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
#SBATCH --output=fv-{{RUN_ID}}-v4-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v4-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v4}"
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
from prometheus.verify.v4_fault import run_v4  # noqa: E402

result = run_v4(gpus=int("{{GPUS}}"))
payload = {
    "resumed_bitwise_equal": bool(result["resumed_bitwise_equal"]),
    "sdc_caught_flip": bool(result["sdc_caught_flip"]),
    "spike_rollback_skipped_shard": bool(result["spike_rollback_skipped_shard"]),
}
(out / "v4.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
