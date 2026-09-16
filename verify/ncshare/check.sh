#!/usr/bin/env bash
# Completeness check for S1/S2 code (spec 15.6) and V0-V10 artifacts (spec 16).
# Exit 0 only when the independent review's machine checks would pass.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STATE="${PROMETHEUS_STATE:-$HOME/.local/state/prometheus-build}"
fail=0
say() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

trees=(
  contracts model train parallel kernels ckpt control data tokenizer
  synth serve sglang-fork verifiers
  rl/coordinator rl/loss rl/rewards rl/envs rl/tasks rl/multiagent
  arc evals audit obs
  harness/swarm harness/providers harness/slurm harness/ledger
  harness/pipeline harness/chaos harness/ops harness/kernel
  harness/eval-gate harness/monitors harness/genome-seed
  verify/ncshare
  data/extract data/loader data/dedup data/classifiers
  data/decontam data/mixture
)

say "check: required trees"
for t in "${trees[@]}"; do
  if [ ! -d "$ROOT/$t" ]; then
    bad "missing tree $t"
    continue
  fi
  n=$(find "$ROOT/$t" -type f \( -name '*.py' -o -name '*.rs' -o -name '*.sh' -o -name '*.md' \) ! -path '*/target/*' ! -path '*/__pycache__/*' | wc -l)
  if [ "$n" -lt 1 ]; then
    bad "empty tree $t"
  fi
done

say "check: no production NotImplementedError stubs"
if grep -R --include='*.py' --include='*.rs' -n 'raise NotImplementedError\|unimplemented!\|todo!' \
    "$ROOT/train" "$ROOT/kernels" "$ROOT/parallel" "$ROOT/ckpt" "$ROOT/control" \
    "$ROOT/rl" "$ROOT/verify" "$ROOT/serve" "$ROOT/tokenizer" "$ROOT/synth" \
    "$ROOT/verifiers" "$ROOT/arc" "$ROOT/audit" "$ROOT/obs" "$ROOT/sglang-fork" \
    "$ROOT/data/dedup" "$ROOT/data/classifiers" "$ROOT/data/decontam" "$ROOT/data/mixture" \
    "$ROOT/harness/swarm" "$ROOT/harness/providers" "$ROOT/harness/pipeline" \
    "$ROOT/harness/chaos" "$ROOT/harness/ops" "$ROOT/harness/kernel" \
    "$ROOT/harness/eval-gate" "$ROOT/harness/monitors" "$ROOT/harness/genome-seed" \
    2>/dev/null | grep -v '/tests/' | grep -v '#' | head; then
  bad "stub markers in production"
fi

say "check: V0-V10 templates and checkers"
for i in $(seq 0 10); do
  [ -f "$ROOT/verify/ncshare/V${i}.sh" ] || bad "missing verify/ncshare/V${i}.sh"
done
[ -f "$ROOT/verify/ncshare/checkers.py" ] || bad "missing checkers.py"

say "check: V0-V10 H200 results (not CPU stand-ins)"
for i in $(seq 0 10); do
  f="$STATE/runs/V${i}/result.json"
  if [ ! -f "$f" ]; then
    bad "missing $f"
    continue
  fi
  if ! python3 - "$f" "$i" <<'PY'
import json, sys
p, stage = sys.argv[1], int(sys.argv[2])
r = json.loads(open(p).read())
errs = []
if "H200" not in str(r.get("gpu_name", "")).upper():
    errs.append("gpu_name is not H200")
if not r.get("slurm_job_id"):
    errs.append("missing slurm_job_id")
if r.get("tiny_standin"):
    errs.append("tiny_standin")
if r.get("standin"):
    errs.append("standin")
if stage == 0:
    if str(r.get("nccl_backend", "")).lower() in ("", "numpy", "cpu"):
        errs.append("numpy/CPU NCCL")
    if int(r.get("n_devices", 0)) < 2:
        errs.append("V0 n_devices < 2")
if stage == 2:
    if r.get("identity_mesh") or int(r.get("mesh_size", 1)) <= 1:
        errs.append("identity mesh")
    if int(r.get("n_devices", 0)) < 2:
        errs.append("V2 n_devices < 2")
if errs:
    print("; ".join(errs))
    sys.exit(1)
PY
  then
    bad "V$i result is not an H200 run"
  fi
done

say "check: progress.md"
prog="$STATE/progress.md"
if [ ! -f "$prog" ]; then
  bad "missing progress.md"
else
  if grep -q 'No V stages yet' "$prog"; then
    bad "progress.md still says No V stages yet"
  fi
  if grep -q 'Remaining S1/S2 modules' "$prog"; then
    bad "progress.md still says Remaining S1/S2 modules"
  fi
fi

say "check: HEAD past F3 interface-only"
cd "$ROOT"
head_msg=$(git log -1 --pretty=%s)
if [ "$head_msg" = "Merge F3 eval harness" ]; then
  bad "HEAD is still Merge F3 eval harness"
fi

if [ "$fail" -ne 0 ]; then
  say "COMPLETE: no"
  exit 1
fi
say "COMPLETE: yes"
exit 0
