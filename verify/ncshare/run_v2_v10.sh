#!/usr/bin/env bash
# Retry V2 (mesh) and V10 (soak) on the same 2-GPU allocation.
set -euo pipefail
ROOT="${PROMETHEUS_ROOT:-/work/ttennant1/prometheus}"
export STAGES="2 10"
export SOAK_SECONDS="${SOAK_SECONDS:-180}"
exec "$ROOT/verify/ncshare/run_all.sh"
