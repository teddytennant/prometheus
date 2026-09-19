#!/bin/bash
# V0 2-node × 4 H200 (spec 16.2 second allocation).
# Distinct from templates/v0.sh (1-GPU first allocation).
# Build the venv inside the job. The login node has no AVX.
# sbatch --test-only of a dummy script.
# Cross-node nccl-tests all_reduce is srun --mpi=pmix of all_reduce_perf_mpi
# (not srun -N 2 of the non-MPI all_reduce_perf).
# Record /dev/kvm and NVMe facts.
# Measure queue wait from squeue submit to start.
# Exit: measured bus bandwidth recorded; known whether Firecracker can run there.
#SBATCH -J fv-{{RUN_ID}}-v0
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --output=fv-{{RUN_ID}}-v0-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v0-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v0}"
mkdir -p "$OUT"

{
  echo "job_id=${SLURM_JOB_ID:-none}"
  echo "submit_time=${SLURM_JOB_SUBMIT_TIME:-unknown}"
  echo "start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nodelist=${SLURM_JOB_NODELIST:-$(hostname)}"
  if command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    squeue -j "$SLURM_JOB_ID" -o "%i %V %S %M" || true
  fi
} > "$OUT/queue_wait.txt"

# Login node has no AVX; create the venv on the compute node.
python3 -m venv "$OUT/venv"
# shellcheck disable=SC1091
source "$OUT/venv/bin/activate"
python -m pip install -q --upgrade pip

DUMMY="$OUT/test_only.sh"
cat > "$DUMMY" <<'EOF'
#!/bin/bash
#SBATCH -J fv-v0-test-only
#SBATCH -t 00:01:00
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
true
EOF
sbatch --test-only "$DUMMY" > "$OUT/sbatch_test_only.txt" 2>&1

# Record /dev/kvm (Firecracker) and NVMe facts. Checker reads node_facts.txt.
{
  echo "hostname: $(hostname)"
  if [[ -e /dev/kvm ]]; then
    echo "kvm: yes"
    ls -l /dev/kvm || true
  else
    echo "kvm: no"
    ls -l /dev/kvm 2>&1 || true
  fi
  if ls /dev/nvme* >/dev/null 2>&1; then
    echo "nvme: present"
    ls -l /dev/nvme* || true
  else
    echo "nvme: absent"
  fi
} > "$OUT/node_facts.txt"

# 8 ranks, one per GPU. Multi-rank nccl-tests on this cluster only works as
# srun --mpi=pmix of all_reduce_perf_mpi.
NCCL_BIN="${NCCL_TESTS_ALL_REDUCE_MPI:-all_reduce_perf_mpi}"
NCCL_LOG="$OUT/nccl_allreduce_2node.txt"
if ! command -v "$NCCL_BIN" >/dev/null 2>&1; then
  echo "nccl-tests MPI all_reduce binary not found: $NCCL_BIN" >&2
  exit 1
fi
# PMIx env that works on NCShare (jobs 734353 / 734382). Job 734144 SIGSEGV'd
# in PMIx_Init (gds_shmem) after copying a tools-prefix MCA path.
export PMIX_MCA_gds=hash
unset OMPI_MCA_mca_base_component_path
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/openmpi/lib:${LD_LIBRARY_PATH:-}"
srun --mpi=pmix "$NCCL_BIN" -b 8 -e 128M -f 2 -g 1 | tee "$NCCL_LOG"
cp "$NCCL_LOG" "$OUT/busbw_gbps"
