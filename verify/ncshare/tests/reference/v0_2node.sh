#!/bin/bash
# Independent V0 2-node × 4 H200 stub (spec 16.2 second allocation).
# Not a production template. The reference renderer only substitutes
# {{RUN_ID}} and {{WALLTIME}}.
#
# Login node has no AVX — venv is built inside this job on a compute node.
# Cross-node nccl-tests all_reduce is srun --mpi=pmix of all_reduce_perf_mpi
# (not srun -N 2 of the non-MPI all_reduce_perf).
# Checker reads busbw_gbps (2-node all_reduce) and node_facts.txt (kvm + NVMe).
# Do not set #SBATCH --gpus-per-task or srun --gpus-per-task: NCShare rejects
# combining typed --gres=gpu:h200:N with --gpus-per-task ("Invalid GRES
# specification (with and without type identification)"). One rank per GPU is
# --ntasks=8 and --ntasks-per-node=4 under --gres=gpu:h200:4.
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

# Login node has no AVX; create the venv on the compute node inside the job.
python3 -m venv "$OUT/venv"
# shellcheck disable=SC1091
source "$OUT/venv/bin/activate"
python -m pip install -q --upgrade pip

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
# srun --mpi=pmix of all_reduce_perf_mpi (no --gpus-per-task; see header).
NCCL_BIN="${NCCL_TESTS_ALL_REDUCE_MPI:-all_reduce_perf_mpi}"
NCCL_LOG="$OUT/nccl_allreduce_2node.txt"
srun --mpi=pmix "$NCCL_BIN" -b 8 -e 128M -f 2 -g 1 | tee "$NCCL_LOG"
cp "$NCCL_LOG" "$OUT/busbw_gbps"
