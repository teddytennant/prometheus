#!/bin/bash
# Independent V0 2-node stub (spec 16.2 second allocation).
# Substitutes {{RUN_ID}} and {{WALLTIME}} only. GPU counts are fixed:
# 2 nodes × 4 H200s, 8 MPI ranks (one per GPU).
#
# Login node has no AVX — venv is built inside this job on a compute node.
# Intra-node nccl-tests all_reduce is non-MPI all_reduce_perf -g 4 on one
# node (not srun -N 2 of the non-MPI binary). Override with NCCL_TESTS_ALL_REDUCE.
# Intra stdout is a separate file; busbw_gbps is the 2-node measurement.
# Cross-node nccl-tests all_reduce is srun --mpi=pmix of all_reduce_perf_mpi
# (not srun -N 2 of the non-MPI all_reduce_perf; not an mpirun wrapper).
# Override the MPI binary with NCCL_TESTS_ALL_REDUCE_MPI; do not hardcode a site
# path as the only way to find either ELF.
# PMIx env that works on NCShare (jobs 734353 / 734382): export PMIX_MCA_gds=hash,
# unset OMPI_MCA_mca_base_component_path, LD_LIBRARY_PATH includes openmpi/lib.
# Checker reads busbw_gbps (2-node all_reduce) and node_facts.txt (kvm + NVMe
# recorded on every allocated compute node via srun -N 2 --ntasks-per-node=1).
# Do not set #SBATCH --gpus-per-task or srun --gpus-per-task: NCShare rejects
# combining typed --gres=gpu:h200:N with --gpus-per-task ("Invalid GRES
# specification (with and without type identification)").
#SBATCH -J fv-{{RUN_ID}}-v0-2node
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --output=fv-{{RUN_ID}}-v0-2node-%j.out
#SBATCH --error=fv-{{RUN_ID}}-v0-2node-%j.err
set -euo pipefail

ROOT="${PROMETHEUS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT="${VERIFY_OUTPUT_DIR:-$ROOT/verify-out/{{RUN_ID}}/v0-2node}"
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

# Login node has no AVX; create the venv on a compute node.
python3 -m venv "$OUT/venv"
# shellcheck disable=SC1091
source "$OUT/venv/bin/activate"
python -m pip install -q --upgrade pip

# Record /dev/kvm (Firecracker) and NVMe facts on every allocated compute
# node, not only the batch-script host. Spec 16.2: "on compute nodes".
srun -N 2 --ntasks-per-node=1 bash -s <<'EOF' > "$OUT/node_facts.txt"
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
EOF

# Intra-node nccl-tests all_reduce: 4 GPUs on one node, non-MPI
# all_reduce_perf -g 4 (not srun -N 2 of the non-MPI binary).
# Override with NCCL_TESTS_ALL_REDUCE. Stdout is a separate file;
# busbw_gbps is the 2-node measurement.
NCCL_INTRA_BIN="${NCCL_TESTS_ALL_REDUCE:-all_reduce_perf}"
NCCL_INTRA_LOG="$OUT/nccl_allreduce_intra.txt"
"$NCCL_INTRA_BIN" -b 8 -e 128M -f 2 -g 4 | tee "$NCCL_INTRA_LOG"

# 8 ranks, one per GPU. Multi-rank nccl-tests on this cluster only works as
# srun --mpi=pmix of all_reduce_perf_mpi (no --gpus-per-task; see header).
# Binary override is NCCL_TESTS_ALL_REDUCE_MPI; do not hardcode a site path
# as the only way to find the ELF.
NCCL_BIN="${NCCL_TESTS_ALL_REDUCE_MPI:-all_reduce_perf_mpi}"
NCCL_LOG="$OUT/nccl_allreduce_2node.txt"

# PMIx env that actually works on NCShare (jobs 734353 / 734382).
# Job 734144 copied a tools-prefix MCA path and launched via mpirun:
# PMIX_ERR_FILE_OPEN_FAILURE then SIGSEGV in PMIx_Init (gds_shmem).
export PMIX_MCA_gds=hash
unset OMPI_MCA_mca_base_component_path
# System OpenMPI lib dir on LD_LIBRARY_PATH (`openmpi/lib` is enough).
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/openmpi/lib:${LD_LIBRARY_PATH:-}"

srun --mpi=pmix "$NCCL_BIN" -b 8 -e 128M -f 2 -g 1 | tee "$NCCL_LOG"
cp "$NCCL_LOG" "$OUT/busbw_gbps"
