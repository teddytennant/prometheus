#!/bin/bash
# V0 environment (spec 16.2). 1 GPU first, then 2 nodes x 4.
# Exit: measured bus bandwidth recorded; known whether Firecracker can run there.
#SBATCH -J fv-{{RUN_ID}}-v0
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
# Build the venv inside the job. The login node has no AVX.
# sbatch --test-only of a dummy script.
# nccl-tests all_reduce intra-node, then across 2 nodes.
# Record /dev/kvm and NVMe facts.
# Measure queue wait from squeue submit to start.
echo "V0 template is an interface stub; implementer fills the body."
exit 1
