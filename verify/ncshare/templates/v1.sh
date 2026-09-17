#!/bin/bash
# V1 reference parity (spec 16.2). 1 GPU.
# Exit: JAX vs PyTorch reference logits match to 1e-5 in FP32; grad checks pass; overfits one batch.
#SBATCH -J fv-{{RUN_ID}}-v1
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V1 template is an interface stub; implementer fills the body."
exit 1
