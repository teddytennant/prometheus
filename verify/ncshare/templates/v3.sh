#!/bin/bash
# V3 precision (spec 16.2). 8 GPUs.
# Exit: FP8 loss within 0.5% of BF16; NVFP4 numerics only, never speed.
#SBATCH -J fv-{{RUN_ID}}-v3
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V3 template is an interface stub; implementer fills the body."
exit 1
