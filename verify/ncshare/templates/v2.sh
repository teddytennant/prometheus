#!/bin/bash
# V2 parallel equivalence (spec 16.2). 8 GPUs, then 2x4.
# Stop gpu-opportunist around this window.
# Exit: loss matches single-device to 1e-6 relative in FP32 over 200 steps; routing identical.
#SBATCH -J fv-{{RUN_ID}}-v2
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V2 template is an interface stub; implementer fills the body."
exit 1
