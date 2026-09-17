#!/bin/bash
# V7 RL end to end (spec 16.2). 8 GPUs (4 SGLang + 4 JAX).
# Exit: reward rises on a tiny verifiable task; log-prob drift halts RL; planted write flagged.
#SBATCH -J fv-{{RUN_ID}}-v7
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V7 template is an interface stub; implementer fills the body."
exit 1
