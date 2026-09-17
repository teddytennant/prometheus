#!/bin/bash
# V5 rung 0 (spec 16.2). 8 GPUs. Stop gpu-opportunist around this window.
# Exit: loss curve matches the ladder's small-scale fit; checkpoint/resume across jobs works.
#SBATCH -J fv-{{RUN_ID}}-v5
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V5 template is an interface stub; implementer fills the body."
exit 1
