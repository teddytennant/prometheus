#!/bin/bash
# V9 lab dry run (spec 16.2). 1 to 4 GPUs.
# Exit: planted known-positive found and replicated; planted negative recorded as negative.
#SBATCH -J fv-{{RUN_ID}}-v9
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V9 template is an interface stub; implementer fills the body."
exit 1
