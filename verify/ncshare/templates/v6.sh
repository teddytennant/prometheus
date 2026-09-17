#!/bin/bash
# V6 latent (spec 16.2). 4 to 8 GPUs.
# Exit: curriculum trains without collapse; accuracy rises with latent budget; thoughts decode.
#SBATCH -J fv-{{RUN_ID}}-v6
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V6 template is an interface stub; implementer fills the body."
exit 1
