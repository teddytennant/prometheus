#!/bin/bash
# V8 serving (spec 16.2). 1 to 4 GPUs.
# Exit: SGLang vs JAX log-probs within threshold; tiered session restore matches.
#SBATCH -J fv-{{RUN_ID}}-v8
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V8 template is an interface stub; implementer fills the body."
exit 1
