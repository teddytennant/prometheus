#!/bin/bash
# V4 fault tolerance (spec 16.2). 4 to 8 GPUs.
# Exit: resumed run bitwise-equal; SDC hash catches the flip; spike rollback skips the shard.
#SBATCH -J fv-{{RUN_ID}}-v4
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:{{GPUS}}
set -euo pipefail
echo "V4 template is an interface stub; implementer fills the body."
exit 1
