#!/bin/bash
# V10 soak (spec 16.2). Shared. Chaos suite while V5 to V9 jobs run.
# Exit: 72h with no lost task, no duplicated output, no dead token.
#SBATCH -J fv-{{RUN_ID}}-v10
#SBATCH -t {{WALLTIME}}
#SBATCH -p gpu
set -euo pipefail
echo "V10 template is an interface stub; implementer fills the body."
exit 1
