#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

for SEED in 1 2 3 4 5; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=G2D2_det_seed${SEED}
#SBATCH --output=G2D2_det_seed${SEED}.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

export NOISE_SEED=${SEED}
${PY} runQuickMCMC_G2D2_narrow_simulated.py \
  --save_filename G2D2_detect_seed${SEED}.h5

EOF

  echo "Submitted detection job for seed ${SEED}"
done
