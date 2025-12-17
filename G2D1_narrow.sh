#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

# -------------------------------------------------------------
# 1. Submit detection job
# -------------------------------------------------------------
detect_jobid=$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=G2D1_narrow_detect
#SBATCH --output=G2D1_narrow_detect.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

$PY runQuickMCMC_G2D1_narrow.py \
    --save_filename G2D1_narrow_detect_fixed_gamma_17_dec_2025.h5 \
    --amplitude_prior detection
EOF
)

echo "Submitted detection job: ${detect_jobid}"

# -------------------------------------------------------------
# 2. Submit UL job, dependent on detection success
# -------------------------------------------------------------
ul_jobid=$(
  sbatch --parsable --dependency=afterok:${detect_jobid} <<EOF
#!/bin/bash
#SBATCH --job-name=G2D1_narrow_UL
#SBATCH --output=G2D1_narrow_UL.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

$PY runQuickMCMC_G2D1_narrow.py \
    --save_filename G2D1_narrow_UL_fixed_gamma_17_dec_2025.h5 \
    --amplitude_prior UL
EOF
)

echo "Submitted UL job (afterok:${detect_jobid}): ${ul_jobid}"
