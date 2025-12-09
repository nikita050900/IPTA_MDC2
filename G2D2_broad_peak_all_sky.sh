#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

# ------------------------------------------------------------
# Submit ONLY the detection analysis
# ------------------------------------------------------------
detect_jobid=$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=G2D2_peak_all_sky
#SBATCH --output=G2D2_peak_all_sky.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

$PY runQuickMCMC_G2D2_broad_peak_all_sky.py \
    --save_filename G2D2_broad_peak_all_sky.h5 \
    --amplitude_prior detection
EOF
)

echo "Submitted detection job: ${detect_jobid}"
