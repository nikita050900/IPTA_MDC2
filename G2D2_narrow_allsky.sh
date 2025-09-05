#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

detect_jobid=$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=G2D2_narrow_detect_allsky
#SBATCH --output=G2D2_narrow_detect_allsky.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
$PY runQuickMCMC_G2D2_all_sky.py --save_filename G2D2_detect_allsky.h5 --amplitude_prior detection
EOF
)
echo "Submitted detection job: ${detect_jobid}"

sbatch --dependency=afterok:${detect_jobid} <<EOF
#!/bin/bash
#SBATCH --job-name=G2D2_narrow_UL_allsky
#SBATCH --output=G2D2_narrow_UL_allsky.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
$PY runQuickMCMC_G2D2_all_sky.py --save_filename G2D2_UL_allsky.h5 --amplitude_prior UL
EOF
echo "Submitted UL job (afterok:${detect_jobid})"
