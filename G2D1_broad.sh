#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

detect_jobid=$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=G2D1_broad_detect
#SBATCH --output=G2D1_broad_detect.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
$PY runQuickMCMC_G2D1_broad.py --save_filename G2D1_broad_detect_1e9.h5 --amplitude_prior detection
EOF
)
echo "Submitted detection job: ${detect_jobid}"

sbatch --dependency=afterok:${detect_jobid} <<EOF
#!/bin/bash
#SBATCH --job-name=G2D1_broad_UL
#SBATCH --output=G2D1_broad_UL.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
$PY runQuickMCMC_G2D1_broad.py --save_filename G2D1_broad_UL_1e9.h5 --amplitude_prior UL
EOF
echo "Submitted UL job (afterok:${detect_jobid})"
