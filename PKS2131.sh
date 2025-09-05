#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

detect_jobid=$(
  sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=PKS2131_detect
#SBATCH --output=PKS2131_detect.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
$PY runQuickMCMC_PKS2131.py --save_filename PKS2131_detect_narrow.h5 --amplitude_prior detection
EOF
)
echo "Submitted detection job: ${detect_jobid}"

sbatch --dependency=afterok:${detect_jobid} <<EOF
#!/bin/bash
#SBATCH --job-name=PKS2131_UL
#SBATCH --output=PKS2131_UL.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
$PY runQuickMCMC_PKS2131.py --save_filename PKS2131_UL_narrow.h5 --amplitude_prior UL
EOF
echo "Submitted UL job (afterok:${detect_jobid})"
