#!/usr/bin/env bash
set -euo pipefail
PY="/scratch/na00078/conda_envs/QuickCW/bin/python"
SCRIPT="/scratch/na00078/projects/IPTA_MDC2/runQuickMCMC_G2D2_broad.py"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=G2D2_broad_detect_tref
#SBATCH --output=/scratch/na00078/projects/IPTA_MDC2/G2D2_broad_detect_tref.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1
/usr/bin/time -v $PY $SCRIPT \
    --save_filename G2D2_broad_detect_tref_09_Jun_2026.h5 \
    --amplitude_prior detection
EOF
echo "Submitted G2D2 broad detection run with tref"
