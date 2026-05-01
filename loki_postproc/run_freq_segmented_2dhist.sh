#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=loki_2dhist_compare
#SBATCH --output=loki_2dhist_compare.out
#SBATCH --error=loki_2dhist_compare.err
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

cd /scratch/na00078/projects/IPTA_MDC2/loki_postproc
$PY compare_freq_segmented_UL_2dhist.py
EOF