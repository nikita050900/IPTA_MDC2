#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=plots_G2D2
#SBATCH --output=plots_G2D2.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=32G
#SBATCH --ntasks=1
$PY plots.py
EOF

echo "Submitted plotting job."
