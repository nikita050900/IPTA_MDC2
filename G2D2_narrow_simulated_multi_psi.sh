#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"
SCRIPT="runQuickMCMC_G2D2_narrow_simulated_multi_psi.py"
BASE_PKL="/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/psr_objects"
OUT_DIR="/scratch/na00078/projects/IPTA_MDC2/h5_files"

# ψ grid tags
psi_tags=(
  "0p00pi"
  "0p6283"
  "1p2566"
  "1p8850"
  "2p5133"
  "3p1416"
)

for tag in "${psi_tags[@]}"; do
  PKL_FILE="${BASE_PKL}/G2D2_simulated_all_pulsars_psi_${tag}.pkl"
  OUT_FILE="G2D2_narrow_detect_simulated_psi_${tag}.h5"
  OUT_LOG="G2D2_${tag}.out"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=detect_${tag}
#SBATCH --output=${OUT_LOG}
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

export DATA_PKL="${PKL_FILE}"

$PY $SCRIPT --save_filename ${OUT_FILE} --amplitude_prior detection
EOF

done

echo "Submitted 6 detection-only QuickCW jobs with custom .out filenames."
