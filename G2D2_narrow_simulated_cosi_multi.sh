#!/usr/bin/env bash
set -euo pipefail

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"
SCRIPT="runQuickMCMC_G2D2_narrow_simulated_cosi_multi.py"
BASE_PKL="/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/psr_objects"

cosi_tags=(
  "0p00"
  "0p71"
  "1p00"
)

for tag in "${cosi_tags[@]}"; do

  PKL_FILE="${BASE_PKL}/G2D2_simulated_all_pulsars_cosi_${tag}.pkl"
  OUT_FILE="G2D2_narrow_detect_simulated_cosi_${tag}.h5"
  OUT_LOG="G2D2_cosi_${tag}.out"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=detect_cosi_${tag}
#SBATCH --output=${OUT_LOG}
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

export DATA_PKL="${PKL_FILE}"

$PY $SCRIPT --save_filename ${OUT_FILE} --amplitude_prior detection
EOF

done

echo "Submitted 3 detection-only QuickCW jobs for cos(i) sweep."
