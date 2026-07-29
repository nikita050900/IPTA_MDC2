#!/bin/bash
#SBATCH --job-name=dL_masking_4core
#SBATCH --output=dL_masking_4core.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1

source /shared/software/conda/etc/profile.d/conda.sh
conda activate /scratch/na00078/conda_envs/QuickCW
which python

BASE_PATH="/scratch/na00078/projects/IPTA_MDC2/h5_files"
SCRIPT_PATH="/scratch/na00078/projects/IPTA_MDC2"
SCRIPT_NAME="dL_masked_h5_file_generator_multiple.py"

declare -a TARGETS=(
"G2D1_broad_detect_4core.h5 75.4"
"G2D1_narrow_detect_4core.h5 75.4"
"G2D1_broad_UL_4core.h5 75.4"
"G2D1_narrow_UL_4core.h5 75.4"
"G2D2_broad_detect_tref_4core.h5 75.4"
"G2D2_narrow_detect_tref_4core.h5 75.4"
)

for entry in "${TARGETS[@]}"; do
  set -- $entry
  FILENAME=$1
  TARGET_DL=$2
  INFILE="${BASE_PATH}/${FILENAME}"
  BASE=$(basename "$FILENAME" .h5)
  OUTLOG="dl_masking_${BASE}_${TARGET_DL}Mpc.out"
  echo "--------------------------------------------------------"
  echo "Processing file: $INFILE"
  echo "Target dL: ${TARGET_DL} Mpc"
  echo "Output log: $OUTLOG"
  echo "--------------------------------------------------------"
  python ${SCRIPT_PATH}/${SCRIPT_NAME} "$INFILE" "$TARGET_DL" > "$OUTLOG" 2>&1
done
echo "All masking runs complete."
