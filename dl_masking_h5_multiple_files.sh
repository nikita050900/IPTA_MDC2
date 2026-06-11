#!/bin/bash
#SBATCH --job-name=dL_masking_G2D2_09_Jun_2026
#SBATCH --output=dL_masking_G2D2_09_Jun_2026.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1
which python
# ---------------------- Paths ---------------------- #
BASE_PATH="/scratch/na00078/projects/IPTA_MDC2/h5_files"
SCRIPT_PATH="/scratch/na00078/projects/IPTA_MDC2"
SCRIPT_NAME="dL_masked_h5_file_generator_multiple.py"
# ---------------------- (infile, target_dL) pairs ---------------------- #
declare -a TARGETS=(
"G2D2_narrow_detect_tref_09_Jun_2026.h5 75.4"
"G2D2_broad_detect_tref_09_Jun_2026.h5 75.4"
)
# ---------------------- Loop over files ---------------------- #
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
echo "All runs complete."