#!/bin/bash
#SBATCH --job-name=outfile_all_4core
#SBATCH --output=outfile_all_4core.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

source /shared/software/conda/etc/profile.d/conda.sh
conda activate /scratch/na00078/conda_envs/QuickCW
which python

H5=/scratch/na00078/projects/IPTA_MDC2/h5_files
DLM=$H5/dl_masked
MGEN=/scratch/na00078/projects/IPTA_MDC2/outfile_from_masked_4core.py
RGEN=/scratch/na00078/projects/IPTA_MDC2/outfile_generator_4core.py

# A to F: outfile extracted from the dL masked file
declare -a MASKED=(
"G2D1_broad_detect_4core"
"G2D1_narrow_detect_4core"
"G2D1_broad_UL_4core"
"G2D1_narrow_UL_4core"
"G2D2_broad_detect_tref_4core"
"G2D2_narrow_detect_tref_4core"
)
for B in "${MASKED[@]}"; do
  echo "==================== masked outfile: $B ===================="
  python "$MGEN" "$DLM/${B}_dLmasked_75.400Mpc.h5" "$H5/${B}.h5" "$H5/${B}_outfile.h5"
done

# G, L to O: outfile from raw chain
declare -a RAW=(
"G2D2_detect_allsky_4core"
"G2D2_broad_UL_loki_100M_lastTOA_4core"
"G2D2_fixed_UL_loki_100M_lastTOA_ntol_10_4core"
"G2D2_broad_detect_loki_100M_lastTOA_4core"
"G2D2_fixed_detect_loki_100M_lastTOA_4core"
)
for f in "${RAW[@]}"; do
  echo "==================== raw outfile: $f ===================="
  python "$RGEN" "$H5/${f}.h5"
done
echo "All outfiles complete."
