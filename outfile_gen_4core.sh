#!/bin/bash
#SBATCH --job-name=outfile_gen_4core
#SBATCH --output=outfile_gen_4core.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=128G
#SBATCH --ntasks=1

source /shared/software/conda/etc/profile.d/conda.sh
conda activate /scratch/na00078/conda_envs/QuickCW
which python

BASE=/scratch/na00078/projects/IPTA_MDC2/h5_files
GEN=/scratch/na00078/projects/IPTA_MDC2/outfile_generator_4core.py

declare -a FILES=(
"G2D1_broad_detect_4core.h5"
"G2D1_narrow_detect_4core.h5"
"G2D1_broad_UL_4core.h5"
"G2D1_narrow_UL_4core.h5"
"G2D2_broad_detect_tref_4core.h5"
"G2D2_narrow_detect_tref_4core.h5"
"G2D2_detect_allsky_4core.h5"
"G2D2_broad_UL_loki_100M_lastTOA_4core.h5"
"G2D2_fixed_UL_loki_100M_lastTOA_ntol_10_4core.h5"
"G2D2_broad_detect_loki_100M_lastTOA_4core.h5"
"G2D2_fixed_detect_loki_100M_lastTOA_4core.h5"
)

for f in "${FILES[@]}"; do
  echo "=================================================="
  echo "Generating outfile for: $f"
  python "$GEN" "${BASE}/${f}"
done
echo "All outfiles complete."
