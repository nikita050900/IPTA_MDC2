#!/bin/bash
#SBATCH --job-name=thin_h5_batch
#SBATCH --output=thin_h5_batch_%j.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=256G
#SBATCH --ntasks=1

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

# List your input filenames (no paths)
FILES=(
  "G2D1_narrow_detect_fixed_gamma_17_dec_2025.h5"
  "G2D2_broad_detect_no_tref_17_Dec_2025.h5"
  "G2D1_narrow_UL_fixed_gamma_17_dec_2025.h5"
  "G2D2_broad_UL_no_tref_17_Dec_2025.h5"
  
  
  
)

for IN in "${FILES[@]}"; do
  # make outfile name by suffixing _thinned
  OUT="${IN%.h5}_outfile.h5"
  echo "Processing $IN -> $OUT"
  $PY thinning_8_new.py --infile "$IN" --outfile "$OUT"
done

echo "All done."
