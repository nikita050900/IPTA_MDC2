#!/bin/bash
#SBATCH --job-name=thin_h5_batch
#SBATCH --output=thin_h5_batch_%j.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

# List your input filenames (no paths)
FILES=(
  "G2D2_detect_seed4.h5"
  "G2D2_detect_seed1.h5"
  "G2D2_detect_seed2.h5"
  "G2D2_detect_seed3.h5"
  "G2D2_detect_seed5.h5"
  
  
  
)

for IN in "${FILES[@]}"; do
  # make outfile name by suffixing _thinned
  OUT="${IN%.h5}_outfile.h5"
  echo "Processing $IN -> $OUT"
  $PY thinning_8_params.py --infile "$IN" --outfile "$OUT"
done

echo "All done."
