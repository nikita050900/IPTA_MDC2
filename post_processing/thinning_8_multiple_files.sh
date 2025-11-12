#!/bin/bash
#SBATCH --job-name=thin_h5_batch
#SBATCH --output=thin_h5_batch_%j.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

# List your input filenames (no paths)
FILES=(
  "G2D2_narrow_detect_simulated_psi_1p8850.h5"
  "G2D2_narrow_detect_simulated_psi_2p5133.h5"
  "G2D2_narrow_detect_simulated_psi_1p2566.h5"
  "G2D2_narrow_detect_simulated_psi_3p1416.h5"
  "G2D2_narrow_detect_simulated_psi_0p00pi.h5"
  "G2D2_narrow_detect_simulated_psi_0p6283.h5"
  
  
)

for IN in "${FILES[@]}"; do
  # make outfile name by suffixing _thinned
  OUT="${IN%.h5}_outfile.h5"
  echo "Processing $IN -> $OUT"
  $PY thinning_8_params.py --infile "$IN" --outfile "$OUT"
done

echo "All done."
