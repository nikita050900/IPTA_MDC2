#!/bin/bash
#SBATCH --job-name=thin_h5_batch
#SBATCH --output=thin_h5_batch_%j.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

# List your input filenames (no paths)
FILES=(
  "3C120.h5"
  "NGC3115.h5"
  "3C66B.h5"
)

for IN in "${FILES[@]}"; do
  # make outfile name by suffixing _thinned
  OUT="${IN%.h5}_outfile.h5"
  echo "Processing $IN -> $OUT"
  $PY thinning_8_paras.py --infile "$IN" --outfile "$OUT"
done

echo "All done."
