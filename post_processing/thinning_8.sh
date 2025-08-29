#!/bin/bash
#SBATCH --job-name=thin_h5
#SBATCH --output=thin_h5_%j.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

INFILE_NAME="trial_pkl.h5"
OUTFILE_NAME="trial_pkl_outfile.h5"

$PY thinning_8_paras.py \
  --infile "$INFILE_NAME" \
  --outfile "$OUTFILE_NAME"
