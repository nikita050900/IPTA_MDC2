#!/bin/bash
#SBATCH --job-name=pint_pkl
#SBATCH --output=pint_pkl.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

# Offline cache + ephemeris for PINT
export XDG_CACHE_HOME=/gpfs20/scratch/na00078/.cache
export ASTROPY_CACHE_DIR=/gpfs20/scratch/na00078/astropy_cache
export PINT_EPHEM=DE436

which python
#python runQuickMCMC_narrow_G2D1_bjorn_version_2.py
#python runQuickMCMC_narrow_G2D2.py
python G2D2_pint_pkl.py
echo "Run complete."
