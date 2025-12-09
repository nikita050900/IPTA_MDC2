#!/bin/bash
#SBATCH --job-name=skymap_corner_plot
#SBATCH --output=skymap_corner_plot.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=256G
#SBATCH --ntasks=1

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

$PY G2D2_detect_all_sky.py
done

echo "All done."
