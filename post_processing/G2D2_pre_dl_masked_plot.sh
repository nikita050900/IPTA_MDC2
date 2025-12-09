#!/bin/bash
#SBATCH --job-name=G2D2_pre_dl_masked_plot
#SBATCH --output=G2D2_pre_dl_masked_plot.out
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=256G
#SBATCH --ntasks=1

PY="/scratch/na00078/conda_envs/QuickCW/bin/python"

$PY G2D2_pre_dl_masked_plot.py
done

echo "All done."
