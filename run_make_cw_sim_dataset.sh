#!/bin/bash
#SBATCH --job-name=make_sim_dataset2
#SBATCH --output=logs/make_sim_dataset2_%A_%a.out
#SBATCH --array=0-1
#SBATCH -p sbs0016
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G

export USE_ROTATED=${SLURM_ARRAY_TASK_ID}
echo "Running make_cw_sim_dataset_from_enterprise.py with USE_ROTATED=${USE_ROTATED}"
python make_cw_sim_dataset_libstempo.py
