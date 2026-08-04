#!/bin/bash
#SBATCH --job-name=sec6_ess
#SBATCH --partition=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sec6_ess_%j.out
#SBATCH --error=sec6_ess_%j.err

source /users/na00078/.bashrc
conda activate /scratch/na00078/conda_envs/QuickCW

export PYTHONPATH=/scratch/na00078/projects/IPTA_MDC2:/scratch/na00078/projects/IPTA_MDC2/QuickCW
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS=4

cd /scratch/na00078/projects/IPTA_MDC2/post_processing
python sec6_ess.py
