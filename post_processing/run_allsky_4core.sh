#!/bin/bash
#SBATCH --job-name=allsky_map
#SBATCH --partition=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=allsky_map_%j.out
#SBATCH --error=allsky_map_%j.err

module purge
module load lang/python/cpython_3.11.3_gcc122

export PYTHONPATH=/scratch/na00078/projects/IPTA_MDC2:/scratch/na00078/projects/IPTA_MDC2/QuickCW
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS=4

cd /scratch/na00078/projects/IPTA_MDC2/post_processing

python all_sky_healpy_4core.py