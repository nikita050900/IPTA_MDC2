#!/bin/bash
#SBATCH --job-name=G2D1_narrow_detect_4core
#SBATCH --output=/scratch/na00078/projects/IPTA_MDC2/G2D1_narrow_detect_4core.out
#SBATCH -p sbs0016
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
cd /scratch/na00078/projects/IPTA_MDC2
source /shared/software/conda/etc/profile.d/conda.sh
conda activate /scratch/na00078/conda_envs/QuickCW
export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
echo "JOBID=$SLURM_JOB_ID NODE=$(hostname) ALLOC_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK NUMBA=$NUMBA_NUM_THREADS OMP=$OMP_NUM_THREADS TEMPO2=$TEMPO2"
lscpu | grep -iE "model name|socket\(s\)|core\(s\) per socket|thread\(s\) per core"
echo "START: $(date)"
/usr/bin/time -v /scratch/na00078/conda_envs/QuickCW/bin/python /scratch/na00078/projects/IPTA_MDC2/runQuickMCMC_G2D1_narrow_4core.py --save_filename G2D1_narrow_detect_4core.h5 --amplitude_prior detection
echo "END: $(date)"
