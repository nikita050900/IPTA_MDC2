#!/bin/bash
#SBATCH --job-name=G2D1_master
#SBATCH --output=G2D1_master.log
#SBATCH -p sbs0016
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

# Path to your QuickCW python executable (Python 3.6+)
PYTHON_PATH="/scratch/na00078/conda_envs/QuickCW/bin/python"

# Submit detection job first
detect_jobid=$(sbatch --job-name=G2D1_detect \
    --output=G2D1_detect.out \
    --parsable \
    --wrap="$PYTHON_PATH runQuickMCMC_narrow_G2D1.py --save_filename G2D2_detect_narrow.h5 --amplitude_prior detection")

echo "Submitted detection job with JobID: $detect_jobid"

# Submit UL job only after detection finishes successfully
sbatch --job-name=G2D1_UL \
    --output=G2D1_UL.out \
    --dependency=afterok:$detect_jobid \
    --wrap="$PYTHON_PATH runQuickMCMC_narrow_G2D1.py --save_filename G2D2_UL_narrow.h5 --amplitude_prior UL"

echo "Submitted UL job (will start after detection finishes successfully)"
