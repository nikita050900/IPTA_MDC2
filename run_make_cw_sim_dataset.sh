#!/bin/bash
#SBATCH --job-name=make_sim_dataset2
#SBATCH --output=logs/make_sim_dataset2_%j.out
#SBATCH -p sbs0016
#SBATCH --ntasks=1
#SBATCH --mem=64G


# Load Singularity on host (not inside container)
module load singularity/3.7.0

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

CONTAINER="/shared/containers/nanograv_singularity.sif"
SCRIPT="/scratch/na00078/projects/IPTA_MDC2/make_cw_sim_dataset_libstempo.py"

# Run script inside container
singularity exec "$CONTAINER" python "$SCRIPT"

echo "Job finished at: $(date)"
