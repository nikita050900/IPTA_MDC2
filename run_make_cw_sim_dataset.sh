#!/bin/bash
#SBATCH --job-name=make_sim_dataset2
#SBATCH --output=logs/make_sim_dataset.out
#SBATCH -p sbs0016
#SBATCH --ntasks=1
#SBATCH --mem=64G

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"



# Path to your Python script
SCRIPT="/scratch/na00078/projects/IPTA_MDC2/psi_convention.py"

# Run the script directly in the conda environment
python "$SCRIPT"

echo "Job finished at: $(date)"
