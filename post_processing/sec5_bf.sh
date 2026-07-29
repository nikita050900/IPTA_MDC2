#!/bin/bash
#SBATCH --job-name=sec5_bf
#SBATCH --output=sec5_bf.out
#SBATCH -p sbs0016
#SBATCH --mem=220G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
source /shared/software/conda/etc/profile.d/conda.sh
conda activate /scratch/na00078/conda_envs/QuickCW
cd /scratch/na00078/projects/IPTA_MDC2/post_processing
python -c "import corner, enterprise_extensions" || pip install corner enterprise_extensions
for s in run_broad_detect_4core run_fixed_detect_4core; do
  echo "===== $s $(date) ====="
  python $s.py > ${s}.slurm.log 2>&1
  echo "----- Bayes factors from $s -----"
  grep -iE "Loki:|BF10 =|Ent :|QCW :|limit ~" ${s}.slurm.log | tail -20
done
echo ALL_DONE
