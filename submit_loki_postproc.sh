#!/bin/bash
#SBATCH --job-name=loki_postproc
#SBATCH --output=/scratch/na00078/projects/IPTA_MDC2/loki_postproc/loki_postproc_%j.log
#SBATCH --error=/scratch/na00078/projects/IPTA_MDC2/loki_postproc/loki_postproc_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=sbs0016

/usr/bin/time -v /scratch/na00078/conda_envs/QuickCW/bin/python \
    /scratch/na00078/projects/IPTA_MDC2/postproc_loki_vs_runF.py