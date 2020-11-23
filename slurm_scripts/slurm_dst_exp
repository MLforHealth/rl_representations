#!/bin/bash
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # All cores on a single node
#SBATCH --gres=gpu:1
#SBATCH -c 2 # Number of cpus requested
#SBATCH -p p100 # Partition to submit to
#SBATCH --output OUTPUTS/dst_sepsis-%j-%a.out
#SBATCH --mem=32GB
#SBATCH --array=1-140%140

echo $(tail -n+$SLURM_ARRAY_TASK_ID dst_exp_params.txt | head -n1)

cd ../scripts

python -u train_model.py $(tail -n+$SLURM_ARRAY_TASK_ID ../slurm_scripts/dst_exp_params.txt | head -n1)
