#!/bin/bash
#SBATCH --job-name=slurm_metamf
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=70:00:00

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate metamf_dp

echo "Using python: $( python --version ) from $( which python )"

# now run your program using python from the conda environment:
#python run_model_meta.py
srun python3 /home/pm91887/MetaMF/hyperparameter_tuning_nometa.py