#!/bin/bash
#SBATCH --job-name=bsl_meta
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=70:00:00

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate metamf_dp

echo "Using python: $( python --version ) from $( which python )"

# now run your program using python from the conda environment:
srun python3 /home/pm91887/MetaMF/run_baselines_meta.py