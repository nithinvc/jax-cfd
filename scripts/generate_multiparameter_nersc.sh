#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/short_%j.out
#SBATCH --nodelist=germain

# Load conda environment properly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cfd2

# Check that we're using the right Python
which python
python --version

# Set environment variables
export PYTHONPATH=/home/divyam123/jaxcfd/jax-cfd:$PYTHONPATH

# Set the working directory to the project directory
cd ~/jaxcfd/jax-cfd/scripts

# source setupNERSC.sh

srun ./multiparameter.sh
