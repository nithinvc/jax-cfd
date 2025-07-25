#!/bin/bash 

### Test runner that makes sure we can see all gpus and run properly ###

#SBATCH --job-name=baseline-10-03-2024-%j
#SBATCH --account m4319
#SBATCH -C gpu                       # GPU constraint. Use gpu&hbm80g for 80GB A100s
#SBATCH --gpus-per-node 4            # Number of GPUs per node - always fixed on NERSC
#SBATCH --qos premium 
#SBATCH -o logs/ns-generation-%j.out
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --cpus-per-task=64          # number of cores per task - NERSC always has 64. We also only have 1 task per node
#SBATCH --nodes 1                   # Number of nodes

## Notifications
#SBATCH --mail-user=nithinc@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00                 # 12 hours should be sufficient for generation and if it is not we can resubmit

source setupNERSC.sh

srun ./multiparameter-reverse.sh
