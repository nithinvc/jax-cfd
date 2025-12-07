#!/bin/bash
#SBATCH --job-name=evaluation-dec-6-%a
#SBATCH --account m4558
#SBATCH -C gpu                       # GPU constraint. Use gpu&hbm80g for 80GB A100s
#SBATCH --gpus=1                     # 1 GPU per task - allows multiple tasks per node
#SBATCH --qos debug 
#SBATCH -o dev-logs/evaluation-dec-6-task-%a-%j.out
#SBATCH --ntasks=1                   # 1 task per array job
#SBATCH --cpus-per-task=16           # 64 cores / 4 GPUs = 16 cores per GPU
#SBATCH --array=0-1                 # 3 seeds × 4 viscosities = 12 jobs

## Notifications
#SBATCH --mail-user=nithinc@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00                 # 12 hours should be sufficient for generation and if it is not we can resubmit

# dec 6, 2025 - script to get parameter predictions
# Uses job arrays to parallelize: 3 seeds × 4 viscosities = 12 parallel jobs
# Each task uses 1 GPU, so up to 4 tasks can run on a single node

### consts
batchsize=4
resolution=2048
downsample=-1
numtraj=4

# Compute seed index (0, 1, or 2) and viscosity index (0, 1, 2, or 3)
seed_index=$((SLURM_ARRAY_TASK_ID / 4))
visc_index=$((SLURM_ARRAY_TASK_ID % 4))

# Per-seed viscosities and output paths
seed_zero_viscosities=("0.0001" "0.001" "0.0005" "0.005")
seed_zero_out_paths=("ic_0_param_0.0001" "ic_0_param_0.001" "ic_0_param_0.0005" "ic_0_param_0.005")

seed_one_viscosities=("0.0001" "0.001" "0.0005" "0.005")
seed_one_out_paths=("ic_0_param_0.0001" "ic_0_param_0.001" "ic_0_param_0.0005" "ic_0_param_0.005")

seed_two_viscosities=("0.0001" "0.001" "0.0005" "0.005")
seed_two_out_paths=("ic_0_param_0.0001" "ic_0_param_0.001" "ic_0_param_0.0005" "ic_0_param_0.005")

# Select datadir, viscosity, and out_path based on seed_index
if [ $seed_index -eq 0 ]; then
    seed=0
    datadir=/global/cfs/cdirs/m4558/shared/meta-pde/evals/seed_0
    viscosity="${seed_zero_viscosities[$visc_index]}"
    out_path="${seed_zero_out_paths[$visc_index]}"
elif [ $seed_index -eq 1 ]; then
    seed=1
    datadir=/global/cfs/cdirs/m4558/shared/meta-pde/evals/seed_1
    viscosity="${seed_one_viscosities[$visc_index]}"
    out_path="${seed_one_out_paths[$visc_index]}"
elif [ $seed_index -eq 2 ]; then
    seed=2
    datadir=/global/cfs/cdirs/m4558/shared/meta-pde/evals/seed_2
    viscosity="${seed_two_viscosities[$visc_index]}"
    out_path="${seed_two_out_paths[$visc_index]}"
fi

echo "Running task $SLURM_ARRAY_TASK_ID: seed=$seed, viscosity=$viscosity, out_path=$out_path"
source setupNERSC.sh
python generate-navier-stokes.py \
    --output_dir $datadir/$out_path \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --num_trajectories $numtraj \
    --batch_size $batchsize \
    --forcing_func kolmogorov \
    --resolution $resolution \
    --burn_in 41 \
    --kolmogorov_wavenumber 2 \
    --viscosity "$viscosity" \
    --downsample $downsample
