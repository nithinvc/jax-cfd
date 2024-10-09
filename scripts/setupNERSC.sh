# Variety of setup commands to setup nersc environment

# Ensure normal bash is setup for mamba
source ~/.bashrc 

# CUDA
module load cudatoolkit/12.4
module load cudnn/9.1.0

# Activate mamba environment
micromamba activate jax-cfd

# Helpful environment variables
export GPUS_PER_NODE=4