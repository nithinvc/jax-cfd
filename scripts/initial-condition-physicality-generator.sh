#!/bin/bash

datadir="/mnt/local_storage/physicality/initial-conditions/"
batchsize=4
viscosities=("0.01" "0.005" "0.001" "0.0005" "0.0001" "0.00005" "0.00001")
num_gpus=7
resolution=2048

echo "Total viscosities: ${#viscosities[@]}"
echo "Starting parallel jobs across GPUs 1-7..."

# Start jobs in parallel, one per GPU
pids=()
for i in "${!viscosities[@]}"; do
    gpu_id=$((i + 1))  # Use GPU IDs 1-7
    viscosity="${viscosities[$i]}"
    
    echo "  GPU $gpu_id: viscosity=$viscosity"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python generate-initial-condition.py \
        --output_dir $datadir \
        --drag 0.1 \
        --simulation_time 1.0 \
        --save_dt 0.25 \
        --batch_size $batchsize \
        --forcing_func kolmogorov \
        --resolution $resolution \
        --burn_in 41 \
        --kolmogorov_wavenumber 2 \
        --viscosity "$viscosity" &
    
    pids+=($!)
done

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All jobs completed!"
