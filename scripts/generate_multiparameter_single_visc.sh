#!/bin/bash

datadir="/mnt/local_storage/physicality/"
batchsize=4
numtraj=4
viscosities=("0.01" "0.001" "0.0001" "0.00001")
#resolutions=(64 128 256 512 1024 2048 4096)
resolutions=(1024 2048 4096)
num_gpus=8

# Create job queue
job_queue=()
for resolution in "${resolutions[@]}"; do
    for i in "${!viscosities[@]}"; do
        job_queue+=("$resolution:${viscosities[$i]}")
    done
done

echo "Total jobs: ${#job_queue[@]}"

# Process jobs in batches of 8
job_index=0
batch_num=1

while [[ $job_index -lt ${#job_queue[@]} ]]; do
    echo "Starting batch $batch_num..."
    
    # Start up to 8 jobs in parallel
    pids=()
    for gpu_id in $(seq 0 $((num_gpus-1))); do
        if [[ $job_index -lt ${#job_queue[@]} ]]; then
            job_info="${job_queue[$job_index]}"
            resolution=$(echo $job_info | cut -d':' -f1)
            viscosity=$(echo $job_info | cut -d':' -f2)
            
            echo "  GPU $gpu_id: resolution=$resolution, viscosity=$viscosity"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python generate-navier-stokes.py \
            --output_dir $datadir \
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
            --max_resolution 4096 &
            
            pids+=($!)
            ((job_index++))
        fi
    done
    
    # Wait for all jobs in this batch to complete
    echo "Waiting for batch $batch_num to complete..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    echo "Batch $batch_num completed!"
    ((batch_num++))
done

echo "All jobs completed!"