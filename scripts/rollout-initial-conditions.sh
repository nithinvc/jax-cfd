#!/bin/bash

resolutions=(64 128 256 512 1024 2048)
viscosities=("0.01" "0.005" "0.001" "0.0005" "0.0001" "0.00005" "0.00001")
datadir="/mnt/local_storage/physicality/rollouts/"
numgpus=8

# Function to format viscosity for input file path
format_viscosity_for_file() {
    local visc=$1
    case $visc in
        "0.00001") echo "1e-5" ;;
        "0.00005") echo "5e-5" ;;
        *) echo "$visc" ;;
    esac
}

echo "Total resolutions: ${#resolutions[@]}"
echo "Total viscosities per resolution: ${#viscosities[@]}"

# Process each resolution with all viscosities in parallel
for resolution in "${resolutions[@]}"; do
    echo "Starting resolution $resolution with all viscosities in parallel..."
    
    # Start jobs for all viscosities in parallel
    pids=()
    for i in "${!viscosities[@]}"; do
        gpu_id=$((i % numgpus))  # Distribute across 8 GPUs (0-7)
        viscosity="${viscosities[$i]}"
        formatted_viscosity=$(format_viscosity_for_file "$viscosity")
        
        inputfile="/mnt/local_storage/physicality/initial-conditions/ns_2048x2048_visc_${formatted_viscosity}"
        
        echo "  GPU $gpu_id: resolution=$resolution, viscosity=$viscosity (file: $formatted_viscosity)"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python generate-navier-stokes-single-ic.py \
            --output_dir "$datadir" \
            --input_file "$inputfile" \
            --drag 0.1 \
            --simulation_time 4.0 \
            --save_dt 0.01 \
            --batch_size 4 \
            --forcing_func kolmogorov \
            --resolution "$resolution" \
            --kolmogorov_wavenumber 2 \
            --viscosity "$viscosity" &
        
        pids+=($!)
    done
    
    # Wait for all viscosities at this resolution to complete
    echo "Waiting for all viscosities at resolution $resolution to complete..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    echo "Resolution $resolution completed!"
done

echo "All resolutions and viscosities completed!"