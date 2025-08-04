#!/bin/bash

# Define viscosities (same as in rollout-initial-conditions.sh)
viscosities=("0.01" "0.005" "0.001" "0.0005" "0.0001" "0.00005" "0.00001")
numgpus=8

# Input and output directories
input_dir="/mnt/local_storage/fluids/rollouts/"
output_base_dir="./fluid-stats-downsampled"

# Function to format viscosity for output directory
format_viscosity_for_dir() {
    local visc=$1
    case $visc in
        "0.00001") echo "1e-05" ;;
        "0.00005") echo "5e-05" ;;
        "0.0001") echo "1e-04" ;;
        "0.0005") echo "5e-04" ;;
        *) echo "$visc" ;;
    esac
}

echo "Computing fluid statistics for ${#viscosities[@]} viscosities in parallel..."
echo "Input directory: $input_dir"
echo "Output base directory: $output_base_dir"

# Create output base directory if it doesn't exist
mkdir -p "$output_base_dir"

# Start jobs for all viscosities in parallel
pids=()
for i in "${!viscosities[@]}"; do
    gpu_id=$((i % numgpus))  # Distribute across GPUs (0-7)
    viscosity="${viscosities[$i]}"
    formatted_viscosity=$(format_viscosity_for_dir "$viscosity")
    
    # Create output directory for this viscosity
    output_dir="${output_base_dir}/visc=${formatted_viscosity}"
    mkdir -p "$output_dir"
    
    echo "Starting GPU $gpu_id: viscosity=$viscosity (formatted: $formatted_viscosity)"
    
    # Run fluids_stats.py with specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python fluids_stats.py \
        --loc "$input_dir" \
        --out_dir "$output_dir" \
        --downsample \
        --viscosity "$viscosity" &
    
    pids+=($!)
    
    # Add small delay to avoid simultaneous file access issues
    sleep 0.5
done

# Wait for all processes to complete
echo ""
echo "Waiting for all computations to complete..."
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    viscosity="${viscosities[$i]}"
    formatted_viscosity=$(format_viscosity_for_dir "$viscosity")
    
    if wait $pid; then
        echo "✓ Completed: viscosity=$viscosity (visc=${formatted_viscosity})"
    else
        echo "✗ Failed: viscosity=$viscosity (visc=${formatted_viscosity})"
    fi
done

echo ""
echo "All fluid statistics computations completed!"
echo "Results saved in: $output_base_dir/"

# List generated files
echo ""
echo "Generated energy spectrum plots:"
find "$output_base_dir" -name "*.png" -type f | sort