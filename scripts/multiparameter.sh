#!/bin/bash
gpuone=0
gputwo=1
gputhree=2
gpufour=3

datadir="/data/divyam123/meta-pde/turbulent-flow_short_2"
batchsize=1
resolution=2048 #512
downsample=1
save_dt=0.01
num_trajectories=1
simulation_time=2

all_viscosities=(
    "0.0011290133559092666"
    "0.001319994226153501"
    "0.001331121608073689"
    "0.0013795402040204172"
    "0.0015382308040279"
    "0.001570297088405539"
    "0.0018477934173519257"
    "0.0019158219548093154"
    "0.002055424552015075"
    "0.0020736445177905022"
    "0.002115429079726122"
    "0.0022673986523780395"
    "0.002550298070162891"
    "0.0025764174425233167"
    "0.002656813924114493"
    "0.002661901888489057"
    "0.003063462210622081"
    "0.0038842777547031426"
    "0.003967605077052989"
    "0.0045881565491609705"
    "0.004760767751809498"
    "0.004835952776465951"
    "0.005280796376895365"
    "0.006153085601625313"
    "0.006584106160121612"
    "0.00314288089084011"
    "0.00582938454299474"
    "0.004115113049561088"
    "0.0020597335357437196"
    "0.00534516611064682"
    "0.0027950159165083337"
    "0.0015446089075047066"
    "0.007025166339242158"
    "0.0071144760093434225"
    "0.00788671412999049"
    "0.008105016126411584"
    "0.008123245085588688"
    "0.009133995846860976"
)

# export JAX_TRACEBACK_FILTERING="off"
# export JAX_DISABLE_JIT=0
export CUDA_VISIBLE_DEVICES=9

# for viscosity in "${all_viscosities[@]}"; do
#     echo "Running simulation for viscosity: $viscosity"
#     python3 generate-navier-stokes.py \
#         --output_dir $datadir \
#         --drag 0.1 \
#         --simulation_time $simulation_time \
#         --save_dt $save_dt \
#         --num_trajectories $num_trajectories \
#         --batch_size $batchsize \
#         --forcing_func kolmogorov \
#         --resolution $resolution \
#         --burn_in 0.1 \
#         --kolmogorov_wavenumber 2 \
#         --viscosity "$viscosity" \
#         --downsample $downsample
# done






# python3 generate-navier-stokes.py \
#     --output_dir $datadir \
#     --drag 0.1 \
#     --simulation_time $simulation_time \
#     --save_dt $save_dt \
#     --num_trajectories $num_trajectories \
#     --batch_size $batchsize \
#     --forcing_func kolmogorov \
#     --resolution $resolution \
#     --burn_in 0.1 \
#     --kolmogorov_wavenumber 2 \
#     --viscosity "1.0388823104027935e-05" \
#     --downsample $downsample

python3 generate-navier-stokes.py \
    --output_dir $datadir \
    --drag 0.1 \
    --simulation_time $simulation_time \
    --save_dt $save_dt \
    --num_trajectories $num_trajectories \
    --batch_size $batchsize \
    --forcing_func kolmogorov \
    --resolution $resolution \
    --burn_in 0.1 \
    --kolmogorov_wavenumber 2 \
    --viscosity "0.009133995846860976" \
    --downsample $downsample

# python3 generate-navier-stokes.py \
#     --output_dir $datadir \
#     --drag 0.1 \
#     --simulation_time $simulation_time \
#     --save_dt $save_dt \
#     --num_trajectories $num_trajectories \
#     --batch_size $batchsize \
#     --forcing_func kolmogorov \
#     --resolution $resolution \
#     --burn_in 0.1 \
#     --kolmogorov_wavenumber 2 \
#     --viscosity "0.00030296104428212476" \
#     --downsample $downsample

# python3 generate-navier-stokes.py \
#     --output_dir $datadir \
#     --drag 0.1 \
#     --simulation_time $simulation_time \
#     --save_dt $save_dt \
#     --num_trajectories $num_trajectories \
#     --batch_size $batchsize \
#     --forcing_func kolmogorov \
#     --resolution $resolution \
#     --burn_in 0.1 \
#     --kolmogorov_wavenumber 2 \
#     --viscosity "0.0019158219548093154" \
#     --downsample $downsample


# python3 generate-navier-stokes.py \
#     --output_dir $datadir \
#     --drag 0.1 \
#     --simulation_time 5.0 \
#     --save_dt $save_dt \
#     --num_trajectories $num_trajectories \
#     --batch_size $batchsize \
#     --forcing_func kolmogorov \
#     --resolution $resolution \
#     --burn_in 41 \
#     --kolmogorov_wavenumber 2 \
#     --viscosity "0.00030296104428212476" \
#     --downsample $downsample


# for i in "${!viscositiesgpuone[@]}";
# do
#     CUDA_VISIBLE_DEVICES=$gpuone python generate-navier-stokes.py \
#     --output_dir $datadir \
#     --drag 0.1 \
#     --simulation_time 15.0 \
#     --save_dt $save_dt \
#     --num_trajectories $num_trajectories \
#     --batch_size $batchsize \
#     --forcing_func kolmogorov \
#     --resolution $resolution \
#     --burn_in 41 \
#     --kolmogorov_wavenumber 2 \
#     --viscosity "${viscositiesgpuone[$i]}" \
#     --downsample $downsample &

#     CUDA_VISIBLE_DEVICES=$gputwo python generate-navier-stokes.py \
#     --output_dir $datadir \
#     --drag 0.1 \
#     --simulation_time 15.0 \
#     --save_dt $save_dt \
#     --forcing_func kolmogorov \
#     --num_trajectories $num_trajectories \
#     --batch_size $batchsize \
#     --resolution $resolution \
#     --burn_in 41 \
#     --kolmogorov_wavenumber 2 \
#     --viscosity "${viscositiesgputwo[$i]}" \
#     --downsample $downsample &

#     CUDA_VISIBLE_DEVICES=$gputhree python generate-navier-stokes.py \
#     --output_dir $datadir \
#     --drag 0.1 \
#     --simulation_time 15.0 \
#     --save_dt $save_dt \
#     --forcing_func kolmogorov \
#     --resolution $resolution \
#     --num_trajectories $num_trajectories \
#     --batch_size $batchsize \
#     --burn_in 41 \
#     --kolmogorov_wavenumber 2 \
#     --viscosity "${viscositiesgputhree[$i]}" \
#     --downsample $downsample &

#     CUDA_VISIBLE_DEVICES=$gpufour python generate-navier-stokes.py \
#     --output_dir $datadir \
#     --drag 0.1 \
#     --simulation_time 15.0 \
#     --save_dt $
#     --forcing_func kolmogorov \
#     --num_trajectories $
#     --batch_size $batchsize \
#     --resolution $resolution \
#     --burn_in 41 \
#     --kolmogorov_wavenumber 2 \
#     --viscosity "${viscositiesgpufour[$i]}" \
#     --downsample $downsample &


# done
