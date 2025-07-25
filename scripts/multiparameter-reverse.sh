#!/bin/bash
gpuone=0
gputwo=1
gputhree=2
gpufour=3

datadir="/global/cfs/cdirs/m4319/nithinc/meta-pde/turbulent-flow-very-high-res"
batchsize=4
resolution=2048
downsample=-1
numtraj=16
viscositiesgpuone=("0.0025764174425233167" "0.006153085601625313" "4.857295179217165e-05" "0.005280796376895365" "1.2424747083660186e-05" "0.0003699972431463808" "0.0019158219548093154" "0.0045881565491609705" "8.569331925053982e-05" "0.0038842777547031426" "0.0020597335357437196" "1.0388823104027935e-05" "1.6736010167825783e-05")
viscositiesgputwo=("3.6283583803549155e-05" "0.002656813924114493" "1.7019223026554023e-05" "5.59598687800608e-05" "0.0008113929572637835" "0.00019170041589170651" "0.00048287152161792117" "0.00026100256506134784" "9.452571391072311e-05" "0.0007411299781083245" "1.667761543019792e-05" "0.0027950159165083337" "0.009133995846860976")
viscositiesgputhree=("0.004760767751809498" "0.000794714742465374" "7.40038575908737e-05" "0.0001702741688676439" "8.771380343280557e-05" "1.1919481947918725e-05" "0.002055424552015075" "2.284455685002053e-05" "0.0015446089075047066" "9.833181933644887e-05" "0.0001189589673755355" "0.001319994226153501" "0.0020736445177905022")
viscositiesgpufour=("0.0004149795789891589" "0.004115113049561088" "3.0455368715396772e-05" "0.0018477934173519257" "0.0003355151022721483" "2.1070472806578224e-05" "0.00030296104428212476" "0.0013795402040204172" "0.0008178476574339542" "1.551225912648474e-05" "2.2264204303769678e-05" "0.0015382308040279" "3.9459088110999965e-05")


for i in "${!viscositiesgpuone[@]}";
do
    CUDA_VISIBLE_DEVICES=$gpuone python generate-navier-stokes.py \
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
    --viscosity "${viscositiesgpuone[$i]}" \
    --downsample $downsample &

    CUDA_VISIBLE_DEVICES=$gputwo python generate-navier-stokes.py \
    --output_dir $datadir \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --forcing_func kolmogorov \
    --num_trajectories $numtraj \
    --batch_size $batchsize \
    --resolution $resolution \
    --burn_in 41 \
    --kolmogorov_wavenumber 2 \
    --viscosity "${viscositiesgputwo[$i]}" \
    --downsample $downsample &

    CUDA_VISIBLE_DEVICES=$gputhree python generate-navier-stokes.py \
    --output_dir $datadir \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --forcing_func kolmogorov \
    --resolution $resolution \
    --num_trajectories $numtraj \
    --batch_size $batchsize \
    --burn_in 41 \
    --kolmogorov_wavenumber 2 \
    --viscosity "${viscositiesgputhree[$i]}" \
    --downsample $downsample &

    CUDA_VISIBLE_DEVICES=$gpufour python generate-navier-stokes.py \
    --output_dir $datadir \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --forcing_func kolmogorov \
    --num_trajectories $numtraj \
    --batch_size $batchsize \
    --resolution $resolution \
    --burn_in 41 \
    --kolmogorov_wavenumber 2 \
    --viscosity "${viscositiesgpufour[$i]}" \
    --downsample $downsample &

    # CUDA_VISIBLE_DEVICES=$gpufive python generate-navier-stokes.py \
    # --output_dir /data/nithinc/pdes/NavierStokes-jax-cfd/multiparameter \
    # --drag 0.1 \
    # --simulation_time 15.0 \
    # --save_dt 0.25 \
    # --forcing_func kolmogorov \
    # --resolution 512 \
    # --burn_in 41 \
    # --num_trajectories 112 \ 
    # --batch_size 16 \
    # --kolmogorov_wavenumber 2 \
    # --viscosity "{$viscositiesgpufive[$i]}" \
    # --downsample 8 &

   wait
done
