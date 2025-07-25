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
viscositiesgpuone=("0.0001329291894316216" "2.9380279387035334e-05" "0.0006358358856676254" "0.00314288089084011" "8.17949947521167e-05" "0.000684792009557478" "0.00023345864076016249" "0.0005987474910461401" "1.5673095467235405e-05" "8.200518402245828e-05" "2.32335035153901e-05" "5.9750279999602906e-05" "0.00043664735929796326" "0.006584106160121612" "1.8427970406864546e-05" "0.00014656553886225324" "6.963114377829287e-05" "1.6736010167825783e-05" "1.0388823104027935e-05" "0.0020597335357437196" "0.0038842777547031426" "8.569331925053982e-05" "0.0045881565491609705" "0.0019158219548093154" "0.0003699972431463808" "1.2424747083660186e-05" "0.005280796376895365" "4.857295179217165e-05" "0.006153085601625313" "0.0025764174425233167")
viscositiesgputwo=("0.0071144760093434225" "2.9375384576328295e-05" "0.001331121608073689" "4.335281794951564e-05" "0.00037520558551242813" "2.621087878265438e-05" "0.0022673986523780395" "1.3783237455007187e-05" "0.007025166339242158" "1.9634341572933304e-05" "0.0003058656666978527" "0.0009717775305059633" "3.585612610345396e-05" "0.004835952776465951" "3.872118032174584e-05" "6.516990611177177e-05" "0.00042470585622618684" "0.009133995846860976" "0.0027950159165083337" "1.667761543019792e-05" "0.0007411299781083245" "9.452571391072311e-05" "0.00026100256506134784" "0.00048287152161792117" "0.00019170041589170651" "0.0008113929572637835" "5.59598687800608e-05" "1.7019223026554023e-05" "0.002656813924114493" "3.6283583803549155e-05")
viscositiesgputhree=("0.001570297088405539" "1.4936568554617619e-05" "1.1527987128232394e-05" "3.511356313970405e-05" "0.00019762189340280086" "7.52374288453485e-05" "3.972110727381908e-05" "0.0006647135865318024" "0.00788671412999049" "0.0011290133559092666" "1.2681352169084594e-05" "8.612579192594876e-05" "0.008105016126411584" "0.0006218704727769079" "1.3667272915456215e-05" "0.003063462210622081" "2.6471141828218167e-05" "0.0020736445177905022" "0.001319994226153501" "0.0001189589673755355" "9.833181933644887e-05" "0.0015446089075047066" "2.284455685002053e-05" "0.002055424552015075" "1.1919481947918725e-05" "8.771380343280557e-05" "0.0001702741688676439" "7.40038575908737e-05" "0.000794714742465374" "0.004760767751809498")
viscositiesgpufour=("0.0006251373574521745" "0.003967605077052989" "0.008123245085588688" "3.5498788321965036e-05" "7.476312062252303e-05" "0.00012562773503807024" "0.0003489018845491386" "3.2476735706274465e-05" "0.002661901888489057" "0.00020914981329035593" "0.00534516611064682" "0.00036324869566766035" "0.002115429079726122" "0.00582938454299474" "9.46217535646148e-05" "0.00011756010900231849" "0.002550298070162891" "3.9459088110999965e-05" "0.0015382308040279" "2.2264204303769678e-05" "1.551225912648474e-05" "0.0008178476574339542" "0.0013795402040204172" "0.00030296104428212476" "2.1070472806578224e-05" "0.0003355151022721483" "0.0018477934173519257" "3.0455368715396772e-05" "0.004115113049561088" "0.0004149795789891589")


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
