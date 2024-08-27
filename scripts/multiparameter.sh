#!/bin/bash
gpuone=3
gputwo=4
gputhree=5
gpufour=6
gpufive=7

viscositiesgpuone=(9.999999999999999e-06 1.4174741629268048e-05 2.0092330025650458e-05 2.8480358684358048e-05 4.037017258596558e-05 5.72236765935022e-05 8.111308307896872e-05 0.00011497569953977356 0.00016297508346206434 0.00023101297000831605 0.00032745491628777284 0.0004641588833612782 0.0006579332246575682 0.0009326033468832199 0.0013219411484660286 0.0018738174228603848 0.0026560877829466868 0.0037649358067924675 0.005336699231206312 0.007564633275546291)
viscositiesgputwo=(1.072267222010323e-05 1.5199110829529332e-05 2.1544346900318823e-05 3.053855508833412e-05 4.328761281083062e-05 6.135907273413175e-05 8.697490026177834e-05 0.0001232846739442066 0.0001747528400007683 0.0002477076355991711 0.0003511191734215131 0.0004977023564332113 0.0007054802310718645 0.001 0.0014174741629268048 0.002009233002565048 0.002848035868435802 0.004037017258596553 0.00572236765935022 0.008111308307896872)
viscositiesgputhree=(1.1497569953977357e-05 1.6297508346206434e-05 2.310129700083158e-05 3.274549162877732e-05 4.641588833612782e-05 6.579332246575683e-05 9.326033468832199e-05 0.00013219411484660288 0.0001873817422860383 0.00026560877829466864 0.00037649358067924675 0.0005336699231206312 0.000756463327554629 0.0010722672220103231 0.0015199110829529332 0.0021544346900318843 0.0030538555088334154 0.004328761281083057 0.006135907273413175 0.008697490026177835)
viscositiesgpufour=(1.2328467394420658e-05 1.747528400007683e-05 2.4770763559917088e-05 3.511191734215127e-05 4.977023564332114e-05 7.054802310718646e-05 0.0001 0.00014174741629268049 0.0002009233002565046 0.0002848035868435802 0.0004037017258596558 0.0005722367659350221 0.0008111308307896872 0.0011497569953977356 0.001629750834620645 0.0023101297000831605 0.0032745491628777285 0.004641588833612782 0.006579332246575682 0.0093260334688322)
viscositiesgpufive=(1.3219411484660286e-05 1.873817422860383e-05 2.6560877829466838e-05 3.7649358067924715e-05 5.3366992312063123e-05 7.56463327554629e-05 0.00010722672220103231 0.0001519911082952933 0.00021544346900318845 0.0003053855508833416 0.00043287612810830614 0.0006135907273413176 0.0008697490026177834 0.0012328467394420659 0.0017475284000076847 0.0024770763559917113 0.003511191734215131 0.004977023564332114 0.007054802310718645 0.01)


for i in "${!viscositiesgpuone[@]}";
do
    CUDA_VISIBLE_DEVICES=$gpuone python generate-navier-stokes.py \
    --output_dir /data/nithinc/pdes/NavierStokes-jax-cfd/multiparameter \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --num_trajectories 112 \ 
    --batch_size 16 \
    --forcing_func kolmogorov \
    --resolution 512 \
    --burn_in 41 \
    --kolmogorov_wavenumber 2 \
    --viscosity "{$viscositiesgpuone[$i]}" \
    --downsample 8 &

    CUDA_VISIBLE_DEVICES=$gputwo python generate-navier-stokes.py \
    --output_dir /data/nithinc/pdes/NavierStokes-jax-cfd/multiparameter \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --forcing_func kolmogorov \
    --num_trajectories 112 \ 
    --batch_size 16 \
    --resolution 512 \
    --burn_in 41 \
    --kolmogorov_wavenumber 2 \
    --viscosity "{$viscositiesgputwo[$i]}" \
    --downsample 8 &

    CUDA_VISIBLE_DEVICES=$gputhree python generate-navier-stokes.py \
    --output_dir /data/nithinc/pdes/NavierStokes-jax-cfd/multiparameter \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --forcing_func kolmogorov \
    --resolution 512 \
    --num_trajectories 112 \ 
    --batch_size 16 \
    --burn_in 41 \
    --kolmogorov_wavenumber 2 \
    --viscosity "{$viscositiesgputhree[$i]}" \
    --downsample 8 &

    CUDA_VISIBLE_DEVICES=$gpufour python generate-navier-stokes.py \
    --output_dir /data/nithinc/pdes/NavierStokes-jax-cfd/multiparameter \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --forcing_func kolmogorov \
    --num_trajectories 112 \ 
    --batch_size 16 \
    --resolution 512 \
    --burn_in 41 \
    --kolmogorov_wavenumber 2 \
    --viscosity "{$viscositiesgpufour[$i]}" \
    --downsample 8 &

    CUDA_VISIBLE_DEVICES=$gpufive python generate-navier-stokes.py \
    --output_dir /data/nithinc/pdes/NavierStokes-jax-cfd/multiparameter \
    --drag 0.1 \
    --simulation_time 15.0 \
    --save_dt 0.25 \
    --forcing_func kolmogorov \
    --resolution 512 \
    --burn_in 41 \
    --num_trajectories 112 \ 
    --batch_size 16 \
    --kolmogorov_wavenumber 2 \
    --viscosity "{$viscositiesgpufive[$i]}" \
    --downsample 8 &

   wait
done
