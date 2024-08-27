#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python generate-navier-stokes.py \
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
    --viscosity 1e-3 \
    --downsample 8
