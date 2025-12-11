#!/bin/bash

JAX_ENABLE_X64=True LD_LIBRARY_PATH="" python fluids_stats.py --loc data/rollouts/rollouts/ --out_dir fluids_stats_fp64 --viscosity 0.01
JAX_ENABLE_X64=True LD_LIBRARY_PATH="" python fluids_stats.py --loc data/rollouts/rollouts/ --out_dir fluids_stats_fp64 --viscosity 0.001
JAX_ENABLE_X64=True LD_LIBRARY_PATH="" python fluids_stats.py --loc data/rollouts/rollouts/ --out_dir fluids_stats_fp64 --viscosity 0.0001
JAX_ENABLE_X64=True LD_LIBRARY_PATH="" python fluids_stats.py --loc data/rollouts/rollouts/ --out_dir fluids_stats_fp64 --viscosity 1e-5




JAX_ENABLE_X64=True LD_LIBRARY_PATH="" python fluids_stats.py --loc data/rollouts/rollouts/ --out_dir fluids_stats_fp64 --viscosity 0.005
JAX_ENABLE_X64=True LD_LIBRARY_PATH="" python fluids_stats.py --loc data/rollouts/rollouts/ --out_dir fluids_stats_fp64 --viscosity 0.0005
JAX_ENABLE_X64=True LD_LIBRARY_PATH="" python fluids_stats.py --loc data/rollouts/rollouts/ --out_dir fluids_stats_fp64 --viscosity 5e-5