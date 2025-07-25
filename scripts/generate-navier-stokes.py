import numpy as np
import argparse
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
from loguru import logger
from tqdm import tqdm

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types
from jax_cfd.spectral import utils as spectral_utils
import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.debug
from jax.debug import callback


def generate_random_vorticity_field(key, grid, max_velocity, peak_wavenumber):
    v0 = cfd.initial_conditions.filtered_velocity_field(
        key, grid, max_velocity, peak_wavenumber
    )
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat = jnp.fft.rfftn(vorticity0)
    print(
        f"Generated vorticity_hat shape: {vorticity_hat.shape}, has NaNs: {jnp.any(jnp.isnan(vorticity_hat))}")
    return vorticity_hat


def generate_output_folder_name(args):
    folder_name = f"ns_{args.resolution}x{args.resolution}_visc_{args.viscosity}_drag_{args.drag}_T{args.simulation_time}_forcing_{args.forcing_func}"
    return os.path.join(args.output_dir, folder_name)


def main():
    parser = argparse.ArgumentParser(description="Generate Navier Stokes")
    parser.add_argument("--output_dir", type=str,
                        default="/data/nithinc/pdes/NavierStokes-jax-cfd")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_trajectories", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viscosity", type=float, default=1e-3)
    parser.add_argument("--drag", type=float, default=0.0)
    parser.add_argument("--max_velocity", type=float, default=7)
    parser.add_argument("--max_courant_number", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--burn_in", type=float, default=40.0)
    parser.add_argument("--simulation_time", type=float, default=50.0)
    parser.add_argument("--save_dt", type=float, default=1.0)
    parser.add_argument("--forcing_func", type=str,
                        choices=["kolmogorov", "transient", "none"], default="kolmogorov")
    parser.add_argument("--kolmogorov_scale", type=float, default=1)
    parser.add_argument("--kolmogorov_wavenumber", type=int, default=4)
    parser.add_argument("--transient_scale", type=float, default=0.1)
    parser.add_argument("--downsample", type=int, default=-1)
    args = parser.parse_args()
    downsample = args.downsample

    output_dir = generate_output_folder_name(args)
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    args_file = os.path.join(output_dir, "args.json")
    logger.info(f"Dumping arguments to {args_file}")
    with open(args_file, "w") as f:
        json.dump(vars(args), f)

    resolution = args.resolution
    max_velocity = args.max_velocity
    viscosity = args.viscosity
    logger.info(
        f"Resolution: {resolution}x{resolution}, Max Velocity: {max_velocity}, Viscosity: {viscosity}")
    grid = grids.Grid((resolution, resolution), domain=(
        (0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    stable_dt = cfd.equations.stable_time_step(
        max_velocity, 0.5, viscosity, grid)
    logger.info(f"Stable time step: {stable_dt}")

    forcing_func = None
    forcing_func_type = args.forcing_func
    logger.info(f"Forcing function: {forcing_func_type}")
    if forcing_func_type == "kolmogorov":
        def forcing_func(grid): return cfd.forcings.kolmogorov_forcing(
            grid, scale=args.kolmogorov_scale, k=args.kolmogorov_wavenumber)
        logger.info(
            f"Forcing function args: scale={args.kolmogorov_scale}, k={args.kolmogorov_wavenumber}")
    elif forcing_func_type == "transient":
        def forcing_func(grid): return cfd.forcings.transient_flow_forcing(
            grid, scale=args.transient_scale)
        logger.info(f"Forcing function args: scale={args.transient_scale}")

    smooth = True
    drag = args.drag
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.NavierStokes2D(
            viscosity, grid, smooth=smooth, forcing_fn=forcing_func, drag=drag
        ),
        stable_dt,
    )
    logger.info("Time stepping scheme: Crank-Nicolson RK4")
    logger.info(f"Anti-aliasing: {smooth}")
    logger.info(f"Drag: {drag}")

    save_dt = args.save_dt  # stable_dt # temporary change
    burn_in_time = args.burn_in
    simulation_time = args.simulation_time
    total_time = burn_in_time + simulation_time
    outer_steps = total_time // save_dt
    inner_steps = (total_time // stable_dt) // outer_steps
    num_burn_in_frames = int(burn_in_time // save_dt)
    logger.info(
        f"Simulation Time: {simulation_time}, Burn-in Time: {burn_in_time}")
    logger.info(
        f"Number of Burn-in Frames: {num_burn_in_frames}, Real burn in time: {num_burn_in_frames * save_dt}")
    logger.info(f"Total Simulation Time: {total_time}, Save Time: {save_dt}")
    logger.info(f"Inner Steps: {inner_steps}, Outer Steps: {outer_steps}")

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps
    )

    sample_ic = partial(
        generate_random_vorticity_field,
        grid=grid,
        max_velocity=max_velocity,
        peak_wavenumber=4,
    )

    @jax.vmap
    def downsample_fn(trajectory):
        vel_sp = spectral.utils.vorticity_to_velocity(grid)(trajectory)
        vel_real = [jnp.fft.irfftn(v, axes=(0, 1)) for v in vel_sp]
        dst_grid = grids.Grid(
            (resolution // downsample, resolution // downsample),
            domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
        )
        small_traj = cfd.resize.downsample_staggered_velocity(
            grid, dst_grid, vel_real)
        kx, ky = dst_grid.rfft_mesh()
        small_traj = [jnp.fft.rfftn(v, axes=(0, 1)) for v in small_traj]
        small_traj = spectral.utils.spectral_curl_2d((kx, ky), small_traj)
        return small_traj

    def generate_solution_template(key, sample_ic, trajectory_fn):
        vorticity_hat0 = sample_ic(key)
        trajectory, spectral_trajectory = trajectory_fn(vorticity_hat0)

        spectral_trajectory = spectral_trajectory[num_burn_in_frames:]
        if downsample > 0:
            spectral_trajectory = downsample_fn(spectral_trajectory)
        return jnp.fft.irfftn(spectral_trajectory, axes=(1, 2))

    def generate_solution_template_with_save(key, sample_ic, trajectory_fn):
        vorticity_hat0 = sample_ic(key)
        trajectory, spectral_trajectory = trajectory_fn(vorticity_hat0)

        spectral_trajectory = spectral_trajectory[:]
        vorticity_hat = spectral_trajectory
        vxhat, vyhat = spectral.utils.vorticity_to_velocity(
            grid)(vorticity_hat)
        vx, vy = jnp.fft.irfftn(vxhat, axes=(
            1, 2)), jnp.fft.irfftn(vyhat, axes=(1, 2))
        kx, ky = grid.rfft_mesh()
        grad_x_hat = 2j * jnp.pi * kx * vorticity_hat
        grad_y_hat = 2j * jnp.pi * ky * vorticity_hat
        grad_x, grad_y = jnp.fft.irfftn(grad_x_hat, axes=(
            1, 2)), jnp.fft.irfftn(grad_y_hat, axes=(1, 2))
        laplace = (jnp.pi * 2j)**2 * (kx**2 + ky**2)
        advection = -(grad_x * vx + grad_y * vy)
        diffusion_term = jnp.fft.irfftn(
            viscosity * laplace * vorticity_hat, axes=(1, 2))
        diffusion_term_without_viscosity = jnp.fft.irfftn(
            laplace * vorticity_hat, axes=(1, 2))
        damping_term = jnp.fft.irfftn(drag * vorticity_hat, axes=(1, 2))

        # fields_dict = {
        #     "vorticity": vorticity,
        #     "vx": vx,
        #     "vy": vy,
        #     "advection": advection,
        #     "grad_x": grad_x,
        #     "grad_y": grad_y
        # }

        fields_dict = {
            "vorticity": jnp.fft.irfftn(vorticity_hat, axes=(1, 2)),
            "vx": vx,
            "vy": vy,
            "advection": advection,
            "w_dx": grad_x,
            "w_dy": grad_y,
            "diffusion_term": diffusion_term,
            "damping_term": damping_term,
            "diffusion_without_viscosity": diffusion_term_without_viscosity,
            "initial_vorticity_hat": vorticity_hat0,
            "initial_vorticity": jnp.fft.irfftn(vorticity_hat0, axes=(0, 1)),
            "initial_diffusion_term": jnp.fft.irfftn(viscosity * laplace * vorticity_hat0, axes=(0, 1)),
            "initial_damping_term": jnp.fft.irfftn(drag * vorticity_hat0, axes=(0, 1)),
        }

        def _save(fields_dict):
            jax.debug.print("Saving fields")
            advection = fields_dict["advection"]
            shape = advection.shape[-1]
            if forcing_func_type == "kolmogorov":
                output_dir = f"/data/divyam123/meta-cfd/jaxcfd_fields/2dtf_{viscosity}_{shape}_end_save"
            elif forcing_func_type == "none":
                output_dir = f"/data/divyam123/meta-cfd/jaxcfd_fields/2dns_{viscosity}_{shape}_end_save"
            os.makedirs(output_dir, exist_ok=True)
            total_files = len(os.listdir(output_dir))
            total_files = total_files + 1
            save_path = os.path.join(output_dir, f"index_{total_files}")
            os.makedirs(save_path, exist_ok=True)
            for name_str, field_array in fields_dict.items():
                field_np = np.array(field_array)
                np.save(os.path.join(
                    save_path, f"{name_str}.npy"), field_np)
            # print(f"Saved {name_str}_{step_num} to {save_path}")

        # Use a static counter that gets passed as an argument
        jax.debug.callback(_save, fields_dict)

        if downsample > 0:
            spectral_trajectory = downsample_fn(spectral_trajectory)
        return jnp.fft.irfftn(spectral_trajectory, axes=(1, 2))

    generate_solution = jax.jit(
        jax.vmap(
            partial(
                generate_solution_template_with_save,
                # generate_solution_template,
                sample_ic=sample_ic,
                trajectory_fn=trajectory_fn,
            ),
            in_axes=0,
            out_axes=0,
        )
    )

    seed = args.seed
    batch_size = args.batch_size
    num_batches = args.num_trajectories // batch_size
    rng_key = jax.random.PRNGKey(seed)
    logger.info(
        f"Batch Size: {batch_size}, Num Batches: {num_batches}, Seed: {seed}, Total Trajectories: {num_batches * batch_size}")
    logger.info("Generating trajectories...")
    for batch_number in tqdm(range(num_batches)):
        batch_key = jax.random.fold_in(rng_key, batch_number)
        batch_rngs = jax.random.split(batch_key, batch_size)
        # if os.path.exists(os.path.join(output_dir, f"batch_{batch_number}.npy")):
        #     continue
        batch_trajectories = generate_solution(batch_rngs)

        jnp.save(os.path.join(
            output_dir, f"batch_{batch_number}.npy"), batch_trajectories)
    logger.info("Done generating trajectories!")


if __name__ == "__main__":
    main()
