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


def generate_random_vorticity_field(key, grid, max_velocity, peak_wavenumber):
    v0 = cfd.initial_conditions.filtered_velocity_field(
        key, grid, max_velocity, peak_wavenumber
    )
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    return jnp.fft.rfftn(vorticity0)


def generate_output_folder_name(args):
    folder_name = f"ns_{args.resolution}x{args.resolution}_visc_{args.viscosity}_drag_{args.drag}_T{args.simulation_time}_forcing_{args.forcing_func}"
    return os.path.join(args.output_dir, folder_name)


def main():
    parser = argparse.ArgumentParser(description="Generate Navier Stokes")

    ## IO & Compute Parameters
    parser.add_argument(
        "--output_dir", type=str, default="/data/nithinc/pdes/NavierStokes-jax-cfd"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_trajectories", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)

    ## General NS parameters
    parser.add_argument("--viscosity", type=float, default=1e-3)
    parser.add_argument("--drag", type=float, default=0.0)
    parser.add_argument("--max_velocity", type=float, default=7)

    ## Solver Parmeters
    parser.add_argument("--max_courant_number", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--burn_in", type=float, default=40.0)
    parser.add_argument("--simulation_time", type=float, default=50.0)
    parser.add_argument(
        "--save_dt",
        type=float,
        default=1.0,
        help="Time between every saved frame (approx.). This may not be exactly followed due to the required stable_dt.",
    )
    parser.add_argument("--max_resolution", type=int, default=4096)

    ## Forcing Function Args
    parser.add_argument(
        "--forcing_func",
        type=str,
        choices=["kolmogorov", "transient", "none"],
        default="kolmogorov",
    )
    # KF args
    parser.add_argument("--kolmogorov_scale", type=float, default=1)
    parser.add_argument("--kolmogorov_wavenumber", type=int, default=4)
    # TF args
    parser.add_argument("--transient_scale", type=float, default=0.1)
    parser.add_argument("--downsample", type=int, default=-1)

    args = parser.parse_args()
    downsample = args.downsample

    # Ensure output directory exists
    output_dir = generate_output_folder_name(args)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Dump args to a json file
    args_file = os.path.join(output_dir, "args.json")
    logger.info(f"Dumping arguments to {args_file}")
    with open(args_file, "w") as f:
        json.dump(vars(args), f)

    # Setup grid and get stable time step
    resolution: int = args.resolution
    max_velocity: float = args.max_velocity
    viscosity: float = args.viscosity
    logger.info(
        f"Resolution: {resolution}x{resolution}, Max Velocity: {max_velocity}, Viscosity: {viscosity}"
    )
    logger.info(f"Max Resolution: {args.max_resolution}")
    max_resolution: int = args.max_resolution
    grid = grids.Grid(
        (resolution, resolution), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi))
    )

    max_resolution_grid = grids.Grid(
        (max_resolution, max_resolution), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi))
    )
    max_resolution_stable_dt = cfd.equations.stable_time_step(
        max_velocity, args.max_courant_number, viscosity, max_resolution_grid
    )
    logger.info(f"Max Resolution Stable Time Step: {max_resolution_stable_dt}")

    stable_dt = cfd.equations.stable_time_step(
        max_velocity, args.max_courant_number, viscosity, grid
    )
    logger.info(f"Stable time step: {stable_dt}")

    if max_resolution_stable_dt < stable_dt:
        logger.warning(
            f"Max resolution stable time step is less than the stable time step. This will cause the simulation to be unstable. Max resolution stable time step: {max_resolution_stable_dt}, Stable time step: {stable_dt}"
        )
        stable_dt = max_resolution_stable_dt

    # Setup forcing function
    forcing_func = None
    forcing_func_type = args.forcing_func
    logger.info(f"Forcing function: {forcing_func_type}")
    if forcing_func_type == "kolmogorov":
        forcing_func = lambda grid: cfd.forcings.kolmogorov_forcing(  # noqa: E731
            grid,
            scale=args.kolmogorov_scale,
            k=args.kolmogorov_wavenumber,
        )
        logger.info(
            f"Forcing function args: scale={args.kolmogorov_scale}, k={args.kolmogorov_wavenumber}"
        )
    elif forcing_func_type == "transient":
        forcing_func = lambda grid: cfd.forcings.transient_flow_forcing(  # noqa: E731
            grid, scale=args.transient_scale
        )
        logger.info(f"Forcing function args: scale={args.transient_scale}")

    # Setup time stepping scheme
    smooth = True  # Use anti-aliasing
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

    # Compute inner, outer, and total simulation times
    save_dt = args.save_dt
    burn_in_time = args.burn_in
    simulation_time = args.simulation_time
    total_time = burn_in_time + simulation_time
    outer_steps = total_time // save_dt
    inner_steps = (total_time // stable_dt) // outer_steps
    num_burn_in_frames = int(burn_in_time // save_dt)
    logger.info(f"Simulation Time: {simulation_time}, Burn-in Time: {burn_in_time}")
    logger.info(
        f"Number of Burn-in Frames: {num_burn_in_frames}, Real burn in time: {num_burn_in_frames * save_dt}"
    )
    logger.info(f"Total Simulation Time: {total_time}, Save Time: {save_dt}")
    logger.info(f"Inner Steps: {inner_steps}, Outer Steps: {outer_steps}")

    # Setup trajectory function
    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps
    )

    # setup ic sampling
    sample_ic = partial(
        generate_random_vorticity_field,
        grid=grid,
        max_velocity=max_velocity,
        peak_wavenumber=4,
    )

    # Function mapping rngkey -> trajectory
    def downsample_fn_template(
        trajectory, downsample_factor, original_grid, current_resolution
    ):
        vel_sp = spectral.utils.vorticity_to_velocity(original_grid)(trajectory)
        vel_real = [jnp.fft.irfftn(v, axes=(0, 1)) for v in vel_sp]
        dst_grid = grids.Grid(
            (
                current_resolution // downsample_factor,
                current_resolution // downsample_factor,
            ),
            domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
        )
        small_traj = cfd.resize.downsample_staggered_velocity(
            original_grid, dst_grid, vel_real
        )
        kx, ky = dst_grid.rfft_mesh()
        small_traj = [jnp.fft.rfftn(v, axes=(0, 1)) for v in small_traj]
        small_traj = spectral.utils.spectral_curl_2d((kx, ky), small_traj)
        return small_traj

    initial_condition_downsample_factor = max_resolution / resolution
    assert initial_condition_downsample_factor % 1 == 0, (
        "Initial condition downsample factor must be an integer"
    )
    logger.info(
        f"Initial condition downsample factor: {initial_condition_downsample_factor}"
    )
    initial_condition_downsample_factor = int(initial_condition_downsample_factor)

    downsample_fn = jax.jit(
        jax.vmap(
            partial(
                downsample_fn_template,
                downsample_factor=downsample,
                original_grid=grid,
                current_resolution=resolution,
            )
        )
    )
    ic_downsample_fn = jax.jit(
        partial(
            downsample_fn_template,
            downsample_factor=initial_condition_downsample_factor,
            original_grid=max_resolution_grid,
            current_resolution=max_resolution,
        )
    )

    @jax.jit
    def get_initial_condition_from_high_res(key):
        sample_traj_pre_sampling = generate_random_vorticity_field(
            key, max_resolution_grid, max_velocity, 4
        )
        # Add time dim
        sample_traj = ic_downsample_fn(sample_traj_pre_sampling)
        return sample_traj_pre_sampling, sample_traj

    def generate_solution_template(key, trajectory_fn):
        _, vorticity_hat0 = get_initial_condition_from_high_res(key)
        _, spectral_trajectory = trajectory_fn(vorticity_hat0)
        # Not a necessary step. We could store in a spectral representation but,
        # they consume the same amount of space so we preprocess.

        # Remove burn in frames so we don't have to repeat downsampling
        spectral_trajectory = spectral_trajectory[num_burn_in_frames:]
        if downsample > 0:
            spectral_trajectory = downsample_fn(spectral_trajectory)

        return jnp.fft.irfftn(spectral_trajectory, axes=(1, 2))

    generate_solution = jax.jit(
        jax.vmap(
            partial(
                generate_solution_template,
                trajectory_fn=trajectory_fn,
            ),
            in_axes=0,
            out_axes=0,
        )
    )

    # Iterate through number of batches
    seed = args.seed
    batch_size = args.batch_size
    num_batches = args.num_trajectories // batch_size
    rng_key = jax.random.PRNGKey(seed)
    logger.info(
        f"Batch Size: {batch_size}, Num Batches: {num_batches}, Seed: {seed}, Total Trajectories: {num_batches * batch_size}"
    )
    logger.info("Generating trajectories...")
    for batch_number in tqdm(range(num_batches)):
        batch_key = jax.random.fold_in(rng_key, batch_number)
        batch_rngs = jax.random.split(batch_key, batch_size)
        # Quick check to see if the file already exists
        # If it does, we ignore it
        # This is so we can resubmit jobs to NERSC without having to manually remove the generated parameters
        if os.path.exists(os.path.join(output_dir, f"batch_{batch_number}.npy")):
            continue
        batch_trajectories = generate_solution(batch_rngs)
        # Remove burn in frames - Batch x Time x X x Y
        # batch_trajectories = batch_trajectories[:, :]
        jnp.save(
            os.path.join(output_dir, f"batch_{batch_number}.npy"), batch_trajectories
        )
    logger.info("Done generating trajectories!")


if __name__ == "__main__":
    main()
