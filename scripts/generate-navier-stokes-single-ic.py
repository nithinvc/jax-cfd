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
    parser.add_argument("--input_file", type=str, default=None)
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
    parser.add_argument("--simulation_time", type=float, default=1.0)
    parser.add_argument(
        "--save_dt",
        type=float,
        default=0.1,
        help="Time between every saved frame (approx.). This may not be exactly followed due to the required stable_dt.",
    )

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
    parser.add_argument('--single_trajectory', action='store_true', help="do a single trajectory at a time")

    args = parser.parse_args()
    input_file = args.input_file
    initial_condition = jnp.load(input_file)
    assert input_file is not None, "Input file is required"
    downsample = args.downsample
    single_trajectory = args.single_trajectory

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
    grid = grids.Grid(
        (resolution, resolution), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi))
    )
    
    # Initial condition dims = B, 1, X, Y
    original_resolution = initial_condition.shape[2]
    print('original_resolution', original_resolution)
    assert original_resolution % resolution == 0, "Original resolution must be divisible by resolution"
    downsample_factor = original_resolution // resolution
    logger.info(f"Downsample factor: {downsample_factor}")
    logger.info(f"Original resolution: {original_resolution}")
    logger.info(f"Initial condition shape: {initial_condition.shape}")

    original_grid = grids.Grid(
        (original_resolution, original_resolution), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi))
    )

    stable_dt = cfd.equations.stable_time_step(
        max_velocity, args.max_courant_number, viscosity, grid
    )
    logger.info(f"Stable time step: {stable_dt}")

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
    simulation_time = args.simulation_time
    total_time = simulation_time
    outer_steps = total_time // save_dt
    inner_steps = (total_time // stable_dt) // outer_steps
    num_burn_in_frames = 0
    logger.info(f"Simulation Time: {simulation_time}")
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

    TO_JIT = False
    if TO_JIT:
        jit = jax.jit
    else:
        jit = lambda x: x

    ic_downsample_fn = jit(
        partial(
            downsample_fn_template,
            downsample_factor=downsample_factor,
            original_grid=original_grid,
            current_resolution=original_resolution,
        )
    )

    def generate_solution_template(ic, trajectory_fn):
        ic = ic[0]
        assert ic.ndim == 2, "Initial condition must be 2D"
        ic = jnp.fft.rfftn(ic)
        vorticity_hat0 = ic_downsample_fn(ic)

        print('vorticity_hat0.shape', vorticity_hat0.shape)
        _, spectral_trajectory = trajectory_fn(vorticity_hat0)
        # Not a necessary step. We could store in a spectral representation but,
        # they consume the same amount of space so we preprocess.

        return jnp.fft.irfftn(spectral_trajectory, axes=(1, 2))

    generate_solution = jit(
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
    rng_key = jax.random.PRNGKey(seed)
    num_batches = 1
    logger.info(
        f"Batch Size: {batch_size}, Num Batches: {num_batches}, Seed: {seed}, Total Trajectories: {num_batches * batch_size}"
    )
    logger.info("Generating trajectories...")

    if single_trajectory:
        initial_condition = [initial_condition[i] for i in range(initial_condition.shape[0])]
        initial_condition = [jnp.expand_dims(i, axis=0) for i in initial_condition]
        num_batches = len(initial_condition)

    for batch_number in tqdm(range(num_batches)):
        batch_key = jax.random.fold_in(rng_key, batch_number)
        batch_rngs = jax.random.split(batch_key, batch_size)
        # Quick check to see if the file already exists
        # If it does, we ignore it
        # This is so we can resubmit jobs to NERSC without having to manually remove the generated parameters
        if single_trajectory:
            batch_trajectories = generate_solution(initial_condition[batch_number])
        else:
            batch_trajectories = generate_solution(initial_condition)
        # Remove burn in frames - Batch x Time x X x Y
        # batch_trajectories = batch_trajectories[:, :]
        jnp.save(
            os.path.join(output_dir, f"batch_{batch_number}.npy"), batch_trajectories
        )
    logger.info("Done generating trajectories!")


if __name__ == "__main__":
    main()
