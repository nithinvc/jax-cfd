import matplotlib.pyplot as plt
import numpy as np

import os


import click
import json
from collections import defaultdict

import scienceplots

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import jax
import jax.numpy as jnp
from functools import partial
plt.style.use(['science'])


@partial(jax.vmap, in_axes=(0, None, None))
def downsample_fn_template(
    trajectory,  original_grid, dst_grid
):
    trajectory = jnp.fft.rfftn(trajectory, axes=(0, 1))
    vel_sp = spectral.utils.vorticity_to_velocity(original_grid)(trajectory)
    vel_real = [jnp.fft.irfftn(v, axes=(0, 1)) for v in vel_sp]
    small_traj = cfd.resize.downsample_staggered_velocity(
        original_grid, dst_grid, vel_real
    )
    kx, ky = dst_grid.rfft_mesh()
    small_traj = [jnp.fft.rfftn(v, axes=(0, 1)) for v in small_traj]
    small_traj = spectral.utils.spectral_curl_2d((kx, ky), small_traj)
    return jnp.fft.irfftn(small_traj, axes=(0, 1))

def downsample(field, new_resolution):
    # T, X, Y

    original_grid = grids.Grid(
        (
            field.shape[1],
            field.shape[2],
        ),
        domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
    )

    dst_grid = grids.Grid(
        (
            new_resolution,
            new_resolution,
        ),
        domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)),
    )

    downsampled_field = downsample_fn_template(field, original_grid, dst_grid)


    return downsampled_field
    

def plot_solution_field(field, viscosity, resolution, time_indices=[0, -1], downsampled= False):
    """Plot solution field snapshots and save to plots/{viscosity}/{resolution}.jpg"""
    
    # Create directory structure
    plot_dir = f"plots/{viscosity}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create subplots for different time snapshots
    fig, axes = plt.subplots(1, len(time_indices), figsize=(12, 5))
    if len(time_indices) == 1:
        axes = [axes]
    
    for i, t_idx in enumerate(time_indices):
        im = axes[i].imshow(field[t_idx], cmap='RdBu_r', origin='lower')
        axes[i].set_title(f't = {t_idx}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle(f'Solution Field ($\\nu = {viscosity}$, res = {resolution})')
    plt.tight_layout()
    
    # Save the plot
    filename = os.path.join(plot_dir, f"{resolution}.jpg")
    if downsampled:
        new_resolution = field.shape[-1]
        filename = os.path.join(plot_dir, f"{resolution}_downsampled_{new_resolution}.jpg")
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f'Saved solution field plot: {filename}')
    

@click.command("main")
@click.option("--loc", type=click.Path(exists=True), required=True, help="location of all data")
def main(loc):
    print("Jax devices", jax.devices())
    generated_solutions_dirs = os.listdir(loc)
    generated_solutions_metdata = []
    generated_solutions_data = []
    missing_data = []
    for d in generated_solutions_dirs:
        metadata_file = os.path.join(loc, d, 'args.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        generated_solutions_metdata.append(metadata)
        current_batches = []
        for files in os.listdir(os.path.join(loc, d)):
            if files.endswith('.npy'):
                current_batches.append(np.load(os.path.join(loc, d, files)))

        if len(current_batches) == 0:
            missing_data.append(d)
            generated_solutions_metdata.pop()
        else:
            assert len(current_batches) == 1, 'more than one batch not supported'
            generated_solutions_data.append(current_batches[0]) # TODO: this should be cat in the case we support multiple batches
    
    print(f'Folders with missing data: {missing_data}')

    # map viscosity -> dict: res -> array
    data: dict[float, dict[int, np.ndarray]] = defaultdict(lambda: dict()) # type: ignore

    for metadata, field in zip(generated_solutions_metdata, generated_solutions_data):
        data[metadata['viscosity']][metadata['resolution']] = field

    batch_index = 0
    for viscosity, res_dict in data.items():
        highest_res = max(res_dict.keys())
        print(f'Processing viscosity: {viscosity}, highest resolution: {highest_res}, shape: {res_dict[highest_res].shape}')
        errors = []
        high_res_field = res_dict[highest_res][batch_index]
        resolutions = list(res_dict.keys())
        resolutions.sort()

        res_order = []

        for res in resolutions:
            field = res_dict[res][batch_index]
            print(f'Processing resolution: {res}, shape: {field.shape}')
            
            # Plot solution field
            plot_solution_field(field, viscosity, res)

            downsampled_field = downsample(high_res_field, res)
            plot_solution_field(downsampled_field, viscosity, highest_res, downsampled=True)
            
            error = np.linalg.norm(field - downsampled_field)
            errors.append(error)
            res_order.append(res)

        errors = np.asarray(errors)
        plt.figure(figsize=(10, 6))
        # plt.loglog(resolutions, errors, 'o-', linewidth=2, markersize=8)
        plt.plot(resolutions, errors, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Resolution', fontsize=12)
        plt.ylabel('Error', fontsize=12)
        plt.title(f'Resolution vs Error (Viscosity = {viscosity})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure with viscosity in filename
        fig_filename = f'error_vs_resolution_viscosity_{viscosity}.png'
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved figure: {fig_filename}')



if __name__ == "__main__":
    main()