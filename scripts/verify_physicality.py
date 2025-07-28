import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import os


import click
import json
from collections import defaultdict

import scienceplots
from tqdm import tqdm
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import jax
import jax.numpy as jnp
from functools import partial
plt.style.use(['science'])

# batched and across time
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

@partial(jax.vmap, in_axes=(0, None))
def downsample(field, new_resolution):
    # T, X, Y
    print(field.shape)

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

def downsample_scipy(field, new_resolution: int):
    """
    Downsample field using scipy.ndimage.zoom
    Field: Batch size x T x X x Y
    new_resolution: int - target resolution for X and Y dimensions
    """
    # Get current shape
    if field.ndim == 3:  # T x X x Y
        current_resolution_x = field.shape[1]
        current_resolution_y = field.shape[2]
        
        # Calculate zoom factors
        zoom_x = new_resolution / current_resolution_x
        zoom_y = new_resolution / current_resolution_y
        
        # Apply zoom (1.0 for time dimension, zoom factors for spatial dimensions)
        zoom_factors = (1.0, zoom_x, zoom_y)
        
    elif field.ndim == 4:  # Batch x T x X x Y
        current_resolution_x = field.shape[2]
        current_resolution_y = field.shape[3]
        
        # Calculate zoom factors
        zoom_x = new_resolution / current_resolution_x
        zoom_y = new_resolution / current_resolution_y
        
        # Apply zoom (1.0 for batch and time dimensions, zoom factors for spatial dimensions)
        zoom_factors = (1.0, 1.0, zoom_x, zoom_y)
        
    else:
        raise ValueError(f"Expected 3D (T,X,Y) or 4D (Batch,T,X,Y) field, got {field.ndim}D")
    
    # Use scipy zoom with order=1 (bilinear interpolation) for smooth downsampling
    downsampled_field = ndimage.zoom(field, zoom_factors, order=1)
    
    return downsampled_field
    

def plot_solution_field(field, viscosity, resolution, out_dir, time_indices=[0, -1], downsampled= False):
    """Plot solution field snapshots and save to {out_dir}/{viscosity}/{resolution}.jpg"""
    
    # Create directory structure
    plot_dir = os.path.join(out_dir, str(viscosity))
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create subplots for different time snapshots
    fig, axes = plt.subplots(1, len(time_indices), figsize=(12, 5))
    if len(time_indices) == 1:
        axes = [axes]
    
    for i, t_idx in enumerate(time_indices):
        im = axes[i].imshow(field[t_idx], cmap='RdBu_r', origin='lower', vmin=-8, vmax=8)
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
@click.option("--out_dir", type=click.Path(), help="output directory for plots")
@click.option("--plot_sols", is_flag=True, help="plot solutions")
@click.option("--list_table", is_flag=True, help="print table of errors and skip plotting")
@click.option("--dev", is_flag=True, help="use dev mode")
def main(loc, out_dir, plot_sols, list_table, dev):
    print("Jax devices", jax.devices())
    
    # Create output directory if it doesn't exist (only if plotting)
    if not list_table:
        os.makedirs(out_dir, exist_ok=True)
    
    generated_solutions_dirs = os.listdir(loc)
    generated_solutions_metdata = []
    generated_solutions_data = []
    missing_data = []
    for d in tqdm(generated_solutions_dirs, desc='Processing directories'):
        if dev and ('2048' in d or '1024' in d):
            continue
        if not os.path.isdir(os.path.join(loc, d)):
            continue
        metadata_file = os.path.join(loc, d, 'args.json')

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        generated_solutions_metdata.append(metadata)
        current_batches = []
        files = os.listdir(os.path.join(loc, d))
        files.sort()
        for files in files:
            if files.endswith('.npy'):
                current_batches.append(np.load(os.path.join(loc, d, files)))

        if len(current_batches) == 0:
            missing_data.append(d)
            generated_solutions_metdata.pop()
        else:
            generated_solutions_data.append(np.concatenate(current_batches, axis=0))

    print(f'Folders with missing data: {missing_data}')

    # map viscosity -> dict: res -> array
    data: dict[float, dict[int, np.ndarray]] = defaultdict(lambda: dict()) # type: ignore

    for metadata, field in zip(generated_solutions_metdata, generated_solutions_data):
        data[metadata['viscosity']][metadata['resolution']] = field

    # Store results for table
    table_results = {}

    for viscosity, res_dict in data.items():
        highest_res = max(res_dict.keys())
        print(f'Processing viscosity: {viscosity}, highest resolution: {highest_res}, shape: {res_dict[highest_res].shape}')
        errors = []
        high_res_field = res_dict[highest_res]
        resolutions = list(res_dict.keys())
        resolutions.sort()

        res_order = []

        for res in resolutions:
            field = res_dict[res]
            print(f'Processing resolution: {res}, shape: {field.shape}')
            
            # Plot solution field (only if not list_table mode)
            if plot_sols and not list_table:
                plot_solution_field(field[0], viscosity, res, out_dir)

            downsampled_field = downsample_scipy(high_res_field, res)
            if plot_sols and not list_table:
                plot_solution_field(downsampled_field[0], viscosity, highest_res, out_dir, downsampled=True)
            
            error = (np.linalg.norm(field - downsampled_field) / np.linalg.norm(downsampled_field)) * 100
            errors.append(error)
            res_order.append(res)

        # Store results for table
        table_results[viscosity] = dict(zip(res_order, errors))

        errors = np.asarray(errors)
        
        # Only create plots if not in list_table mode
        if not list_table:
            plt.figure(figsize=(10, 6))
            # plt.loglog(resolutions, errors, 'o-', linewidth=2, markersize=8)
            plt.plot(resolutions, errors, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Resolution', fontsize=12)
            plt.ylabel('Error (\%)', fontsize=12)
            plt.title(f'Resolution vs Error (Viscosity = {viscosity})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save figure with viscosity in filename
            fig_filename = os.path.join(out_dir, f'error_vs_resolution_viscosity_{viscosity}.png')
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Saved figure: {fig_filename}')

    # Print table if requested
    if list_table:
        print("\n" + "="*80)
        print("ERROR COMPARISON TABLE")
        max_resolution = max([max(res_dict.keys()) for res_dict in data.values()])
        print("(Errors compared to " + str(max_resolution) + " resolution downsampled)")
        print("="*80)
        
        # Get all unique resolutions for header
        all_resolutions = set()
        for res_dict in table_results.values():
            all_resolutions.update(res_dict.keys())
        all_resolutions = sorted(list(all_resolutions))
        
        # Print header
        header = f"{'Viscosity':<12}"
        for res in all_resolutions:
            header += f"{res:>12}"
        print(header)
        print("-" * len(header))
        
        # Print data rows
        for viscosity in sorted(table_results.keys()):
            row = f"{viscosity:<12.6f}"
            for res in all_resolutions:
                if res in table_results[viscosity]:
                    error = table_results[viscosity][res]
                    row += f"{error:>12.4f}"
                else:
                    row += f"{'--':>12}"
            print(row)
        
        print("="*80)
        print("Error values are percentages (%)")
        print("="*80)


if __name__ == "__main__":
    main()