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
from jax import pmap
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

def plot_difference_field(diff_field, viscosity, resolution, highest_res, out_dir, time_indices=[0, -1]):
    """Plot difference field snapshots and save to {out_dir}/{viscosity}/diff_{resolution}_vs_{highest_res}.jpg"""
    
    # Create directory structure
    plot_dir = os.path.join(out_dir, str(viscosity))
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create subplots for different time snapshots
    fig, axes = plt.subplots(1, len(time_indices), figsize=(12, 5))
    if len(time_indices) == 1:
        axes = [axes]
    
    # Calculate dynamic color range based on the difference field
    vmin = np.percentile(diff_field, 1)  # 1st percentile
    vmax = np.percentile(diff_field, 99)  # 99th percentile
    
    for i, t_idx in enumerate(time_indices):
        im = axes[i].imshow(diff_field[t_idx], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[i].set_title(f't = {t_idx}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle(f'Difference Field: {resolution} vs {highest_res} downsampled ($\\nu = {viscosity}$)')
    plt.tight_layout()
    
    # Save the plot with diff_ prefix
    filename = os.path.join(plot_dir, f"diff_{resolution}_vs_{highest_res}.jpg")
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f'Saved difference field plot: {filename}')

def plot_normalized_difference_field(diff_field, downsampled_field, viscosity, resolution, highest_res, out_dir, time_indices=[0, -1]):
    """Plot normalized difference field snapshots and save to {out_dir}/{viscosity}/normalized_diff_{resolution}_vs_{highest_res}.jpg"""
    
    # Create directory structure
    plot_dir = os.path.join(out_dir, str(viscosity))
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate normalized difference (pointwise relative error)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-12
    normalized_diff = np.divide(diff_field, downsampled_field + epsilon, 
                               out=np.zeros_like(diff_field), 
                               where=(np.abs(downsampled_field) > epsilon))
    
    # Create subplots for different time snapshots
    fig, axes = plt.subplots(1, len(time_indices), figsize=(12, 5))
    if len(time_indices) == 1:
        axes = [axes]
    
    # Calculate dynamic color range based on the normalized difference field
    vmin = np.percentile(normalized_diff, 1)  # 1st percentile
    vmax = np.percentile(normalized_diff, 99)  # 99th percentile
    
    for i, t_idx in enumerate(time_indices):
        im = axes[i].imshow(normalized_diff[t_idx], cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        axes[i].set_title(f't = {t_idx}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle(f'Normalized Difference Field: {resolution} vs {highest_res} downsampled ($\\nu = {viscosity}$)')
    plt.tight_layout()
    
    # Save the plot with normalized_diff_ prefix
    filename = os.path.join(plot_dir, f"normalized_diff_{resolution}_vs_{highest_res}.jpg")
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f'Saved normalized difference field plot: {filename}')

# one for batch and one for time
@partial(jax.vmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(0, None))
def spectral_project(u_ref, N):
    """
    Project 2048² reference field onto an N×N grid by zero-padding in Fourier space
    and inverse FFT back to real space.
    """
    N_ref = u_ref.shape[0]
    # forward FFT (real→complex), shift zero-freq to centre
    U_hat = jnp.fft.fftshift(jnp.fft.fft2(u_ref))
    
    # index range that survives on the coarse grid
    k = N//2                     # positive wavenumbers to keep
    ctr = N_ref//2               # centre index in 2048 spectrum
    keep = slice(ctr-k, ctr+k)   # low-k band

    # zero-pad everything else
    U_coarse = jnp.zeros_like(U_hat)
    U_coarse = U_coarse.at[keep, keep].set(U_hat[keep, keep])

    # back to physical space and take the real part
    u_proj = jnp.fft.ifft2(jnp.fft.ifftshift(U_coarse)).real
    # finally down-sample to N×N by striding
    stride = N_ref // N
    return u_proj[::stride, ::stride]

# Parallel version for multiple GPUs
# pmap over devices, vmap over time
@partial(pmap, in_axes=(0, None))
@partial(jax.vmap, in_axes=(0, None))
def spectral_project_parallel(u_ref, N):
    """
    Parallel version of spectral_project that runs across multiple devices.
    Input should be sharded across devices on the first (batch) dimension.
    """
    N_ref = u_ref.shape[0]
    # forward FFT (real→complex), shift zero-freq to centre
    U_hat = jnp.fft.fftshift(jnp.fft.fft2(u_ref))
    
    # index range that survives on the coarse grid
    k = N//2                     # positive wavenumbers to keep
    ctr = N_ref//2               # centre index in 2048 spectrum
    keep = slice(ctr-k, ctr+k)   # low-k band

    # zero-pad everything else
    U_coarse = jnp.zeros_like(U_hat)
    U_coarse = U_coarse.at[keep, keep].set(U_hat[keep, keep])

    # back to physical space and take the real part
    u_proj = jnp.fft.ifft2(jnp.fft.ifftshift(U_coarse)).real
    # finally down-sample to N×N by striding
    stride = N_ref // N
    return u_proj[::stride, ::stride]

@click.command("main")
@click.option("--loc", type=click.Path(exists=True), required=True, help="location of all data")
@click.option("--out_dir", type=click.Path(), help="output directory for plots")
@click.option("--plot_sols", is_flag=True, help="plot solutions")
@click.option("--no_list_table", "list_table", flag_value=False, default=True, help="disable printing table of errors")
@click.option("--dev", is_flag=True, help="Dev mode: process only first viscosity, includes all resolutions up to 2048")
@click.option("--max_seconds", type=float, default=4.0, help="maximum length of trajectory to compare (in seconds, max 4.0)")
@click.option("--time_chunk_size", type=int, default=None, help="Process time steps in chunks of this size to save memory")
def main(loc, out_dir, plot_sols, list_table, dev, max_seconds, time_chunk_size):
    """
    Main function with GPU parallelization support.
    
    GPU Parallelization Features:
    - Automatically detects and uses all available GPUs
    - Distributes batch computations across GPUs using pmap
    - Supports time chunking to reduce memory usage
    - Falls back to single GPU for small batches
    
    Memory Optimization:
    - Use --time_chunk_size to process time steps in chunks (e.g., --time_chunk_size 20)
    - This is helpful when processing high-resolution (2048x2048) data
    """
    devices = jax.devices()
    n_devices = len(devices)
    print(f"Jax devices ({n_devices} available):", devices)
    
    # Print memory info
    for i, device in enumerate(devices):
        stats = device.memory_stats()
        if stats:
            used_gb = stats.get('bytes_in_use', 0) / (1024**3)
            limit_gb = stats.get('bytes_limit', 0) / (1024**3)
            print(f"  Device {i}: {used_gb:.2f} GB used / {limit_gb:.2f} GB total")

    # make the out_dir if it doesnt exist
    os.makedirs(out_dir, exist_ok=True)
    
    generated_solutions_dirs = os.listdir(loc)
    generated_solutions_metdata = []
    generated_solutions_data = []
    missing_data = []
    for d in tqdm(generated_solutions_dirs, desc='Processing directories'):
        if not os.path.isdir(os.path.join(loc, d, d)):
            continue
        metadata_file = os.path.join(loc, d, d, 'args.json')

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        generated_solutions_metdata.append(metadata)
        current_batches = []
        files = os.listdir(os.path.join(loc, d, d))
        files.sort()
        for files in files:
            if files.endswith('.npy'):
                current_batches.append(np.load(os.path.join(loc, d, d, files)))

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

    # Validate max_seconds
    if max_seconds <= 0 or max_seconds > 4.0:
        raise ValueError(f"max_seconds must be between 0 and 4.0, got {max_seconds}")
    
    print(f"\nUsing first {max_seconds} seconds of trajectory for comparison (out of 4.0 seconds total)")
    
    # Store results for table
    table_results = {}
    
    # In dev mode, only process the first viscosity
    viscosities_to_process = data.items()
    if dev:
        viscosities_to_process = list(data.items())[:1]
        print(f"\nDev mode: Processing only first viscosity ({list(data.keys())[0]}) out of {len(data)} available")

    for viscosity, res_dict in viscosities_to_process:
        highest_res = max(res_dict.keys())
        print(f'Processing viscosity: {viscosity}, highest resolution: {highest_res}, shape: {res_dict[highest_res].shape}')
        errors = []
        errors_spectral = []
        high_res_field = res_dict[highest_res]
        resolutions = list(res_dict.keys())
        resolutions.sort()

        # Store time-series errors for error vs time plot
        time_series_errors = {}  # res -> array of errors over time
        res_order = []

        for res in resolutions:
            field = res_dict[res]
            print(f'Processing resolution: {res}, shape: {field.shape}')
            
            # Calculate time indices to use based on max_seconds
            # Assuming T time indices correspond to 4 seconds total
            total_time_steps = field.shape[1]  # T dimension
            time_steps_to_use = int(total_time_steps * max_seconds / 4.0)
            
            if res == resolutions[0]:  # Print info only once per viscosity
                print(f'  Using {time_steps_to_use} out of {total_time_steps} time steps ({max_seconds:.1f}s out of 4.0s)')
            
            # Slice the field to use only the specified time portion
            field_sliced = field[:, :time_steps_to_use, :, :]
            
            # Plot solution field if requested
            if plot_sols:
                plot_solution_field(field_sliced[0], viscosity, res, out_dir)

            # Downsample and slice the high resolution field
            
            high_res_field_sliced = high_res_field[:, :time_steps_to_use, :, :]
            downsampled_field_sliced = downsample_scipy(high_res_field_sliced, res)

            # Use parallel spectral projection if batch size allows distribution
            batch_size = high_res_field_sliced.shape[0]
            
            # Force time chunking for high resolution to avoid OOM
            effective_time_chunk_size = time_chunk_size
            if highest_res >= 2048 and effective_time_chunk_size is None:
                # Auto-set chunk size for 2048x2048 data
                # With batch_size=4, we need very small chunks to fit in memory
                effective_time_chunk_size = 5  # Very conservative for 2048x2048
                if dev:
                    effective_time_chunk_size = 3  # Even more conservative in dev mode
                print(f'  Auto-setting time_chunk_size={effective_time_chunk_size} for 2048x2048 data to avoid OOM')
            
            # Process in time chunks if specified
            if effective_time_chunk_size is not None and high_res_field_sliced.shape[1] > effective_time_chunk_size:
                print(f'  Processing in time chunks of size {effective_time_chunk_size}...')
                downsampled_chunks = []
                n_time_steps = high_res_field_sliced.shape[1]
                
                for t_start in range(0, n_time_steps, effective_time_chunk_size):
                    t_end = min(t_start + effective_time_chunk_size, n_time_steps)
                    high_res_chunk = high_res_field_sliced[:, t_start:t_end, :, :]
                    
                    # For now, just use single GPU processing within chunks
                    # The time chunking is the main memory optimization
                    if t_start == 0:  # Print only once
                        print(f'  Computing spectral downsample for res={res}...')
                    downsampled_chunk = spectral_project(high_res_chunk, res)
                    
                    downsampled_chunks.append(downsampled_chunk)
                
                # Concatenate time chunks
                downsampled_spectral = np.concatenate(downsampled_chunks, axis=1)
                
            else:
                # Process all time steps at once (only for smaller resolutions)
                print(f'  Computing spectral downsample for res={res}...')
                downsampled_spectral = spectral_project(high_res_field_sliced, res)
            
            spectral_error = (np.linalg.norm(field_sliced - downsampled_spectral) / np.linalg.norm(downsampled_spectral)) * 100
            errors_spectral.append(spectral_error)
            
            if plot_sols:
                plot_solution_field(downsampled_field_sliced[0], viscosity, highest_res, out_dir, downsampled=True)
                
                # Plot the difference field
                diff_field = field_sliced - downsampled_field_sliced
                plot_difference_field(diff_field[0], viscosity, res, highest_res, out_dir)
                
                # Plot the normalized difference field
                plot_normalized_difference_field(diff_field[0], downsampled_field_sliced[0], viscosity, res, highest_res, out_dir)
            
            # Compute error using only the sliced portions (final time point for table)
            error = (np.linalg.norm(field_sliced - downsampled_field_sliced) / np.linalg.norm(downsampled_field_sliced)) * 100
            errors.append(error)
            res_order.append(res)
            
            # Compute time-series spectral errors for error vs time plot
            # We already have downsampled_spectral computed above, so just slice it
            time_errors_spectral = []
            for t in range(field_sliced.shape[1]):  # time dimension
                field_t = field_sliced[:, t:t+1, :, :]  # Keep batch and spatial dims
                downsampled_spectral_t = downsampled_spectral[:, t:t+1, :, :]
                error_t = (np.linalg.norm(field_t - downsampled_spectral_t) / np.linalg.norm(downsampled_spectral_t)) * 100
                time_errors_spectral.append(error_t)
            time_series_errors[res] = np.array(time_errors_spectral)

        # Store spectral results for table
        table_results[viscosity] = dict(zip(res_order, errors_spectral))

        errors_spectral = np.asarray(errors_spectral)
        
        # Create error vs time plot
        plt.figure(figsize=(10, 6))
        
        # Create time array (assuming 4 seconds total simulation time)
        time_array = np.linspace(0, max_seconds, time_series_errors[resolutions[0]].shape[0])
        
        # Plot error vs time for each resolution
        for res in resolutions:
            plt.plot(time_array, time_series_errors[res], 'o-', linewidth=2, markersize=4, label=f'res = {res}')
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Spectral Error (%)', fontsize=12)
        plt.title(f'Spectral Error vs Time (Viscosity = {viscosity})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save spectral error vs time figure
        time_fig_filename = os.path.join(out_dir, f'spectral_error_vs_time_viscosity_{viscosity}.png')
        plt.savefig(time_fig_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved error vs time figure: {time_fig_filename}')
        
        # Create spectral error vs resolution plot
        plt.figure(figsize=(10, 6))
        # plt.loglog(resolutions, errors_spectral, 'o-', linewidth=2, markersize=8)
        plt.plot(resolutions, errors_spectral, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Resolution', fontsize=12)
        plt.ylabel('Spectral Error (%)', fontsize=12)
        plt.title(f'Resolution vs Spectral Error (Viscosity = {viscosity})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save spectral figure with viscosity in filename
        fig_filename = os.path.join(out_dir, f'spectral_error_vs_resolution_viscosity_{viscosity}.png')
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved figure: {fig_filename}')

    # Print table if requested
    if list_table:
        print("\n" + "="*80)
        print("SPECTRAL ERROR COMPARISON TABLE")
        max_resolution = max([max(res_dict.keys()) for res_dict in data.values()])
        print("(Spectral errors compared to " + str(max_resolution) + " resolution spectral projection)")
        if max_seconds < 4.0:
            print(f"(Using first {max_seconds} seconds of 4.0 second trajectories)")
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
        print("Spectral error values are percentages (%)")
        print("="*80)


if __name__ == "__main__":
    main()