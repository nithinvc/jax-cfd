import click
import jax
import jax.numpy as jnp
import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial

import utils as fluids_utils


# Spectral projection function for downsampling - single frame
@partial(jax.jit, static_argnums=(1,))
def spectral_project_single(u_ref, N):
    """
    Project a single high-resolution field onto an N×N grid by zero-padding in Fourier space
    and inverse FFT back to real space.
    """
    N_ref = u_ref.shape[0]
    
    # If already at target resolution or smaller, return as is
    if N_ref <= N:
        return u_ref
    
    # forward FFT (real→complex), shift zero-freq to centre
    U_hat = jnp.fft.fftshift(jnp.fft.fft2(u_ref))
    
    # index range that survives on the coarse grid
    k = N // 2                   # positive wavenumbers to keep
    ctr = N_ref // 2            # centre index in reference spectrum
    keep = slice(ctr - k, ctr + k)   # low-k band

    # zero-pad everything else
    U_coarse = jnp.zeros_like(U_hat)
    U_coarse = U_coarse.at[keep, keep].set(U_hat[keep, keep])

    # back to physical space and take the real part
    u_proj = jnp.fft.ifft2(jnp.fft.ifftshift(U_coarse)).real
    # finally down-sample to N×N by striding
    stride = N_ref // N
    return u_proj[::stride, ::stride]


def spectral_project_batch(field, N):
    """
    Memory-efficient spectral projection for batch of fields.
    Processes one sample at a time to avoid OOM errors.
    
    field: shape (batch, time, X, Y)
    N: target resolution
    """
    batch_size, time_steps, height, width = field.shape
    
    # If already at target resolution or smaller, return as is
    if height <= N and width <= N:
        return field
    
    # Process one sample at a time to save memory
    downsampled_list = []
    
    # Show progress for large datasets
    total_frames = batch_size * time_steps
    with tqdm(total=total_frames, desc=f"  Downsampling {height}x{width} -> {N}x{N}") as pbar:
        for b in range(batch_size):
            time_list = []
            for t in range(time_steps):
                # Process single frame
                frame = field[b, t, :, :]
                downsampled_frame = spectral_project_single(frame, N)
                time_list.append(downsampled_frame)
                pbar.update(1)
                
            # Stack time dimension
            downsampled_batch = jnp.stack(time_list, axis=0)
            downsampled_list.append(downsampled_batch)
    
    # Stack batch dimension
    return jnp.stack(downsampled_list, axis=0)


def get_cache_path(out_dir, viscosity, res, effective_res, downsample):
    """
    Generate a cache file path based on parameters.
    
    Parameters:
    -----------
    out_dir : str
        Output directory
    viscosity : float
        Viscosity value
    res : int
        Original resolution
    effective_res : int
        Effective resolution after downsampling
    downsample : bool
        Whether downsampling was applied
    
    Returns:
    --------
    cache_path : str
        Path to cache file
    """
    cache_dir = os.path.join(out_dir, '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache key from parameters
    cache_key = f"visc_{viscosity}_res_{res}_eff_{effective_res}_downsample_{downsample}"
    cache_filename = f"spectrum_{cache_key}.npz"
    return os.path.join(cache_dir, cache_filename)


def load_cached_spectrum(cache_path):
    """
    Load cached spectrum if it exists.
    
    Parameters:
    -----------
    cache_path : str
        Path to cache file
    
    Returns:
    --------
    spectrum : np.ndarray or None
        Cached spectrum if exists, None otherwise
    """
    if os.path.exists(cache_path):
        try:
            cached_data = np.load(cache_path)
            # The spectrum is stored with key 'spectrum'
            if 'spectrum' in cached_data:
                return jnp.array(cached_data['spectrum'])
        except Exception as e:
            print(f"  Warning: Failed to load cache from {cache_path}: {e}")
    return None


def save_cached_spectrum(cache_path, spectrum):
    """
    Save spectrum to cache.
    
    Parameters:
    -----------
    cache_path : str
        Path to cache file
    spectrum : jnp.ndarray
        Spectrum to cache
    """
    try:
        np.savez_compressed(cache_path, spectrum=np.array(spectrum))
    except Exception as e:
        print(f"  Warning: Failed to save cache to {cache_path}: {e}")


@click.command()
@click.option("--loc", type=str, default="data/fluids_data", help="Location of the fluids data")
@click.option("--out_dir", type=str, default="data/fluids_stats", help="Location of the output directory")
@click.option("--dev", is_flag=True, help="Dev mode: process only first viscosity, includes all resolutions up to 2048")
@click.option("--viscosity", type=float, default=None, help="Specific viscosity to process. If not provided, processes all viscosities.")
@click.option("--downsample", is_flag=True, help="Downsample all fields to 256x256 resolution (fields smaller than 256 are left untouched)")
@click.option("--data_only", is_flag=True, help="Save energy spectrums data only (no plot)")
def main(loc, out_dir, dev, viscosity, downsample, data_only):
    print("Jax devices", jax.devices())


    # make the out_dir if it doesnt exist
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

        if metadata['viscosity'] != float(viscosity):
            continue
        
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


    fields = data[viscosity]
    spectrums = {}

    for res, field in fields.items():
        print(f"Processing resolution {res}, field shape: {field.shape}")
        
        # Apply downsampling if requested
        effective_res = res
        if downsample and res > 256:
            print(f"  Downsampling from {res}x{res} to 256x256...")
            field = spectral_project_batch(field, 256)
            effective_res = 256
            print(f"  Downsampled field shape: {field.shape}")
        elif downsample and res <= 256:
            print(f"  Field already at or below 256x256, keeping original resolution")
        
        # Check cache first
        cache_path = get_cache_path(out_dir, viscosity, res, effective_res, downsample)
        cached_spectrum = load_cached_spectrum(cache_path)
        
        if cached_spectrum is not None:
            print(f"  Using cached spectrum for resolution {res}")
            spectrums[res] = cached_spectrum
        else:
            print(f"  Computing spectrum for resolution {res} (not in cache)")
            # Option 3: Sequential processing (most memory-safe)
            spectra_list = []
            for i in tqdm(range(field.shape[0]), desc=f"Computing spectra for res {effective_res}"):
                if effective_res > 1024:
                    single_spectrum = fluids_utils.energy_spectrum_from_vorticity_lax_map(field[i], effective_res)
                else:
                    single_spectrum = fluids_utils.energy_spectrum_from_vorticity(field[i], effective_res)
                spectra_list.append(single_spectrum)
            
            # Stack all spectra and compute mean
            curr_spectrum = jnp.stack(spectra_list)
            # Mean over batch and time
            curr_spectrum = curr_spectrum.reshape(-1, curr_spectrum.shape[-1]).mean(axis=0)

            # Cache the result
            save_cached_spectrum(cache_path, curr_spectrum)
            print(f"  Cached spectrum for resolution {res}")

            # each spectrum is of shape K_max
            spectrums[res] = curr_spectrum

    # Construct output filename
    output_filename = f'energy_spectra_viscosity_{viscosity}'
    if downsample:
        output_filename += '_downsampled_256'
    
    if data_only:
        # Save the spectrums dictionary as a numpy file
        # Convert JAX arrays to numpy arrays for saving
        spectrums_np = {res: np.array(spectrum) for res, spectrum in spectrums.items()}
        
        data_filename = output_filename + '_data.npz'
        data_path = os.path.join(out_dir, data_filename)
        
        # Save as compressed numpy archive
        np.savez_compressed(data_path, **{str(res): spectrum for res, spectrum in spectrums_np.items()})
        
        print(f'Energy spectrum data saved to: {data_path}')
        print(f'Keys in saved data: {list(spectrums.keys())}')
        
    else:
        # Plot all spectrums on the same figure
        plt.figure(figsize=(10, 6))
        
        # Sort resolutions for consistent plotting order
        sorted_resolutions = sorted(spectrums.keys())
        
        # Plot energy spectra for each resolution
        for res in sorted_resolutions:
            offset = 2
            spectrum = spectrums[res][1:] # start from second wavenumber?
            k_values = np.arange(offset, len(spectrum) + offset)  # k starts from 1
            plt.loglog(k_values, spectrum, label=f'Resolution {res}', linewidth=2)
        
        # Add -5 power law reference line
        # Use the highest resolution spectrum as reference for scaling
        highest_res = max(sorted_resolutions)
        ref_spectrum = spectrums[highest_res]
        
        # Find a good scaling factor by matching at k=10 (or another suitable k in the inertial range)
        k_match = min(10, len(ref_spectrum) // 3)  # Use k=10 or 1/3 of max k
        scaling_factor = ref_spectrum[k_match - 1] * (k_match ** (5))
        
        # Create -5 power law line
        k_theory = np.logspace(0, np.log10(len(ref_spectrum)), 100)
        power_law = scaling_factor * k_theory ** (-5)
        
        # Plot the -5 line
        plt.loglog(k_theory, power_law, 'k--', alpha=0.7, linewidth=2, label=r'$k^{-5}$ (Kolmogorov)')
        
        plt.xlabel('Wavenumber k', fontsize=12)
        plt.ylabel('Energy Spectrum E(k)', fontsize=12)
        
        # Update title to indicate downsampling if applied
        viscosity_str = f'{viscosity:.0e}'
        title = f'Energy Spectra (Viscosity = {viscosity_str})'
        if downsample:
            title += ' - Downsampled to 256x256'
        plt.title(title, fontsize=14)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        xlim = plt.xlim()
        print(f'xlim: {xlim}')
        plt.xlim(1, xlim[1])
        plt.tight_layout()
        
        # Save the plot to the output directory
        plot_filename = output_filename + '.png'
        plot_path = os.path.join(out_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Energy spectrum plot saved to: {plot_path}')


if __name__ == "__main__":
    main()