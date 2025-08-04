import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

@jit
def vorticity_to_velocity(vorticity):
    """
    Convert vorticity to velocity components using streamfunction.
    
    Parameters:
    -----------
    vorticity : array, shape (X, Y)
        Vorticity field
    
    Returns:
    --------
    u_x, u_y : arrays, shape (X, Y)
        Velocity components
    """
    N = vorticity.shape[0]
    
    # Compute streamfunction from vorticity using Poisson equation
    # In Fourier space: psi_hat = -vorticity_hat / k^2
    vort_hat = jnp.fft.fft2(vorticity)
    
    # Create wavenumber arrays
    kx = jnp.fft.fftfreq(N, d=1.0) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(N, d=1.0) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    # Avoid division by zero at k=0
    K2 = K2.at[0, 0].set(1.0)
    psi_hat = -vort_hat / K2
    psi_hat = psi_hat.at[0, 0].set(0.0)  # Set mean streamfunction to zero
    
    # Compute velocity components from streamfunction
    # u_x = -∂psi/∂y, u_y = ∂psi/∂x
    u_x_hat = -1j * KY * psi_hat
    u_y_hat = 1j * KX * psi_hat
    
    u_x = jnp.real(jnp.fft.ifft2(u_x_hat))
    u_y = jnp.real(jnp.fft.ifft2(u_y_hat))
    
    return u_x, u_y

@partial(jit, static_argnames=['k_max'])
def energy_spectrum_single(u_x, u_y, k_max=None):
    """
    Return azimuthally averaged energy spectrum E(k) for a single time step.
    
    Parameters:
    -----------
    u_x, u_y : arrays, shape (X, Y)
        Velocity components
    k_max : int, optional
        Maximum wavenumber. If None, uses N//3 (2/3 dealiasing rule)
    
    Returns:
    --------
    E : array, shape (k_max+1,)
        Energy spectrum
    """
    N = u_x.shape[0]
    
    # FFT, shifted so k=0 is at centre
    Ux = jnp.fft.fftshift(jnp.fft.fft2(u_x))
    Ux = Ux / (N**2)
    Uy = jnp.fft.fftshift(jnp.fft.fft2(u_y))
    Uy = Uy / (N**2)
    
    # Integer wave numbers
    kx = jnp.fft.fftshift(jnp.fft.fftfreq(N)) * N
    ky = kx
    KX, KY = jnp.meshgrid(kx, ky)
    K = jnp.hypot(KX, KY).astype(jnp.int32)
    
    if k_max is None:  # Nyquist under 2/3 de-alias
        k_max = N // 3  # integer division
    
    # Vectorized computation of energy spectrum
    def compute_E_k(k):
        mask = (K == k)
        return 0.5 * jnp.sum(jnp.abs(Ux)**2 * mask + jnp.abs(Uy)**2 * mask)
    
    k_vals = jnp.arange(k_max + 1)
    E = vmap(compute_E_k)(k_vals)
    
    return E

@partial(jit, static_argnames=['k_max'])
def energy_spectrum_from_vorticity(vorticity, k_max=None):
    """
    Compute energy spectrum from vorticity field.
    
    Parameters:
    -----------
    vorticity : array, shape (T, X, Y)
        Vorticity field over time
    k_max : int, optional
        Maximum wavenumber. If None, uses N//3 (2/3 dealiasing rule)
    
    Returns:
    --------
    E : array, shape (T, k_max+1)
        Energy spectrum for each time step
    """
    N = vorticity.shape[1]
    
    if k_max is None:
        k_max = N // 3
    
    def process_timestep(vort_t):
        u_x, u_y = vorticity_to_velocity(vort_t)
        return energy_spectrum_single(u_x, u_y, k_max)
    
    # Vectorize over time dimension
    E = vmap(process_timestep)(vorticity)
    
    return E

@partial(jit, static_argnames=['k_max'])
def energy_spectrum_from_velocity(u_x, u_y, k_max=None):
    """
    Compute energy spectrum from velocity components.
    
    Parameters:
    -----------
    u_x, u_y : arrays, shape (T, X, Y)
        Velocity components over time
    k_max : int, optional
        Maximum wavenumber. If None, uses N//3 (2/3 dealiasing rule)
    
    Returns:
    --------
    E : array, shape (T, k_max+1)
        Energy spectrum for each time step
    """
    N = u_x.shape[1]
    
    if k_max is None:
        k_max = N // 3
    
    # Vectorize over time dimension
    E = vmap(lambda ux, uy: energy_spectrum_single(ux, uy, k_max))(u_x, u_y)
    
    return E

# Utility functions for common operations
@jit
def compute_enstrophy(vorticity):
    """
    Compute enstrophy (integral of vorticity squared) for each time step.
    
    Parameters:
    -----------
    vorticity : array, shape (T, X, Y) or (X, Y)
        Vorticity field
    
    Returns:
    --------
    enstrophy : array, shape (T,) or scalar
        Enstrophy for each time step
    """
    if vorticity.ndim == 3:
        return jnp.mean(vorticity**2, axis=(1, 2)) / 2
    else:
        return jnp.mean(vorticity**2) / 2

@jit
def compute_total_energy(u_x, u_y):
    """
    Compute total kinetic energy for each time step.
    
    Parameters:
    -----------
    u_x, u_y : arrays, shape (T, X, Y) or (X, Y)
        Velocity components
    
    Returns:
    --------
    energy : array, shape (T,) or scalar
        Total energy for each time step
    """
    if u_x.ndim == 3:
        return jnp.mean(u_x**2 + u_y**2, axis=(1, 2)) / 2
    else:
        return jnp.mean(u_x**2 + u_y**2) / 2