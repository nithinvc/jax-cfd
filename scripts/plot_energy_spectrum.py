import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_energy_spectrum_from_vorticity(
    omega,                    # ndarray (T, X, Y)  – vorticity snapshots
    save_path="spectrum.png",
    kolmogorov_color="k",
    nbins=None,
):
    """
    Compute the time-averaged kinetic-energy spectrum of a 2-D incompressible
    flow given only ζ(x,y,t) on a [0,2π]² grid, and plot it with a –5/3 line.

    Parameters
    ----------
    omega : np.ndarray
        Vorticity array of shape (T, X, Y).  Domain is [0,2π] in both x,y.
    save_path : str
        PNG path for the figure.
    kolmogorov_color : str
        Colour for the reference –5/3 slope.
    nbins : int or None
        Number of radial bins (default: X//2).
    """
    if omega.ndim != 3:
        raise ValueError("omega must have shape (T, X, Y)")

    T, nx, ny = omega.shape
    nbins = nbins or nx // 2

    # ---------- Fourier grid -------------------------------------------------
    kx = np.fft.fftfreq(nx, d=2 * np.pi / nx)   # kx = 0, ±1, ±2, ...
    ky = np.fft.fftfreq(ny, d=2 * np.pi / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = np.inf                         # avoid division by zero
    kr = np.sqrt(k2).ravel()                  # radial wavenumber per mode

    # Remove infinity values for proper binning
    kr_finite = kr[np.isfinite(kr)]
    k_edges = np.linspace(0.0, kr_finite.max(), nbins + 1)
    k_centers = 0.5 * (k_edges[1:] + k_edges[:-1])

    spectrum = np.zeros(nbins)

    # ---------- loop over time snapshots ------------------------------------
    for t in range(T):
        w_hat = np.fft.fft2(omega[t])

        # stream-function  ψ̂ = − ω̂ / k²  (ω = −∇²ψ)
        psi_hat = -w_hat / k2

        # velocity components: û =  i k_y ψ̂ ,  v̂ = −i k_x ψ̂
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat

        # spectral energy density per mode
        e_k = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / (nx * ny) ** 2
        
        # Only use finite kr values for histogram
        finite_mask = np.isfinite(kr)
        hist, _ = np.histogram(kr[finite_mask], bins=k_edges, weights=e_k.ravel()[finite_mask])
        spectrum += hist

    spectrum /= T                             # time-average

    # ---------- Kolmogorov −5/3 reference -----------------------------------
    inertial = spectrum > 0
    i0 = np.argmax(inertial)                  # first non-zero bin
    C_k = spectrum[i0] * k_centers[i0] ** (5 / 3)
    k_ref = k_centers[inertial]
    kolmogorov = C_k * k_ref ** (-5 / 3)

    # ---------- Regression analysis for -5/3 power law ----------------------
    # Filter data in wavenumber range [4, 15] for inertial range analysis
    inertial_range_mask = (k_centers >= 4) & (k_centers <= 15) & (spectrum > 0)
    
    if np.sum(inertial_range_mask) >= 3:  # Need at least 3 points for regression
        k_inertial = k_centers[inertial_range_mask]
        spectrum_inertial = spectrum[inertial_range_mask]
        
        # Linear regression in log-log space: log(E) = slope * log(k) + intercept
        log_k = np.log10(k_inertial)
        log_E = np.log10(spectrum_inertial)
        
        # Calculate regression coefficient
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_E)
        r_squared = r_value ** 2
        
        # Ideal slope should be -5/3 ≈ -1.667
        ideal_slope = -5/3
        slope_deviation = abs(slope - ideal_slope)
        
        print(f"Inertial range analysis (k=4-15) for {save_path}:")
        print(f"  Measured slope: {slope:.3f} (ideal: {ideal_slope:.3f})")
        print(f"  R² coefficient: {r_squared:.3f}")
        print(f"  Slope deviation from -5/3: {slope_deviation:.3f}")
    else:
        print(f"Insufficient data points in k=4-15 range for regression analysis ({save_path})")

    # ---------- plot ---------------------------------------------------------
    plt.figure()
    plt.loglog(k_centers, spectrum, "o-", label=r"$E(k)$")
    plt.loglog(k_ref, kolmogorov, "--", color=kolmogorov_color,
               label=r"$k^{-5/3}$")
    
    # Add finer x-axis tick marks and labels
    from matplotlib.ticker import LogLocator, LogFormatter
    plt.gca().xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs=(0.2, 0.4, 0.6, 0.8), numticks=15))
    plt.gca().xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    
    plt.xlabel(r"Wavenumber $k$")
    plt.ylabel(r"Energy spectrum $E(k)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved spectrum to {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot energy spectrum from simulation data')
    parser.add_argument('file_path', type=str, help='Directory path containing batch_0.npy file')
    args = parser.parse_args()
    
    # Load the data file
    data_file = f"{args.file_path}/batch_0.npy"
    print(f"Reading from: {data_file}")
    
    data = np.load(data_file)
    print(f"Array shape: {data.shape}")
    for i in range(data.shape[0]):
        plot_energy_spectrum_from_vorticity(data[i], save_path=f"spectrum_{i}.png", nbins=300)


if __name__ == "__main__":
    main()