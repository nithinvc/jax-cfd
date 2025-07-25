# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pseudospectral equations."""

import dataclasses
from typing import Callable, Optional

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

TimeDependentForcingFn = Callable[[float], types.Array]
RandomSeed = int
ForcingModule = Callable[[grids.Grid, RandomSeed], TimeDependentForcingFn]


@dataclasses.dataclass
class KuramotoSivashinsky(time_stepping.ImplicitExplicitODE):
    """Kuramoto–Sivashinsky (KS) equation split in implicit and explicit parts.

    The KS equation is
      u_t = - u_xx - u_xxxx - 1/2 * (u ** 2)_x

    Implicit parts are the linear terms and explicit parts are the non-linear
    terms.

    Attributes:
      grid: underlying grid of the process
      smooth: smooth the non-linear term using the 3/2-rule
    """
    grid: grids.Grid
    smooth: bool = True

    def __post_init__(self):
        self.kx, = self.grid.rfft_axes()
        self.two_pi_i_k = 2j * jnp.pi * self.kx
        self.linear_term = -self.two_pi_i_k ** 2 - self.two_pi_i_k ** 4
        self.rfft = spectral_utils.truncated_rfft if self.smooth else jnp.fft.rfft
        self.irfft = spectral_utils.padded_irfft if self.smooth else jnp.fft.irfft

    def explicit_terms(self, uhat):
        """Non-linear parts of the equation, namely `- 1/2 * (u ** 2)_x`."""
        uhat_squared = self.rfft(jnp.square(self.irfft(uhat)))
        return -0.5 * self.two_pi_i_k * uhat_squared

    def implicit_terms(self, uhat):
        """Linear parts of the equation, namely `- u_xx - u_xxxx`."""
        return self.linear_term * uhat

    def implicit_solve(self, uhat, time_step):
        """Solves for `implicit_terms`, implicitly."""
        # TODO(dresdner) the same for all linear terms. generalize/refactor?
        return 1 / (1 - time_step * self.linear_term) * uhat


@dataclasses.dataclass
class ForcedBurgersEquation(time_stepping.ImplicitExplicitODE):
    """Burgers' Equation with the option to add a time-dependent forcing function."""
    viscosity: float
    grid: grids.Grid
    seed: int = 0
    forcing_module: Optional[
        ForcingModule] = spectral_forcings.random_forcing_module
    _forcing_fn = None

    def __post_init__(self):
        self.kx, = self.grid.rfft_axes()
        self.two_pi_i_k = 2j * jnp.pi * self.kx
        self.linear_term = self.viscosity * self.two_pi_i_k ** 2
        self.rfft = spectral_utils.truncated_rfft
        self.irfft = spectral_utils.padded_irfft
        if self.forcing_module is None:
            self._forcing_fn = lambda t: jnp.zeros(1)
        else:
            self._forcing_fn = self.forcing_module(self.grid, self.seed)

    def explicit_terms(self, state):
        uhat, t = state
        dudx = self.two_pi_i_k * uhat

        f = self._forcing_fn(t)
        fhat = jnp.fft.rfft(f)

        advection = - self.rfft(self.irfft(uhat) * self.irfft(dudx))

        return (fhat + advection, 1.0)

    def implicit_terms(self, state):
        uhat, _ = state
        return (self.linear_term * uhat, 0.0)

    def implicit_solve(self, state, time_step):
        uhat, t = state
        return (1 / (1 - time_step * self.linear_term) * uhat, t)


def BurgersEquation(viscosity: float, grid: grids.Grid, seed: int = 0):
    """Standard, unforced Burgers' equation."""
    return ForcedBurgersEquation(
        viscosity=viscosity, grid=grid, seed=seed, forcing_module=None)


# pylint: disable=invalid-name
def _get_grid_variable(arr,
                       grid,
                       bc=boundaries.periodic_boundary_conditions(2),
                       offset=(0.5, 0.5)):
    return grids.GridVariable(grids.GridArray(arr, offset, grid), bc)


@dataclasses.dataclass
class NavierStokes2D(time_stepping.ImplicitExplicitODE):
    """Breaks the Navier-Stokes equation into implicit and explicit parts.

    Implicit parts are the linear terms and explicit parts are the non-linear
    terms.

    Attributes:
      viscosity: strength of the diffusion term
      grid: underlying grid of the process
      smooth: smooth the advection term using the 2/3-rule.
      forcing_fn: forcing function, if None then no forcing is used.
      drag: strength of the drag. Set to zero for no drag.
    """
    viscosity: float
    grid: grids.Grid
    drag: float = 0.
    smooth: bool = True
    forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
    _forcing_fn_with_grid = None
    counter: int = dataclasses.field(default=0)  # Instance variable

    def save_fields(self, fields_dict):
        """Save a field using jax.debug.callback to avoid JAX transformation issues."""
        def _save(fields_dict, step_num):
            advection = fields_dict["advection"]
            shape = advection.shape[-1]
            if self.forcing_fn is not None:
                output_dir = f"/data/divyam123/meta-cfd/jaxcfd_fields/2dtf_{self.viscosity}_{shape}"
            else:
                output_dir = f"/data/divyam123/meta-cfd/jaxcfd_fields/2dns_{self.viscosity}_{shape}"
            os.makedirs(output_dir, exist_ok=True)
            total_files = len(os.listdir(output_dir))
            if step_num == 0:
                total_files = total_files + 1
            save_path = os.path.join(output_dir, f"index_{total_files}")
            os.makedirs(save_path, exist_ok=True)
            for name_str, field_array in fields_dict.items():
                field_np = np.array(field_array)
                np.save(os.path.join(
                    save_path, f"{name_str}_{step_num}.npy"), field_np)
            # print(f"Saved {name_str}_{step_num} to {save_path}")

        # Use a static counter that gets passed as an argument
        jax.debug.callback(_save, fields_dict, self.counter)

    def increment_counter(self):
        """Increment the counter outside of JAX transformations."""
        self.counter += 1

    def save_timestep_data(self, vorticity_hat):
        """Save all field data after a complete timestep."""
        velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
        vxhat, vyhat = velocity_solve(vorticity_hat)
        vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

        grad_x_hat = 2j * jnp.pi * self.kx * vorticity_hat
        grad_y_hat = 2j * jnp.pi * self.ky * vorticity_hat
        grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

        advection = -(grad_x * vx + grad_y * vy)
        diffusion_term = jnp.fft.irfftn(
            self.viscosity * self.laplace * vorticity_hat)
        damping_term = jnp.fft.irfftn(self.drag * vorticity_hat)

        self.save_fields(fields_dict={
            "vorticity": jnp.fft.irfftn(vorticity_hat),
            "vx": vx,
            "vy": vy,
            "advection": advection,
            "w_dx": grad_x,
            "w_dy": grad_y,
            "diffusion_term": diffusion_term,
            "damping_term": damping_term})

    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.laplace = (jnp.pi * 2j)**2 * (self.kx**2 + self.ky**2)
        self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        self.linear_term = self.viscosity * self.laplace - self.drag

        # setup the forcing function with the caller-specified grid.
        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)

    def explicit_terms(self, vorticity_hat):
        velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
        vxhat, vyhat = velocity_solve(vorticity_hat)
        vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

        grad_x_hat = 2j * jnp.pi * self.kx * vorticity_hat
        grad_y_hat = 2j * jnp.pi * self.ky * vorticity_hat
        grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

        advection = -(grad_x * vx + grad_y * vy)
        advection_hat = jnp.fft.rfftn(advection)

        if self.smooth is not None:
            advection_hat *= self.filter_

        terms = advection_hat

        if self.forcing_fn is not None:
            fx, fy = self._forcing_fn_with_grid((_get_grid_variable(vx, self.grid),
                                                 _get_grid_variable(vy, self.grid)))
            fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
            terms += spectral_utils.spectral_curl_2d((self.kx, self.ky),
                                                     (fx_hat, fy_hat))
        # Save key fields using debug callbacks
        # terms = jax.block_until_ready(terms)
        # self.viscosity * self.laplace - self.drag
        diffusion_term = jnp.fft.irfftn(
            self.viscosity * self.laplace * vorticity_hat)
        damping_term = jnp.fft.irfftn(self.drag * vorticity_hat)

        # self.save_fields(fields_dict={
        #     "vorticity": jnp.fft.irfftn(vorticity_hat),
        #     "vx": vx,
        #     "vy": vy,
        #     "advection": advection,
        #     "w_dx": grad_x,
        #     "w_dy": grad_y,
        #     "diffusion_term": diffusion_term,
        #     "damping_term": damping_term})
        # self.increment_counter()

        return terms

    def implicit_terms(self, vorticity_hat):
        return self.linear_term * vorticity_hat

    def implicit_solve(self, vorticity_hat, time_step):
        return 1 / (1 - time_step * self.linear_term) * vorticity_hat


# pylint: disable=g-doc-args,g-doc-return-or-yield,invalid-name
def ForcedNavierStokes2D(viscosity, grid, smooth):
    """Sets up the flow that is used in Kochkov et al. [1].

    The authors of [1] based their work on Boffetta et al. [2].

    References:
      [1] Machine learning–accelerated computational fluid dynamics. Dmitrii
      Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, Stephan
      Hoyer Proceedings of the National Academy of Sciences May 2021, 118 (21)
      e2101784118; DOI: 10.1073/pnas.2101784118.
      https://doi.org/10.1073/pnas.2101784118

      [2] Boffetta, Guido, and Robert E. Ecke. "Two-dimensional turbulence."
      Annual review of fluid mechanics 44 (2012): 427-451.
      https://doi.org/10.1146/annurev-fluid-120710-101240
    """
    wave_number = 4
    offsets = ((0, 0), (0, 0))
    # pylint: disable=g-long-lambda
    def forcing_fn(grid): return forcings.kolmogorov_forcing(
        grid, k=wave_number, offsets=offsets)
    return NavierStokes2D(
        viscosity,
        grid,
        drag=0.1,
        smooth=smooth,
        forcing_fn=forcing_fn)


@dataclasses.dataclass
class NonlinearSchrodinger(time_stepping.ImplicitExplicitODE):
    """Nonlinear schrodinger equation split in implicit and explicit parts.

    The NLS equation is
      `psi_t = -i psi_xx/8 - i|psi|^2 psi/2`

    Attributes:
      grid: underlying grid of the process
      smooth: smooth the non-linear by upsampling 2x in fourier and truncating
    """
    grid: grids.Grid
    smooth: bool = True

    def __post_init__(self):
        self.kx, = self.grid.fft_axes()
        assert len(self.kx) % 2 == 0, "Odd grid sizes not supported, try N even"
        self.two_pi_i_k = 2j * jnp.pi * self.kx
        self.fft = spectral_utils.truncated_fft_2x if self.smooth else jnp.fft.fft
        self.ifft = spectral_utils.padded_ifft_2x if self.smooth else jnp.fft.ifft

    def explicit_terms(self, psihat):
        """Non-linear part of the equation `-i|psi|^2 psi/2`."""
        psi = self.ifft(psihat)
        ipsi_cubed = 1j * psi * jnp.abs(psi)**2
        ipsi_cubed_hat = self.fft(ipsi_cubed)
        return -ipsi_cubed_hat / 2

    def implicit_terms(self, psihat):
        """The diffusion term `-i psi_xx/8` to be handled implicitly."""
        return -1j * psihat * self.two_pi_i_k**2 / 8

    def implicit_solve(self, psihat, time_step):
        """Solves for `implicit_terms`, implicitly."""
        return psihat / (1 - time_step * (-1j * self.two_pi_i_k**2 / 8))
