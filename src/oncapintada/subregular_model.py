# -*- coding: utf-8 -*-
# file: subregular_model.py

# This code is part of Onça-pintada.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from itertools import combinations_with_replacement
from typing import Optional, Union

import numpy as np
from scipy.special import xlogy

from ._validation import positive_integer
from .constants import R, kJmol


class BaseAlloy:
    """Base class for alloy thermodynamic calculations."""

    def __init__(self, energy_matrix: Optional[np.ndarray] = None, dilution: float = 0.0):
        self.energy_matrix = energy_matrix
        self.dilution = dilution

    @property
    def energy_matrix(self) -> Optional[np.ndarray]:
        return self._energy_matrix

    @energy_matrix.setter
    def energy_matrix(self, value: Optional[np.ndarray]):
        if value is not None:
            value = np.asarray(value, dtype=float)
            if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
                raise ValueError("Energy matrix must be square with at least 2 components.")
            if isinstance(self, BinaryAlloy) and value.shape != (2, 2):
                raise ValueError("BinaryAlloy energy matrix must be 2 by 2.")
            if not np.isfinite(value).all():
                raise ValueError("Energy matrix must contain finite values.")
            value = value.copy()
            value.setflags(write=False)
        self._energy_matrix = value

    @property
    def dilution(self) -> float:
        return self._dilution

    @dilution.setter
    def dilution(self, value: float):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Dilution parameter must be between 0 and 1.")
        self._dilution = value

    def Mij(self) -> np.ndarray:
        """
        Calculate the Mij interaction matrix from the energy matrix and dilution parameter::

            M_ij = E_ij - ( x0 * E_ii + (1-x0) * E_jj )

        Here i = solute (row) and j = solvent (column). Same units as the energy matrix.
        """
        E = self.energy_matrix
        if E is None:
            raise ValueError("Energy matrix is not set.")

        x0 = self.dilution
        d = np.diag(E)
        return E - (x0 * d[:, np.newaxis] + (1 - x0) * d[np.newaxis, :])

    def _get_mij_matrix(self) -> np.ndarray:
        return self.Mij()

    def _convert_energy(self, value: np.ndarray, to_unit: str) -> np.ndarray:
        """Internal helper for unit conversions (energies are computed in eV/atom)."""
        return value * kJmol if to_unit == "kJ/mol" else value


class MultiComponentAlloy(BaseAlloy):
    """Represents a multicomponent alloy using the subregular mixing model."""

    def _n_components(self) -> int:
        if self.energy_matrix is None:
            raise ValueError("Energy matrix must be set to determine the number of components.")
        return self.energy_matrix.shape[0]

    def _prepare_compositions(self, x):
        x = np.asarray(x, dtype=float)
        was_1d = x.ndim == 1
        if x.ndim not in (1, 2):
            raise ValueError("Composition must be a 1D vector or 2D array.")
        x = np.atleast_2d(x)
        n = self._n_components() if self.energy_matrix is not None else x.shape[1]
        if x.shape[1] != n or n < 2:
            raise ValueError(
                f"Composition array x must have {n} columns corresponding to the number of components."
            )
        if not np.isfinite(x).all():
            raise ValueError("Composition must contain finite values.")
        if np.any(x < 0):
            raise ValueError("Composition array x must have non-negative values.")
        if not np.allclose(x.sum(axis=1), 1.0, atol=1e-4, rtol=0):
            raise ValueError("Each row of composition array x must sum to 1.")
        return x, was_1d

    def simplex_grid(self, n_components: Optional[int] = None, resolution: int = 10) -> np.ndarray:
        """
        Generate a grid of compositions on the (n_components - 1)-simplex,
        i.e. all x with x_1 + ... + x_N = 1 and x_i >= 0.

        Returns an array of shape (num_points, n_components).
        """
        n = self._n_components() if n_components is None else n_components
        if n < 2:
            raise ValueError("Number of components must be at least 2.")
        n = positive_integer(n, "Number of components")
        resolution = positive_integer(resolution, "Resolution")

        grid = [
            np.bincount(c, minlength=n) for c in combinations_with_replacement(range(n), resolution)
        ]
        return np.array(grid) / resolution

    def line_profile(
        self, npoints: int = 11, y1: Optional[np.ndarray] = None, y2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate compositions along the line between y1 and y2 in the simplex:
            x(p) = y1 * (1-p) + y2 * p, with p in [0, 1].

        Returns an array of shape (npoints, N).
        """
        if not isinstance(y1, np.ndarray) or not isinstance(y2, np.ndarray):
            raise ValueError("y1 and y2 must be numpy arrays.")
        if y1.ndim != 1 or y2.ndim != 1 or y1.shape != y2.shape:
            raise ValueError("y1 and y2 must have the same number of components.")

        if len(y1) < 2 or (self.energy_matrix is not None and len(y1) != self._n_components()):
            raise ValueError("Line endpoints must match the number of components.")
        if not np.isfinite(y1).all() or not np.isfinite(y2).all():
            raise ValueError("Line endpoints must be finite.")
        npoints = positive_integer(npoints, "npoints")
        tol = 1e-4
        if not np.isclose(y1.sum(), 1.0, atol=tol) or not np.isclose(y2.sum(), 1.0, atol=tol):
            raise ValueError("y1 and y2 compositions must sum to 1.")
        if np.any(y1 < 0) or np.any(y2 < 0):
            raise ValueError("y1 and y2 compositions must be non-negative.")
        if npoints < 2:
            raise ValueError("npoints must be at least 2 to generate a line profile.")

        p = np.linspace(0, 1, npoints)[:, np.newaxis]
        return y1 * (1 - p) + y2 * p

    def enthalpy_of_mixing(
        self, x: np.ndarray, normalized: bool = True, unit: str = "kJ/mol"
    ) -> np.ndarray:
        """
        Calculate the enthalpy of mixing (vectorized):
            H_mix = sum_{i<j} (M_ij * x_j + M_ji * x_i) * x_i * x_j / (x_i + x_j)

        Supports input x of shape (N_components,) or (N_samples, N_components).
        """
        if unit not in ["eV/atom", "kJ/mol"]:
            raise ValueError("Invalid unit. Supported units are 'eV/atom' and 'kJ/mol'.")

        x, was_1d = self._prepare_compositions(x)
        m = self.Mij()

        # (M_ij * x_j) and (M_ji * x_i) for all pairs, shape (Samples, i, j)
        term1 = m[np.newaxis, :, :] * x[:, np.newaxis, :]
        term2 = m.T[np.newaxis, :, :] * x[:, :, np.newaxis]

        xi_xj = x[:, :, np.newaxis] * x[:, np.newaxis, :]
        h_matrix = (term1 + term2) * xi_xj

        if normalized:
            xi_plus_xj = x[:, :, np.newaxis] + x[:, np.newaxis, :]
            h_matrix = np.divide(
                h_matrix, xi_plus_xj, out=np.zeros_like(h_matrix), where=xi_plus_xj > 0
            )

        # Sum over upper triangle (i < j) to avoid double counting
        h_mix = np.sum(np.triu(h_matrix, k=1), axis=(1, 2))

        result = h_mix[0] if was_1d else h_mix
        return self._convert_energy(result, unit)

    def configurational_entropy(self, x: np.ndarray, unit: str = "kJ/(mol*K)") -> np.ndarray:
        """Calculates configurational entropy: S = -R * sum(xi * ln(xi))."""
        if unit not in ["eV/(atom*K)", "kJ/(mol*K)"]:
            raise ValueError("Invalid unit. Supported units are 'eV/(atom*K)' and 'kJ/(mol*K)'.")

        x, was_1d = self._prepare_compositions(x)
        s_config = -R * np.sum(xlogy(x, x), axis=-1)
        if was_1d:
            s_config = s_config[0]

        if unit == "eV/(atom*K)":
            return s_config / kJmol
        return s_config

    def gibbs_free_energy_of_mixing(
        self, x: np.ndarray, t: Union[float, np.ndarray], unit: str = "kJ/mol"
    ) -> np.ndarray:
        """
        G_mix = H_mix - T * S_config

        Returns an array of shape (N_samples, N_temperatures).
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0):
            raise ValueError("Temperatures must be a finite, non-negative scalar or 1D array.")

        h = np.atleast_1d(self.enthalpy_of_mixing(x, unit=unit))
        s = np.atleast_1d(self.configurational_entropy(x, unit="kJ/(mol*K)"))

        # Match S unit to H unit for the calculation
        if unit == "eV/atom":
            s = s / kJmol

        return h[:, np.newaxis] - t[np.newaxis, :] * s[:, np.newaxis]


class BinaryAlloy(MultiComponentAlloy):
    """
    Specialized class for binary alloys A_{1-x}B_x.
    Accepts x as a scalar or 1D array with the fraction of the second component (B).
    """

    def _prepare_x(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Converts composition x into rows [1-x, x]; passes 2D arrays through unchanged."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            if x.shape[1] != 2:
                raise ValueError("Binary composition arrays must have 2 columns.")
            return x
        if x.ndim > 2:
            raise ValueError("Binary composition must be scalar, 1D or 2D.")
        x = np.atleast_1d(x)
        return np.column_stack([1 - x, x])

    def enthalpy_of_mixing(
        self, x: Union[float, np.ndarray], normalized: bool = False, unit: str = "kJ/mol"
    ) -> np.ndarray:
        # For a binary alloy x_i + x_j = 1, so the normalization denominator is 1
        return super().enthalpy_of_mixing(self._prepare_x(x), normalized=normalized, unit=unit)

    def configurational_entropy(
        self, x: Union[float, np.ndarray], unit: str = "kJ/(mol*K)"
    ) -> np.ndarray:
        return super().configurational_entropy(self._prepare_x(x), unit=unit)

    def gibbs_free_energy_of_mixing(
        self, x: Union[float, np.ndarray], t: Union[float, np.ndarray], unit: str = "kJ/mol"
    ) -> np.ndarray:
        return super().gibbs_free_energy_of_mixing(self._prepare_x(x), t, unit=unit)
