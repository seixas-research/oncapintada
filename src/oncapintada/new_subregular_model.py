# -*- coding: utf-8 -*-
# file: new_subregular_model.py

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

import numpy as np
from typing import Optional, Union
from itertools import combinations_with_replacement
from .constants import kJmol, R

class BaseAlloy:
    """Base class for alloy thermodynamic calculations."""
    
    def __init__(self, energy_matrix: Optional[np.ndarray] = None, dilution: float = 0.0):
        self.energy_matrix = energy_matrix
        self.dilution = dilution

    @property
    def energy_matrix(self) -> Optional[np.ndarray]:
        return self.energy_matrix

    @energy_matrix.setter
    def energy_matrix(self, value: Optional[np.ndarray]):
        if value is not None:
            if value.ndim != 2 or value.shape[0] != value.shape[1]:
                raise ValueError("Energy matrix must be square.")
        self.energy_matrix = value

    @property
    def dilution(self) -> float:
        return self.dilution

    @dilution.setter
    def dilution(self, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError("Dilution parameter must be between 0 and 1.")
        self.dilution = value

    def _get_mij_matrix(self) -> np.ndarray:
        """Calculates the Mij interaction matrix."""
        E = self.energy_matrix
        if E is None:
            raise ValueError("Energy matrix is not set.")
        
        x0 = self.dilution
        d = np.diag(E)
        # M_ij = E_ij - ( x0 * E_ii + (1-x0) * E_jj )
        return E - (x0 * d[np.newaxis, :] + (1 - x0) * d[:, np.newaxis])

    def _convert_energy(self, value: np.ndarray, to_unit: str, reverse: bool = False) -> np.ndarray:
        """Internal helper for unit conversions."""
        conversion = kJmol # eV/atom to kJ/mol
        if reverse:
            return value / conversion if to_unit == "eV/atom" else value
        return value * conversion if to_unit == "kJ/mol" else value

class MultiComponentAlloy(BaseAlloy):
    """Represents a multicomponent alloy using the subregular mixing model."""

    def simplex_grid(self, n_components: Optional[int] = None, resolution: int = 10) -> np.ndarray:
        """Generates a grid of compositions on an N-dimensional simplex."""
        n = n_components or (self.energy_matrix.shape[0] if self.energy_matrix is not None else None)
        if n is None or n < 2:
            raise ValueError("Number of components must be at least 2.")
        
        grid = [np.bincount(c, minlength=n) for c in combinations_with_replacement(range(n), resolution)]
        return np.array(grid) / resolution

    def enthalpy_of_mixing(self, x: np.ndarray, normalized: bool = True, unit: str = "kJ/mol") -> np.ndarray:
        """
        Calculate enthalpy of mixing using vectorized matrix operations.
        Supports input x of shape (N_components,) or (N_samples, N_components).
        """
        if unit not in ["eV/atom", "kJ/mol"]:
            raise ValueError("Invalid unit. Use 'eV/atom' or 'kJ/mol'.")
            
        x = np.atleast_2d(x)
        m = self._get_mij_matrix()
        n = m.shape[0]
        eps = 1e-8

        # Vectorized calculation:
        # We need sum_{i<j} (Mij * xj + Mji * xi) * xi * xj / (xi + xj)
        # Let's compute the interaction for all pairs (i, j) and take the upper triangle
        
        # Term1: (Mij * xj) -> shape (Samples, i, j)
        term1 = m[np.newaxis, :, :] * x[:, np.newaxis, :] 
        # Term2: (Mji * xi) -> shape (Samples, i, j)
        term2 = m.T[np.newaxis, :, :] * x[:, :, np.newaxis]
        
        # numerator: (Term1 + Term2) * xi * xj
        xi_xj = x[:, :, np.newaxis] * x[:, np.newaxis, :]
        h_matrix = (term1 + term2) * xi_xj
        
        if normalized:
            xi_plus_xj = x[:, :, np.newaxis] + x[:, np.newaxis, :] + eps
            h_matrix /= xi_plus_xj
            
        # Sum over upper triangle (i < j) to avoid double counting
        h_mix = np.sum(np.triu(h_matrix, k=1), axis=(1, 2))
        
        # Flatten if input was 1D
        result = h_mix if h_mix.size > 1 else h_mix[0]
        return self._convert_energy(result, unit)

    def configurational_entropy(self, x: np.ndarray, unit: str = "kJ/(mol*K)") -> np.ndarray:
        """Calculates configurational entropy: S = -R * sum(xi * ln(xi))."""
        x = np.clip(x, 1e-12, 1.0)
        s_config = -R * np.sum(x * np.log(x), axis=-1)
        
        if unit == "eV/(atom*K)":
            return s_config / kJmol
        return s_config

    def gibbs_free_energy_of_mixing(self, x: np.ndarray, t: Union[float, np.ndarray], 
                                   unit: str = "kJ/mol") -> np.ndarray:
        """G = H - T*S"""
        x = np.atleast_2d(x)
        t = np.atleast_1d(t)
        
        h = self.enthalpy_of_mixing(x, unit=unit) # Shape (Samples,)
        s = self.configurational_entropy(x, unit="kJ/(mol*K)") # Standardized R unit
        
        # Match S unit to H unit for calculation
        if unit == "eV/atom":
            s /= kJmol

        # Broadcast across temperatures: (Samples, 1) - (1, Temps) * (Samples, 1)
        # Result shape: (Samples, Temps)
        return h[:, np.newaxis] - t[np.newaxis, :] * s[:, np.newaxis]


class BinaryAlloy(MultiComponentAlloy):
    """
    Specialized class for binary alloys. 
    Accepts x as a single float (composition of component 0) or array.
    """
    
    def _prepare_x(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Converts scalar composition x into [x, 1-x] array."""
        x = np.atleast_1d(x)
        return np.column_stack([x, 1 - x])

    def enthalpy_of_mixing(self, x: Union[float, np.ndarray], unit: str = "kJ/mol") -> np.ndarray:
        return super().enthalpy_of_mixing(self._prepare_x(x), normalized=False, unit=unit)

    def configurational_entropy(self, x: Union[float, np.ndarray], unit: str = "kJ/(mol*K)") -> np.ndarray:
        return super().configurational_entropy(self._prepare_x(x), unit=unit)

    def gibbs_free_energy_of_mixing(self, x: Union[float, np.ndarray], t: Union[float, np.ndarray], 
                                   unit: str = "kJ/mol") -> np.ndarray:
        return super().gibbs_free_energy_of_mixing(self._prepare_x(x), t, unit=unit)