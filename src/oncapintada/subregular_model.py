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

import numpy as np
import pandas as pd
from ase import Atoms
from typing import Optional, List, deprecated
from itertools import combinations_with_replacement

class BinaryAlloy:
    '''
    Class to represent a binary alloy and calculate the enthalpy of mixing based on the subregular mixing model.
    '''
    def __init__(self, energy_matrix: Optional[np.ndarray] = None, dilution: float = 0.0):
        self.energy_matrix = energy_matrix
        if self.energy_matrix is not None and self.energy_matrix.shape[0] != self.energy_matrix.shape[1]:
            raise ValueError("Energy matrix must be square.")
        self.dilution = dilution
        if self.dilution < 0.0 or self.dilution > 1.0:
            raise ValueError("Dilution parameter must be between 0 and 1.")
        
    def get_energy_matrix(self) -> np.ndarray:
        return self.energy_matrix
    
    def set_energy_matrix(self, energy_matrix: Optional[np.ndarray] = None):
        self.energy_matrix = energy_matrix

    @deprecated(reason="This method is inefficient. Use get_Mij instead.")
    def get_Mij_deprecated(self):
        M = np.zeros_like(self.energy_matrix)
        x0 = self.dilution
        E = self.energy_matrix
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                M[i, j] = E[i, j]  - ( x0 * E[i, i] + (1-x0) * E[j, j] )
        return M
    
    def get_Mij(self):
        '''
        Calculate the Mij matrix based on the energy matrix and dilution parameter.
            M_ij = E_ij - ( x0 * E_ii + (1-x0) * E_jj )
        '''
        E = self.energy_matrix
        x0 = self.dilution
        d = np.diag(E)  # Extract the diagonal elements (E_ii)
        
        M = E - ( x0 * d[:, np.newaxis] + (1 - x0) * d[np.newaxis, :] )
        return M
    
    def get_enthalpy_of_mixing(self, x: np.ndarray) -> np.ndarray:
        '''
        Calculate the enthalpy of mixing for a subregular mixing model based on the Mij matrix and composition x.
            H_{mix} = (M[0,1] * x + M[1,0] * (1-x)) * x * (1-x)
        '''
        M = self.get_Mij()
        h = (M[0,1] * x + M[1,0] * (1-x)) * x * (1-x)
        return h


# To calculate the enthalpy of mixing based on DSI model for multi-component systems.
class MultiComponentAlloy:
    def __init__(self, energy_matrix: Optional[np.ndarray] = None, dilution: float = 0.0):
        self.energy_matrix = energy_matrix
        self.dilution = dilution
        if self.dilution < 0.0 or self.dilution > 1.0:
            raise ValueError("Dilution parameter must be between 0 and 1.")
        
    
    def simplex_grid(self, N: int, resolution: int) -> np.ndarray:
        '''
        Generate a grid of points on the N-dimensional simplex with a given resolution. Each point represents a composition of the alloy.
        The simplex is defined as the set of points where the sum of the components is equal to 1 (i.e., x_1 + x_2 + ... + x_N = 1), and each component is non-negative (x_i >= 0 for all i).

        Parameters:
        -----------
        N (int): Number of components in the alloy.
        resolution (int): The number of points to generate along each edge of the simplex.

        Returns:
        ---------
        np.ndarray: An array of shape (num_points, N).
        '''
        grid = []

        # Generate all combinations of N components that sum to the resolution
        for combo in combinations_with_replacement(range(N), resolution):
            point = np.bincount(combo, minlength=N)
            grid.append(point)
        return np.array(grid) / resolution
    
    def get_Mij(self):
        '''
        Calculate the Mij matrix based on the energy matrix and dilution parameter.
            M_ij = E_ij - ( x0 * E_ii + (1-x0) * E_jj )
        '''
        E = self.energy_matrix
        x0 = self.dilution
        d = np.diag(E)            # Extract the diagonal elements (E_ii)
        
        M = E - ( x0 * d[:, np.newaxis] + (1 - x0) * d[np.newaxis, :] )
        return M
    
    def get_enthalpy_of_mixing(self, X: np.ndarray, normalized: bool = True) -> np.ndarray:
        '''
        Calculate the enthalpy of mixing for a multi-component alloy based on the Mij matrix and composition X.
            H_{mix} = sum_{j=1}^{N} sum_{i=1}^{N-1} (M[i,j] * X[j] + M[j,i] * X[i]) * X[i] * X[j]/(X[i] + X[j])
        '''
        M = self.get_Mij()        # Matrix of interaction parameters (M_{i[j]})
        N = M.shape[0]            # Number of components
        eps=1e-8                  # Small value to prevent division by zero
        X = np.clip(X, eps, 1.0)  # Ensure that compositions are not exactly zero to avoid division by zero

        h = 0.0                   # Initialize enthalpy of mixing
        for i in range(N):
            for j in range(i+1, N):
                if normalized:
                    h += (M[i,j] * X[j] + M[j,i] * X[i]) * X[i] * X[j] / (X[i] + X[j])
                else:
                    h += (M[i,j] * X[j] + M[j,i] * X[i]) * X[i] * X[j]
    
        return h
        