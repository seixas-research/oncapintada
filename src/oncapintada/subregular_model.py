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
# import pandas as pd
# from ase import Atoms
from typing import Optional
from itertools import combinations_with_replacement

kJ = 96.485364               # Conversion factor from eV/atom to kJ/mol (1 eV/atom = 96.485364 kJ/mol)
R = 8.31446261815324 / 1000  # Universal gas constant in kJ/(mol*K)

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
        if energy_matrix is not None and energy_matrix.shape[0] != energy_matrix.shape[1]:
            raise ValueError("Energy matrix must be square.")
        self.energy_matrix = energy_matrix


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


    def get_enthalpy_of_mixing(self, x: np.ndarray, unit: str="kJ/mol") -> np.ndarray:
        '''
        Calculate the enthalpy of mixing for a subregular mixing model based on the Mij matrix and composition x.
            H_{mix} = (M[0,1] * x + M[1,0] * (1-x)) * x * (1-x)
        '''
        if unit not in ["eV/atom", "kJ/mol"]:
            raise ValueError("Invalid unit. Supported units are 'eV/atom' and 'kJ/mol'.")
        
        M = self.get_Mij()
        h = (M[0,1] * x + M[1,0] * (1-x)) * x * (1-x) * kJ  # Convert eV/atom to kJ/mol

        if unit == "eV/atom":
            h /= kJ  # Convert kJ/mol back to eV/atom
        return h
    
    def get_configurational_entropy(self, x: np.ndarray, unit: str="kJ/(mol*K)") -> np.ndarray:
        '''
        Calculate the configurational entropy of mixing for a binary alloy based on the composition x.
            S_{config} = -R * (x * ln(x) + (1-x) * ln(1-x))
        '''
        if unit not in ["eV/(atom*K)", "kJ/(mol*K)"]:
            raise ValueError("Invalid unit. Supported units are 'eV/(atom*K)' and 'kJ/(mol*K)'.")
        
        eps = 1e-8                  # Small value to prevent log(0) issues
        x = np.clip(x, eps, 1-eps)  # Ensure that compositions are not exactly zero to avoid log(0)

        s_config = -R * (x * np.log(x) + (1-x) * np.log(1-x))

        if unit == "eV/(atom*K)":
            s_config /= kJ          # Convert kJ/(mol*K) to eV/(atom*K)
        return s_config
    
    def get_gibbs_free_energy_of_mixing(self, x: np.ndarray, t: np.ndarray, unit: str="kJ/mol", unit_s: str="kJ/(mol*K)") -> np.ndarray:
        '''
        Calculate the Gibbs free energy of mixing for a binary alloy based on the enthalpy and configurational entropy.
            G_{mix} = H_{mix} - T * S_{config}

        Parameters:
        -----------
        x (np.ndarray): An array of shape (N,) representing the composition of the alloy, where N is the number of components. The elements of x should sum to 1.
        t (np.ndarray): An array of shape (M,) representing the temperatures at which to calculate the Gibbs free energy.
        unit (str): The unit for the output Gibbs free energy. Supported units are 'eV/atom' and 'kJ/mol'. Default is 'kJ/mol'.
        unit_s (str): The unit for the configurational entropy. Supported units are 'eV/(atom*K)' and 'kJ/(mol*K)'. Default is 'kJ/(mol*K)'.
        Returns:
        ---------
        np.ndarray: An array of shape (N, M) representing the Gibbs free energy of mixing for each composition and temperature. The rows correspond to compositions and the columns correspond to temperatures.
        '''
        if unit not in ["eV/atom", "kJ/mol"]:
            raise ValueError("Invalid unit. Supported units are 'eV/atom' and 'kJ/mol'.")
        
        ntemp = len(t)
        h_mix = self.get_enthalpy_of_mixing(x, unit=unit)
        s_config = self.get_configurational_entropy(x, unit=unit_s)

        enthalpy_matrix = np.tile(h_mix, (ntemp, 1)).T
        entropy_matrix = np.tile(s_config, (ntemp, 1)).T

        gibbs_free_energy = enthalpy_matrix - t[np.newaxis, :] * entropy_matrix

        if unit == "eV/atom":
            gibbs_free_energy /= kJ  # Convert kJ/mol to eV/atom
        return gibbs_free_energy



class MultiComponentAlloy:
    def __init__(self, energy_matrix: Optional[np.ndarray] = None, dilution: float = 0.0):
        '''
         Class to represent a multicomponent alloy and calculate the enthalpy of mixing based on the subregular mixing model.
        '''
        self.energy_matrix = energy_matrix
        if self.energy_matrix is not None and self.energy_matrix.shape[0] != self.energy_matrix.shape[1]:
            raise ValueError("Energy matrix must be square.")
        self.dilution = dilution
        if self.dilution < 0.0 or self.dilution > 1.0:
            raise ValueError("Dilution parameter must be between 0 and 1.")


    def get_energy_matrix(self) -> np.ndarray:
        return self.energy_matrix


    def set_energy_matrix(self, energy_matrix: Optional[np.ndarray] = None):
        if energy_matrix is not None and energy_matrix.shape[0] != energy_matrix.shape[1]:
            raise ValueError("Energy matrix must be square.")
        self.energy_matrix = energy_matrix


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
            H_{mix} = sum_{j=1}^{N} sum_{i=1}^{N-1} (M[i,j] * X[j] + M[j,i] * X[i]) * X[i] * X[j]/(X[i] + X[j]).

        Parameters:
        -----------
        X (np.ndarray): An array of shape (N,) representing the composition of the alloy, where N is the number of components. The elements of X should sum to 1.
        normalized (bool): If True, the enthalpy of mixing is normalized by the sum of the compositions (X[i] + X[j]). If False, the enthalpy of mixing is calculated without normalization.

        Returns:
        ---------
        np.ndarray: The calculated enthalpy of mixing for the given composition X.
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
        