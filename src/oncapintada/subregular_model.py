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
from typing import Optional
from itertools import combinations_with_replacement
from .constants import kJmol, R


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


    def get_dilution(self) -> float:
        return self.dilution
    
    
    def set_dilution(self, dilution: float):
        if dilution < 0.0 or dilution > 1.0:
            raise ValueError("Dilution parameter must be between 0 and 1.")
        self.dilution = dilution


    def Mij(self) -> np.ndarray:
        '''
        Calculate the Mij matrix based on the energy matrix and dilution parameter.
            M_ij = E_ij - ( x0 * E_ii + (1-x0) * E_jj )
        '''
        E = self.energy_matrix
        x0 = self.dilution
        d = np.diag(E)  # Extract the diagonal elements (E_ii)
        
        M = E - ( x0 * d[np.newaxis, :] + (1 - x0) * d[:, np.newaxis] )
        return M
    
    
    def Mij_temp(self):
        E = self.energy_matrix
        x0 = self.dilution
        d = np.diag(E)  # Extract the diagonal elements (E_ii)
        
        M = E - ( x0 * d[:, np.newaxis] + (1 - x0) * d[np.newaxis, :] )
        return M
    

    def enthalpy_of_mixing(self, x: np.ndarray, unit: str="kJ/mol") -> np.ndarray:
        '''
        Calculate the enthalpy of mixing for a subregular mixing model based on the Mij matrix and composition x.
            H_{mix} = (M[0,1] * x + M[1,0] * (1-x)) * x * (1-x)
        '''
        if unit not in ["eV/atom", "kJ/mol"]:
            raise ValueError("Invalid unit. Supported units are 'eV/atom' and 'kJ/mol'.")
        
        M = self.Mij()
        h = (M[0,1] * x + M[1,0] * (1-x)) * x * (1-x)

        if unit == "kJ/mol":
            h *= kJmol  # Convert eV/atom to kJ/mol
        return h


    def configurational_entropy(self, x: np.ndarray, unit: str="kJ/(mol*K)") -> np.ndarray:
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
            s_config /= kJmol          # Convert kJ/(mol*K) to eV/(atom*K)
        return s_config


    def gibbs_free_energy_of_mixing(self, x: np.ndarray, t: np.ndarray, unit: str="kJ/mol", unit_s: str="kJ/(mol*K)") -> np.ndarray:
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
        h_mix = self.enthalpy_of_mixing(x, unit=unit)
        s_config = self.configurational_entropy(x, unit=unit_s)

        enthalpy_matrix = np.tile(h_mix, (ntemp, 1)).T
        entropy_matrix = np.tile(s_config, (ntemp, 1)).T

        gibbs_free_energy = enthalpy_matrix - t[np.newaxis, :] * entropy_matrix

        if unit == "eV/atom":
            gibbs_free_energy /= kJmol  # Convert kJ/mol to eV/atom
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


    def simplex_grid(self, N: Optional[int] = None, resolution: int = 10) -> np.ndarray:
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
        if N is None:
            if self.energy_matrix is None:
                raise ValueError("Energy matrix must be set to determine the number of components.")
            if self.energy_matrix.shape[0] != self.energy_matrix.shape[1]:
                raise ValueError("Energy matrix must be square.")
            N = self.energy_matrix.shape[0]
        
        if N < 2:
            raise ValueError("Number of components N must be at least 2.")
        
        if not isinstance(resolution, int):
            raise TypeError("Resolution must be an integer.")
        
        if resolution < 1:
            raise ValueError("Resolution must be a positive integer.")

        # Generate all combinations of N components that sum to the resolution
        for combo in combinations_with_replacement(range(N), resolution):
            point = np.bincount(combo, minlength=N)
            grid.append(point)
        return np.array(grid) / resolution


    def line_profile(self, N: Optional[int] = None, npoints: int = 11, y1: np.ndarray = None, y2: np.ndarray = None) -> np.ndarray:
        '''
        Generate a line of points between two compositions y1 and y2 in the N-dimensional simplex. Each point represents a composition of the alloy.
        The line is defined as the set of points where the composition is a linear combination of y1 and y2 (i.e., x = y1 * (1-p) + y2 * p), where p is a parameter that varies from 0 to 1.

        Parameters:
        -----------
        N (int): Number of components in the alloy.
        npoints (int): The number of points to generate along the line.
        y1 (np.ndarray): An array of shape (N,) representing the first composition.
        y2 (np.ndarray): An array of shape (N,) representing the second composition.

        Returns:
        ---------
        np.ndarray: An array of shape (npoints, N) representing the compositions along the line between y1 and y2. The rows correspond to points along the line, and the columns correspond to components.
        '''
        if N is None:
            if self.energy_matrix is None:
                raise ValueError("Energy matrix must be set to determine the number of components.")
            if self.energy_matrix.shape[0] != self.energy_matrix.shape[1]:
                raise ValueError("Energy matrix must be square.")
            N = self.energy_matrix.shape[0]

        if not isinstance(y1, np.ndarray) or not isinstance(y2, np.ndarray):
            raise ValueError("y1 and y2 must be numpy arrays.")
        
        tol = 1e-4  # Small tolerance to check if compositions sum to 1
        if y1.sum() < 1.0 - tol or y1.sum() > 1.0 + tol:
            raise ValueError("y1 composition must sum to 1.")
        
        if y2.sum() < 1.0 - tol or y2.sum() > 1.0 + tol:
            raise ValueError("y2 composition must sum to 1.")

        if np.any(y1 < 0) or np.any(y2 < 0):
            raise ValueError("y1 and y2 compositions must be non-negative.")
        
        if y1.shape[0] != y2.shape[0]:
            raise ValueError("y1 and y2 must have the same number of components.")
        
        if y1.shape[0] != N:
            raise ValueError("y1 and y2 must have the same number of components as the energy matrix.")
        
        if npoints < 2:
            raise ValueError("npoints must be at least 2 to generate a line profile.")
        
        p = np.linspace(0, 1, npoints)[:, np.newaxis]

        return y1 * (1-p) + y2 * p


    def Mij(self) -> np.ndarray:
        '''
        Calculate the Mij matrix based on the energy matrix and dilution parameter.
            M_ij = E_ij - ( x0 * E_ii + (1-x0) * E_jj )
        '''
        E = self.energy_matrix
        x0 = self.dilution
        d = np.diag(E)            # Extract the diagonal elements (E_ii)

        M = E - ( x0 * d[np.newaxis, :] + (1 - x0) * d[:, np.newaxis] )
        return M


    def enthalpy_of_mixing(self, X: np.ndarray, normalized: bool = True, unit='kJ/mol') -> np.ndarray:
        '''
        Calculate the enthalpy of mixing for a multi-component alloy based on the Mij matrix and composition X.
            H_{mix} = sum_{j=1}^{N} sum_{i=1}^{N-1} (M[i,j] * X[j] + M[j,i] * X[i]) * X[i] * X[j]/(X[i] + X[j]).

        Parameters:
        -----------
        X (np.ndarray): An array of shape (num_points, N) representing the compositions of the alloy, where num_points is the number of compositions and N is the number of components. Each row of X should sum to 1.
        normalized (bool): If True, the enthalpy of mixing is normalized by the sum of the compositions (X[i] + X[j]). Default is True.
        unit (str): The unit for the output enthalpy of mixing. Supported units are 'eV/atom' and 'kJ/mol'. Default is 'kJ/mol'.

        Returns:
        ---------
        np.ndarray: The calculated enthalpy of mixing for the given composition X.
        '''
        M = self.Mij()            # Matrix of interaction parameters (M_{i[j]})
        N = M.shape[0]            # Number of components
        eps=1e-8                  # Small value to prevent division by zero

        if X.shape[1] != N:
            raise ValueError(f"Composition array X must have {N} columns corresponding to the number of components.")
        
        if not np.allclose(X.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("Each row of composition array X must sum to 1.")
        
        if np.any(X < 0):
            raise ValueError("Composition array X must have non-negative values.")
        
        if not isinstance(unit, str):
            raise ValueError("Unit must be a string.")

        if unit not in ["eV/atom", "kJ/mol"]:
            raise ValueError("Invalid unit. Supported units are 'eV/atom' and 'kJ/mol'.")

        h_mix = []
        # for each X in the grid, calculate the enthalpy of mixing
        for ix in X:
            h = 0.0                   # Initialize enthalpy of mixing for this composition
            for i in range(N):
                for j in range(i+1, N):
                    if normalized:
                        h += (M[i,j] * ix[j] + M[j,i] * ix[i]) * ix[i] * ix[j] / (ix[i] + ix[j] + eps)  # Add small value to prevent division by zero
                    else:
                        h += (M[i,j] * ix[j] + M[j,i] * ix[i]) * ix[i] * ix[j]
            h_mix.append(h)
        h_mix = np.array(h_mix)

        if unit == "kJ/mol":
            h_mix *= kJmol  # Convert eV/atom to kJ/mol

        return h_mix
        