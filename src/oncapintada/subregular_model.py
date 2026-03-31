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
from typing import Optional, List


# To calculate the enthalpy of mixing based on DSI model.
class BinaryAlloy:
    def __init__(self, energy_matrix: Optional[np.ndarray] = None, dilution: float = 0.0):
        self.energy_matrix = energy_matrix
        self.dilution = dilution
        if self.dilution < 1e-8:
            raise ValueError("Set the dilution parameter.")
        
    def get_energy_matrix(self) -> np.ndarray:
        return self.energy_matrix
    
    def set_energy_matrix(self, energy_matrix: Optional[np.ndarray] = None):
        self.energy_matrix = energy_matrix

    # def get_Mij_deprecated(self):
    #     M = np.zeros_like(self.energy_matrix)
    #     x0 = self.dilution
    #     E = self.energy_matrix
    #     for i in range(E.shape[0]):
    #         for j in range(E.shape[1]):
    #             M[i, j] = E[i, j]  - ( x0 * E[i, i] + (1-x0) * E[j, j] )
    #     return M
    
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
        if self.dilution < 1e-8:
            raise ValueError("Set the dilution parameter.")
