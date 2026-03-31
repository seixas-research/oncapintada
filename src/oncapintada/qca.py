# -*- coding: utf-8 -*-
# file: qca.py

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
from typing import Optional

class QCABinary:
    '''Quasichemical Approximation (QCA) for thermodynamic modeling of binary alloys.

    Parameters
    ----------
    coordination_number : int
        The coordination number of the alloy (number of nearest neighbors).
    enthalpy_df : pd.DataFrame
        A DataFrame containing the enthalpy of mixing values for different compositions and temperatures.
        The index of the DataFrame should be the composition (x) and the columns should be the temperatures (t).
    '''
    def __init__(self, coordination_number: int = 0, enthalpy_df: Optional[pd.DataFrame] = None):
        if enthalpy_df is None:
            raise ValueError("Enthalpy DataFrame must be provided.")
        
        self.enthalpy_df = enthalpy_df
        self.x_values = self.enthalpy_df.index.values
        self.t_values = self.enthalpy_df.columns.values

        if coordination_number <= 0:
            raise ValueError("Coordination number must be a positive integer.")
        self.coordination_number = coordination_number
        
        if self.enthalpy_df.isnull().values.any():
            raise ValueError("Enthalpy data contains NaN values.")
        
        
    def omega(self) -> pd.DataFrame:
        '''
        Calculate the interaction parameter ⍵(x,t) from the enthalpy of mixing data.
        '''
        x = self.x_values
        t = self.t_values
        h = self.enthalpy_df.values

        # shift x=0 and x=1 to avoid division by zero in omega calculation
        eps = 1e-8
        x = np.clip(x, eps, 1 - eps)

        # enthalpy_df is a matrix of shape (len(x), len(t)) containing the enthalpy of mixing values for each composition and temperature.
        for iT in range(len(t)):
            omega = h[:, iT] / (x * (1 - x))
            if np.any(np.isnan(omega)):
                raise ValueError(f"NaN values found in omega calculation for t={t[iT]} K. Check enthalpy data and composition range.")
            if np.any(np.isinf(omega)):
                raise ValueError(f"Inf values found in omega calculation for t={t[iT]} K. Check enthalpy data and composition range.")
        
        # omega = h / (x * (1 - x))[:, np.newaxis]
        omega_df = pd.DataFrame(omega, index=x, columns=t)
        return omega_df


    def gamma(self) -> pd.DataFrame:
        ''''
        Calculate the parameter γ(x,t) using the QCA.
        '''
        z = self.coordination_number
        R = 8.314 / 1000    # kJ/(mol*K)
        t = self.t_values
        omega = self.omega().values
        
        gamma = np.exp(-2*omega/(z*R*t))
        gamma_df = pd.DataFrame(gamma, index=self.x_values, columns=self.t_values)
        return gamma_df


    def probability(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ''''
        Calculate the probabilities of finding AB, AA, and BB pairs in the alloy using the QCA.
        '''
        g = self.gamma().values
        x = self.x_values
        t = self.t_values

        p_ab = ( -g +np.sqrt(g**2 + 4*g*(1-g)*x*(1-x)) ) / ( 1-g )
        p_aa = x - 0.5 * p_ab
        p_bb = (1-x) - 0.5 * p_ab

        p_ab_df = pd.DataFrame(p_ab, index=x, columns=t)
        p_aa_df = pd.DataFrame(p_aa, index=x, columns=t)
        p_bb_df = pd.DataFrame(p_bb, index=x, columns=t)
        return p_ab_df, p_aa_df, p_bb_df
    

    def warren_cowley_parameters(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Calculate the Warren-Cowley short-range order parameters for AB, AA, and BB pairs.
        '''
        p_ab_df, p_aa_df, p_bb_df = self.probability().values
        x = self.x_values
        t = self.t_values

        alpha_ab = 1 - p_ab_df / (2 * x * (1-x))
        alpha_aa = 1 - p_aa_df / (x**2)
        alpha_bb = 1 - p_bb_df / ((1-x)**2)

        alpha_ab_df = pd.DataFrame(alpha_ab, index=x, columns=t)
        alpha_aa_df = pd.DataFrame(alpha_aa, index=x, columns=t)
        alpha_bb_df = pd.DataFrame(alpha_bb, index=x, columns=t)
        return alpha_ab_df, alpha_aa_df, alpha_bb_df


    def get_enthalpy_of_mixing(self) -> pd.DataFrame:
        ''''
        Calculate the enthalpy of mixing using the QCA.
        '''
        omega = self.omega().values
        p_ab, _, _ = self.probability().values
        h_qca = omega * p_ab/2
        h_qca_df = pd.DataFrame(h_qca, index=self.x_values, columns=self.t_values)
        return h_qca_df


    def get_entropy_of_mixing(self) -> pd.DataFrame:
        ''''
        Calculate the entropy of mixing using the QCA.
        '''
        R = 8.314 / 1000    # kJ/(mol*K)
        p_ab, p_aa, p_bb = self.probability().values
        z = self.coordination_number
        eps = 1e-12  # small value to avoid log(0)

        S_BP = -0.5 * z * ( p_aa * np.log(p_aa+eps) + p_bb * np.log(p_bb+eps) + p_ab * np.log(p_ab/2+eps) )          # Bethe-Peierls entropy.
        S_corr = (z-1) * ( self.x_values*np.log(self.x_values+eps) + (1-self.x_values)*np.log(1-self.x_values+eps) ) # Guggenheim correction.
        s_qca = S_BP + S_corr
        s_qca_df = pd.DataFrame(s_qca, index=self.x_values, columns=self.t_values)
        return s_qca_df


    def get_gibbs_free_energy_of_mixing(self) -> pd.DataFrame:
        ''''
        Calculate the Gibbs free energy of mixing using the QCA.
        '''
        h_qca = self.get_enthalpy_of_mixing().values
        s_qca = self.get_entropy_of_mixing().values
        t = self.t_values

        g_qca = h_qca - t * s_qca
        g_qca_df = pd.DataFrame(g_qca, index=self.x_values, columns=self.t_values)
        return g_qca_df
