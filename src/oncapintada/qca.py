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
        
        self.omega = None
        self.gamma = None
        self.probability = None
        self.warren_cowley_parameters = None
        self.enthalpy_of_mixing = None
        self.entropy_of_mixing = None
        self.gibbs_free_energy_of_mixing = None
        
        
    def get_omega(self) -> pd.DataFrame:
        '''
        Calculate the interaction parameter ⍵(x,t) from the enthalpy of mixing data.
        '''
        x = self.x_values
        t = self.t_values
        h = self.enthalpy_df.values

        # shift x=0 and x=1 to avoid division by zero in omega calculation
        eps = 1e-8
        x = np.clip(x, eps, 1 - eps)
        
        omega = np.zeros((len(x), len(t)))
        for ix in range(len(x)):
            for iT in range(len(t)):
                omega[ix, iT] = h[ix, iT] / (x[ix] * (1 - x[ix]))

        omega_df = pd.DataFrame(omega, index=x, columns=t)

        self.omega = omega_df
        return omega_df


    def get_gamma(self) -> pd.DataFrame:
        ''''
        Calculate the parameter γ(x,t) using the QCA.
        '''
        z = self.coordination_number
        R = 8.314 / 1000    # kJ/(mol*K)

        x = self.x_values
        t = self.t_values
        if self.omega is None:
            self.get_omega()
        omega = self.omega.values
        
        gamma = np.zeros((len(x), len(t)))
        for ix in range(len(x)):
            for iT in range(len(t)):
                gamma[ix, iT] = np.exp(-2*omega[ix, iT]/(z*R*t[iT]))

        gamma_df = pd.DataFrame(gamma, index=x, columns=t)
        self.gamma = gamma_df
        return gamma_df


    def get_probability(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ''''
        Calculate the probabilities of finding AB, AA, and BB pairs in the alloy using the QCA.
        '''
        x = self.x_values
        t = self.t_values
        if self.gamma is None:
            self.get_gamma()
        g = self.gamma.values

        p_ab = np.zeros((len(x), len(t)))
        p_aa = np.zeros((len(x), len(t)))
        p_bb = np.zeros((len(x), len(t)))

        for ix in range(len(x)):
            for iT in range(len(t)):
                g_val = g[ix, iT]
                p_ab[ix, iT] = ( -g_val + np.sqrt(g_val**2 + 4*g_val*(1-g_val)*x[ix]*(1-x[ix])) ) / ( 1-g_val )
                p_aa[ix, iT] = x[ix] - 0.5 * p_ab[ix, iT]
                p_bb[ix, iT] = (1-x[ix]) - 0.5 * p_ab[ix, iT]

        p_ab_df = pd.DataFrame(p_ab, index=x, columns=t)
        p_aa_df = pd.DataFrame(p_aa, index=x, columns=t)
        p_bb_df = pd.DataFrame(p_bb, index=x, columns=t)
        self.probability = (p_ab_df, p_aa_df, p_bb_df)
        return p_ab_df, p_aa_df, p_bb_df
    

    def get_warren_cowley_parameters(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Calculate the Warren-Cowley short-range order parameters for AB, AA, and BB pairs.
        '''
        x = self.x_values
        t = self.t_values
        if self.probability is None:
            self.get_probability()
        p_ab = self.probability[0].values
        p_aa = self.probability[1].values
        p_bb = self.probability[2].values

        alpha_ab = np.zeros((len(x), len(t)))
        alpha_aa = np.zeros((len(x), len(t)))
        alpha_bb = np.zeros((len(x), len(t)))

        for ix in range(len(x)):
            for iT in range(len(t)):
                alpha_ab[ix, iT] = 1 - p_ab[ix, iT] / (2 * x[ix] * (1-x[ix]))
                alpha_aa[ix, iT] = 1 - p_aa[ix, iT] / (x[ix]**2)
                alpha_bb[ix, iT] = 1 - p_bb[ix, iT] / ((1-x[ix])**2)

        alpha_ab_df = pd.DataFrame(alpha_ab, index=x, columns=t)
        alpha_aa_df = pd.DataFrame(alpha_aa, index=x, columns=t)
        alpha_bb_df = pd.DataFrame(alpha_bb, index=x, columns=t)

        self.warren_cowley_parameters = (alpha_ab_df, alpha_aa_df, alpha_bb_df)
        return alpha_ab_df, alpha_aa_df, alpha_bb_df


    def get_enthalpy_of_mixing(self) -> pd.DataFrame:
        ''''
        Calculate the enthalpy of mixing using the QCA.
        '''
        x = self.x_values
        t = self.t_values
        omega = self.omega.values
        p_ab = self.probability[0].values
        h_qca = np.zeros((len(x), len(t)))
        for ix in range(len(x)):
            for iT in range(len(t)):
                h_qca[ix, iT] = omega[ix, iT] * p_ab[ix, iT] / 2

        h_qca_df = pd.DataFrame(h_qca, index=x, columns=t)
        self.enthalpy_of_mixing = h_qca_df
        return h_qca_df


    def get_entropy_of_mixing(self) -> pd.DataFrame:
        ''''
        Calculate the entropy of mixing using the QCA.
        '''
        R = 8.314 / 1000    # kJ/(mol*K)
        eps = 1e-12  # small value to avoid log(0)

        z = self.coordination_number
        x = self.x_values
        t = self.t_values
        if self.probability is None:
            self.get_probability()
        p_ab = self.probability[0].values
        p_aa = self.probability[1].values
        p_bb = self.probability[2].values

        S_BP = np.zeros((len(x), len(t)))
        S_corr = np.zeros((len(x), len(t)))
        for ix in range(len(x)):
            for iT in range(len(t)):
                # Bethe-Peierls entropy
                S_BP[ix, iT] = -0.5 * z * R * ( p_aa[ix, iT] * np.log(p_aa[ix, iT]+eps) + p_bb[ix, iT] * np.log(p_bb[ix, iT]+eps) + p_ab[ix, iT] * np.log(p_ab[ix, iT]/2+eps) )
                # Guggenheim correction
                S_corr[ix, iT] = (z-1) * R * ( x[ix]*np.log(x[ix]+eps) + (1-x[ix])*np.log(1-x[ix]+eps) )

        S_qca = S_BP + S_corr
        S_qca_df = pd.DataFrame(S_qca, index=x, columns=t)
        self.entropy_of_mixing = S_qca_df
        return S_qca_df


    def get_gibbs_free_energy_of_mixing(self) -> pd.DataFrame:
        ''''
        Calculate the Gibbs free energy of mixing using the QCA.
        '''
        x = self.x_values
        t = self.t_values
        
        if self.enthalpy_of_mixing is None:
            self.get_enthalpy_of_mixing()
        if self.entropy_of_mixing is None:
            self.get_entropy_of_mixing()
        
        h_qca = self.enthalpy_of_mixing.values
        s_qca = self.entropy_of_mixing.values

        g_qca = np.zeros((len(x), len(t)))
        for ix in range(len(x)):
            for iT in range(len(t)):
                g_qca[ix, iT] = h_qca[ix, iT] - t[iT] * s_qca[ix, iT]

        g_qca_df = pd.DataFrame(g_qca, index=x, columns=t)
        self.gibbs_free_energy_of_mixing = g_qca_df
        return g_qca_df
