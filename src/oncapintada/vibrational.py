# -*- coding: utf-8 -*-
# file: vibrational.py

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
from .constants import R, h, cm_to_THz, eV_to_THz, kB, kJmol


class Vibrational:
    '''Class for vibrational properties of materials.'''
    def __init__(self, vdos: pd.DataFrame, frequency_unit: str = 'cm^-1'):
        self.vdos = vdos
        self.nu = vdos['Frequency'].values        # frequency in cm^-1
        self.g_nu = vdos['VDOS'].values           # vibrational density of states (in states/cm^-1)
        self.frequency_unit = frequency_unit
        if frequency_unit not in ['THz', 'cm^-1', 'eV']:
            raise ValueError("frequency_unit must be either 'THz', 'cm^-1', or 'eV'")
        
        if frequency_unit == 'THz':
            self.nu_THz = np.array(self.nu, dtype=np.float64)     # frequency in THz
            self.g_nu_THz = np.array(self.g_nu, dtype=np.float64)    # VDOS in states/THz
        elif frequency_unit == 'cm^-1':
            self.nu_THz = np.array(self.nu, dtype=np.float64) * cm_to_THz         # convert frequency to THz
            self.g_nu_THz = np.array(self.g_nu, dtype=np.float64) / cm_to_THz     # convert VDOS to states/THz
        elif frequency_unit == 'eV':
            self.nu_THz = np.array(self.nu, dtype=np.float64) * eV_to_THz         # convert frequency to THz
            self.g_nu_THz = np.array(self.g_nu, dtype=np.float64) / eV_to_THz     # convert VDOS to states/THz

        self.vibrational_enthalpy = None
        self.vibrational_entropy = None
        self.vibrational_free_energy = None
    

    def get_vibrational_enthalpy(self, T: float) -> float:
        '''Calculate the vibrational enthalpy at a given temperature.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        
        Returns
        -------
        float
            Vibrational enthalpy in kJ/mol.

        '''
        nu = self.nu_THz
        g_nu = self.g_nu_THz
        x = h * nu / (kB * T)

        # exclude negative frequencies (imaginary modes) from the integration
        mask = nu > 0
        nu = nu[mask]
        g_nu = g_nu[mask]
        x = x[mask]

        enthalpy = 0.5 * h * np.trapz(g_nu * nu, nu) + h * np.trapz(g_nu * nu / (np.exp(x) - 1), nu)
        enthalpy *= kJmol     # convert eV to kJ/mol
        self.vibrational_enthalpy = enthalpy
        return enthalpy
    

    def get_vibrational_entropy(self, T: float) -> float:
        '''Calculate the vibrational entropy at a given temperature.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        
        Returns
        -------
        float
            Vibrational entropy in kJ/(mol*K).
        '''
        nu = self.nu_THz
        g_nu = self.g_nu_THz
        x = h * nu / (kB * T)

        # exclude negative frequencies (imaginary modes) from the integration
        mask = nu > 0
        nu = nu[mask]
        g_nu = g_nu[mask]
        x = x[mask]

        entropy = R * np.trapz(g_nu * (x / (np.exp(x) - 1) - np.log(1 - np.exp(-x))), nu) # in kJ/(mol*K)
        
        self.vibrational_entropy = entropy
        return entropy


    def get_vibrational_free_energy(self, T: float) -> float:
        '''Calculate the vibrational free energy at a given temperature.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        
        Returns
        -------
        float
            Vibrational free energy in kJ/mol.
        '''
        if self.vibrational_enthalpy is None:
            enthalpy = self.get_vibrational_enthalpy(T)
        else:
            enthalpy = self.vibrational_enthalpy
        
        if self.vibrational_entropy is None:
            entropy = self.get_vibrational_entropy(T)
        else:
            entropy = self.vibrational_entropy
        
        free_energy = enthalpy - T * entropy
        
        self.vibrational_free_energy = free_energy
        return free_energy



class VibrationalSubregularModel:
    '''Class for vibrational properties of subregular solutions.'''
    def __init__(self,
                 x: np.ndarray,
                 vdos_A: pd.DataFrame,
                 vdos_B: pd.DataFrame,
                 vdos_AB: pd.DataFrame,
                 vdos_BA: pd.DataFrame,
                 frequency_unit: str = 'cm^-1'):
        self.x = x
        self.vibrational_A = Vibrational(vdos_A, frequency_unit)
        self.vibrational_B = Vibrational(vdos_B, frequency_unit)
        self.vibrational_AB = Vibrational(vdos_AB, frequency_unit)
        self.vibrational_BA = Vibrational(vdos_BA, frequency_unit)