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

from typing import Optional

import numpy as np
import pandas as pd
from ase import Atoms
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from ase.phonons import Phonons
from scipy.integrate import trapezoid

from .constants import R, cm_to_THz, eV_to_THz, h, kB, kJmol


class Vibrational:
    """Class for vibrational properties of materials."""

    def __init__(self, vdos: Optional[pd.DataFrame] = None, frequency_unit: str = "cm^-1"):
        if frequency_unit not in ("THz", "cm^-1", "eV"):
            raise ValueError("frequency_unit must be either 'THz', 'cm^-1', or 'eV'")
        self.frequency_unit = frequency_unit
        self.vdos = self.nu = self.g_nu = self.nu_THz = self.g_nu_THz = None
        self.number_of_modes = None
        self.vibrational_enthalpy = self.vibrational_entropy = self.vibrational_free_energy = None
        if vdos is not None:
            self._set_vdos(vdos)

    def _set_vdos(self, vdos):
        if not isinstance(vdos, pd.DataFrame) or not {"Frequency", "VDOS"}.issubset(vdos.columns):
            raise ValueError("VDOS must be a DataFrame with Frequency and VDOS columns.")
        data = vdos[["Frequency", "VDOS"]].astype(float).copy()
        nu, density = data["Frequency"].to_numpy(), data["VDOS"].to_numpy()
        if len(nu) < 2 or not np.isfinite(data.values).all() or np.any(np.diff(nu) <= 0):
            raise ValueError("VDOS requires at least two finite, strictly increasing frequencies.")
        if np.any(density < 0):
            raise ValueError("VDOS density must be non-negative.")
        self.vdos, self.nu, self.g_nu = data, nu, density
        scale = {"THz": 1.0, "cm^-1": cm_to_THz, "eV": eV_to_THz}[self.frequency_unit]
        self.nu_THz, self.g_nu_THz = nu * scale, density / scale
        self.number_of_modes = self.get_number_of_modes()
        self.vibrational_enthalpy = self.vibrational_entropy = self.vibrational_free_energy = None

    def get_vdos(
        self,
        atoms: Atoms,
        calc,
        supercell: tuple = (2, 2, 2),
        kpts: tuple = (100, 100, 1),
        npts: int = 3000,
        width: float = 5e-4,
        xmin: float = 1e-4,
        xmax: float = 0.10,
        delta: float = 0.01,
        fmax: float = 0.001,
    ) -> pd.DataFrame:
        """Calculate the vibrational density of states (VDOS) using ASE's Phonons class.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atomic structure for which to calculate the phonons.
        calc : ASE Calculator
            The calculator to use for force calculations.
        supercell : tuple, optional
            The size of the supercell for phonon calculations (default is (2,2,2)).
        kpts : tuple, optional
            The k-point grid for phonon calculations (default is (100,100,1)).
        npts : int, optional
            The number of points in the VDOS (default is 3000).
        width : float, optional
            The width of the Gaussian smearing for the VDOS (default is 5e-4).
        xmin : float, optional
            The minimum frequency for the VDOS (default is 1e-4).
        xmax : float, optional
            The maximum frequency for the VDOS (default is 0.10).
        delta : float, optional
            Displacement for finite difference calculations (default is 0.01).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the frequencies and corresponding VDOS values.
        """
        atoms = atoms.copy()
        atoms.calc = calc
        optim = BFGS(UnitCellFilter(atoms), logfile=None)
        optim.run(fmax=fmax)
        ph = Phonons(atoms, calc, supercell=supercell, delta=delta)
        ph.run()
        ph.read(acoustic=True)
        ph.clean()
        ph_dos = ph.get_dos(kpts=kpts).sample_grid(npts=npts, width=width, xmin=xmin, xmax=xmax)

        # ASE samples phonon energies in eV; convert both axes together.
        scale = {"eV": 1.0, "THz": eV_to_THz, "cm^-1": eV_to_THz / cm_to_THz}[self.frequency_unit]
        self._set_vdos(
            pd.DataFrame(
                {
                    "Frequency": ph_dos.get_energies() * scale,
                    "VDOS": ph_dos.get_weights() / scale,
                }
            )
        )
        return self.vdos.copy()

    def get_number_of_modes(self) -> float:
        """Integral of the supplied DOS; evaluation requires a DOS to be set."""
        if self.vdos is None:
            raise ValueError("VDOS is not set; supply a table or call get_vdos first.")
        return float(trapezoid(self.g_nu, self.nu))

    def _thermal_terms(self, T):
        if self.vdos is None:
            raise ValueError("VDOS is not set; supply a table or call get_vdos first.")
        temperature = np.asarray(T, dtype=float)
        if temperature.ndim > 1 or not np.isfinite(temperature).all() or np.any(temperature < 0):
            raise ValueError("Temperature must be a finite, non-negative scalar or 1D array.")
        was_scalar = temperature.ndim == 0
        temperature = np.atleast_1d(temperature)
        # Imaginary and zero modes are excluded from harmonic thermodynamics.
        mask = self.nu_THz > 0
        nu, density = self.nu_THz[mask], self.g_nu_THz[mask]
        energy = h * nu[:, None]
        with np.errstate(over="ignore", under="ignore"):
            x = np.divide(
                energy,
                kB * temperature[None, :],
                out=np.full((len(nu), len(temperature)), np.inf),
                where=temperature[None, :] > 0,
            )
            denominator = -np.expm1(-x)
            occupation = np.exp(-x) / denominator
            x_occupation = np.multiply(x, occupation, out=np.zeros_like(x), where=np.isfinite(x))
        h_vib = trapezoid(density[:, None] * energy * (0.5 + occupation), nu, axis=0) * kJmol
        s_vib = R * trapezoid(density[:, None] * (x_occupation - np.log(denominator)), nu, axis=0)
        if was_scalar:
            return h_vib[0], s_vib[0]
        return h_vib, s_vib

    def get_vibrational_enthalpy(self, T):
        """Harmonic internal energy in kJ/mol, including zero-point energy.

        The DOS integral determines the mode count; values are per mole of the
        entity represented by the DOS (per atom only for a per-atom DOS).
        """
        self.vibrational_enthalpy = self._thermal_terms(T)[0]
        return self.vibrational_enthalpy

    def get_vibrational_entropy(self, T):
        """Harmonic entropy in kJ/(mol K); exactly zero at zero temperature."""
        self.vibrational_entropy = self._thermal_terms(T)[1]
        return self.vibrational_entropy

    def get_vibrational_free_energy(self, T):
        """H - TS, recomputed for the requested temperatures."""
        enthalpy, entropy = self._thermal_terms(T)
        self.vibrational_enthalpy, self.vibrational_entropy = enthalpy, entropy
        self.vibrational_free_energy = enthalpy - np.asarray(T) * entropy
        return self.vibrational_free_energy


class VibrationalSubregularModel:
    """Class for vibrational properties of subregular solutions."""

    def __init__(
        self,
        x: np.ndarray,
        vdos_AA: pd.DataFrame,
        vdos_BB: pd.DataFrame,
        vdos_AB: pd.DataFrame,
        vdos_BA: pd.DataFrame,
        frequency_unit: str = "cm^-1",
    ):
        self.x = x
        self.vibrational_AA = Vibrational(vdos_AA, frequency_unit)
        self.vibrational_BB = Vibrational(vdos_BB, frequency_unit)
        self.vibrational_AB = Vibrational(vdos_AB, frequency_unit)
        self.vibrational_BA = Vibrational(vdos_BA, frequency_unit)
