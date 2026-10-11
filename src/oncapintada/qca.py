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

"""Binary pair quasi-chemical approximation in kJ/mol and kelvin."""

import numpy as np
import pandas as pd
from scipy.special import xlogy

from ._validation import positive_integer
from .constants import R


class QCABinary:
    """Pair QCA from a table of mean-field mixing enthalpies.

    Rows are fractions of species A in [0, 1], columns are strictly positive
    temperatures in kelvin, and values are kJ/mol. Pure-component enthalpies
    must be zero. Methods compute prerequisites lazily. Returned pair tables
    are unordered AB, AA, BB probabilities and sum to one.
    """

    def __init__(self, coordination_number=0, enthalpy_df=None):
        if not isinstance(enthalpy_df, pd.DataFrame) or enthalpy_df.empty:
            raise ValueError("A non-empty Enthalpy DataFrame must be provided.")
        self.coordination_number = positive_integer(coordination_number, "Coordination number")
        self.enthalpy_df = enthalpy_df.astype(float).copy()
        self.x_values = np.asarray(enthalpy_df.index, dtype=float)
        self.t_values = np.asarray(enthalpy_df.columns, dtype=float)
        if not np.isfinite(self.enthalpy_df.values).all():
            raise ValueError("Enthalpy data must contain finite values.")
        if not np.isfinite(self.x_values).all() or np.any(
            (self.x_values < 0) | (self.x_values > 1)
        ):
            raise ValueError("Compositions must be finite fractions in [0, 1].")
        if len(np.unique(self.x_values)) != len(self.x_values):
            raise ValueError("Compositions must be unique.")
        if not np.isfinite(self.t_values).all() or np.any(self.t_values <= 0):
            raise ValueError("Temperatures must be finite and positive.")
        endpoints = (self.x_values == 0) | (self.x_values == 1)
        if not np.allclose(self.enthalpy_df.values[endpoints], 0, rtol=0, atol=1e-12):
            raise ValueError("Pure-component mixing enthalpy must be zero.")
        self.enthalpy_df.index = self.x_values
        self.enthalpy_df.columns = self.t_values
        self.omega = self.gamma = self.probability = None
        self.warren_cowley_parameters = None
        self.enthalpy_of_mixing = self.entropy_of_mixing = None
        self.gibbs_free_energy_of_mixing = None

    def _table(self, values):
        return pd.DataFrame(values, index=self.x_values, columns=self.t_values)

    def get_omega(self):
        """H/[x(1-x)]; the unobservable pure-endpoint value is defined as zero."""
        q = (self.x_values * (1 - self.x_values))[:, None]
        values = np.divide(
            self.enthalpy_df.values, q, out=np.zeros_like(self.enthalpy_df.values), where=q > 0
        )
        self.omega = self._table(values)
        return self.omega

    def get_gamma(self):
        """Return exp(-2ω/zRT), allowing limiting zero/infinite gamma values."""
        if self.omega is None:
            self.get_omega()
        with np.errstate(over="ignore", under="ignore"):
            self._log_gamma = (
                -2 * self.omega.values / (self.coordination_number * R * self.t_values)
            )
            self.gamma = self._table(np.exp(self._log_gamma))
        return self.gamma

    def get_probability(self):
        """Stable QCA probabilities, including γ=1 and strong ordering limits."""
        if self.gamma is None:
            self.get_gamma()
        x = self.x_values[:, None]
        q = x * (1 - x)
        with np.errstate(under="ignore"):
            sqrt_gamma = np.exp(np.minimum(self._log_gamma, 0) / 2)
            inverse_gamma = np.exp(-np.maximum(self._log_gamma, 0))
        gamma_small = sqrt_gamma**2
        denominator = sqrt_gamma + np.sqrt(gamma_small + 4 * (1 - gamma_small) * q)
        p_small = np.divide(
            4 * sqrt_gamma * q, denominator, out=np.zeros_like(denominator), where=denominator > 0
        )
        p_large = 4 * q / (1 + np.sqrt(np.maximum(1 - 4 * q + 4 * q * inverse_gamma, 0)))
        p_ab = np.where(self._log_gamma <= 0, p_small, p_large)
        p_aa = np.maximum(x - p_ab / 2, 0)
        p_bb = np.maximum(1 - x - p_ab / 2, 0)
        self.probability = tuple(self._table(p) for p in (p_ab, p_aa, p_bb))
        return self.probability

    def get_warren_cowley_parameters(self):
        """Pair SRO relative to random mixing; absent-species entries are NaN."""
        if self.probability is None:
            self.get_probability()
        x = self.x_values[:, None]
        expected = (2 * x * (1 - x), x**2, (1 - x) ** 2)
        values = []
        for probability, reference in zip(self.probability, expected):
            ratio = np.divide(
                probability.values,
                reference,
                out=np.full_like(probability.values, np.nan),
                where=reference > 0,
            )
            values.append(self._table(1 - ratio))
        self.warren_cowley_parameters = tuple(values)
        return self.warren_cowley_parameters

    def get_enthalpy_of_mixing(self):
        if self.probability is None:
            self.get_probability()
        self.enthalpy_of_mixing = self._table(self.omega.values * self.probability[0].values / 2)
        return self.enthalpy_of_mixing

    def get_entropy_of_mixing(self):
        if self.probability is None:
            self.get_probability()
        z = self.coordination_number
        x = self.x_values[:, None]
        ab, aa, bb = (p.values for p in self.probability)
        pair_entropy = -z * R / 2 * (xlogy(aa, aa) + xlogy(bb, bb) + xlogy(ab, ab / 2))
        correction = (z - 1) * R * (xlogy(x, x) + xlogy(1 - x, 1 - x))
        self.entropy_of_mixing = self._table(pair_entropy + correction)
        return self.entropy_of_mixing

    def get_gibbs_free_energy_of_mixing(self):
        h = self.get_enthalpy_of_mixing().values
        s = self.get_entropy_of_mixing().values
        self.gibbs_free_energy_of_mixing = self._table(h - self.t_values * s)
        return self.gibbs_free_energy_of_mixing
