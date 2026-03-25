# -*- coding: utf-8 -*-
# file: thermodynamics.py

# This code is part of Onça-pintada.
# MIT License
#
# Copyright (c) 2025 Leandro Seixas Rocha <leandro.seixas@proton.me> 
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

class PhaseDiagram:
    """
    Class to work with tabulated Gibbs free energy G(x,T) stored in a pandas DataFrame, and calculate spinodal and binodal curves.

    - Rows (index) are compositions x.
    - Columns are temperatures T.
    - Values are G(x, T).

    Main methods:
    - dGdx()                -> returns a DataFrame with the first derivative (∂G/∂x).
    - d2Gdx2()              -> returns a DataFrame with the second derivative (∂²G/∂x²).
    - spinodal_curve()      -> returns spinodal curve(s) x(T) from sign changes in ∂²G/∂x².
    - binodal_curve()       -> returns binodal curve(s) x(T) from common tangent construction on G(x, T).
    """

    def __init__(self, gibbs_df: pd.DataFrame, x_values=None, temperatures=None, atoms: Atoms | None = None):
        """
        Parameters
        ----------
        gibbs_df : pd.DataFrame
            DataFrame with G(x, T):
            - index: compositions x (float or convertible to float).
            - columns: temperatures T (float or int).
            - values: Gibbs free energy G(x, T).
        x_values : array-like, optional
            Values of x corresponding to the index.
            If None, attempts to convert the index labels to float.
        temperatures : array-like, optional
            Values of temperatures corresponding to the columns.
            If None, attempts to convert the column labels to float.
        atoms : ase.Atoms, optional
            Atoms object associated with the thermodynamic data (optional).
        """
        # Ensure numeric values
        self.gibbs = gibbs_df.astype(float).copy()

        # Define composition axis x
        if x_values is None:
            self.x = np.array(self.gibbs.index, dtype=float)
            self.gibbs.index = self.x
        else:
            self.x = np.array(x_values, dtype=float)
            if len(self.x) != self.gibbs.shape[0]:
                raise ValueError(
                    "x_values must have the same length as the number of rows in gibbs_df."
                )
            self.gibbs.index = self.x

        # Define temperature axis T
        if temperatures is None:
            self.T = np.array(self.gibbs.columns, dtype=float)
            self.gibbs.columns = self.T
        else:
            self.T = np.array(temperatures, dtype=float)
            if len(self.T) != self.gibbs.shape[1]:
                raise ValueError(
                    "temperatures must have the same length as the number of columns in gibbs_df."
                )
            self.gibbs.columns = self.T

        # Optional Atoms object
        self.atoms = atoms

    # ------------------------------------------------------------------
    # First derivative ∂G/∂x
    # ------------------------------------------------------------------
    def dGdx(self) -> pd.DataFrame:
        """
        Compute the first derivative ∂G/∂x using np.gradient along the x axis.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same index (x) and columns (T),
            containing (∂G/∂x)(x, T).
        """
        G = self.gibbs.values  # shape: (nX, nT)
        dGdx_array = np.gradient(G, self.x, axis=0, edge_order=2)

        dGdx_df = pd.DataFrame(dGdx_array, index=self.gibbs.index, columns=self.gibbs.columns)
        return dGdx_df

    # ------------------------------------------------------------------
    # Second derivative ∂²G/∂x²
    # ------------------------------------------------------------------
    def d2Gdx2(self) -> pd.DataFrame:
        """
        Compute the second derivative ∂²G/∂x² from G(x, T)
        using two calls to np.gradient along x.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same index (x) and columns (T),
            containing (∂²G/∂x²)(x, T).
        """
        G = self.gibbs.values  # (nX, nT)

        # First derivative
        dGdx = np.gradient(G, self.x, axis=0, edge_order=2)
        # Second derivative
        d2Gdx2_array = np.gradient(dGdx, self.x, axis=0, edge_order=2)

        d2Gdx2_df = pd.DataFrame(d2Gdx2_array, index=self.gibbs.index, columns=self.gibbs.columns)
        return d2Gdx2_df

    # ------------------------------------------------------------------
    # Spinodal curve (sign change in second derivative)
    # ------------------------------------------------------------------
    def spinodal_curve(self, atol: float = 1e-8) -> pd.DataFrame:
        """
        Find, for each T, the composition(s) x where the second derivative ∂²G/∂x²
        changes sign (spinodal boundaries).

        Algorithm
        ---------
        - Compute ∂²G/∂x² on the (x, T) grid.
        - For each row (fixed T):
            * Scan adjacent points in x: (x_j, x_{j+1}).
            * Look for sign changes in d2Gdx2(T, x_j) and d2Gdx2(T, x_{j+1}),
              i.e., y1 * y2 < 0.
            * Approximate the zero by linear interpolation between x_j and x_{j+1}.
            * Also treat values with |y| < atol as "numerical zeros"
              and include their corresponding x.
        - For each T, keep up to two distinct x values (x1, x2), sorted.

        Parameters
        ----------
        atol : float, optional
            Absolute tolerance to treat small values of ∂²G/∂x² as zero.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns "x" and "t", containing the spinodal points (x, T).
        """
        d2_df = self.d2Gdx2()
        d2 = d2_df.values  # shape: (nX, nT)
        x = self.x
        nX, nT = d2.shape

        spinodal_points = []

        for iT in range(nT):
            row = d2[:, iT]
            xs_found = []

            for j in range(nX - 1):
                y1 = row[j]
                y2 = row[j + 1]

                if np.isnan(y1) or np.isnan(y2):
                    continue

                # Treat near-zero values as zeros
                if abs(y1) < atol:
                    xs_found.append(x[j])
                if abs(y2) < atol:
                    xs_found.append(x[j + 1])

                # Check sign change (excluding exact/near zeros already handled)
                if y1 * y2 < 0.0:
                    # Linear interpolation for zero crossing between x[j] and x[j+1]
                    # y(x) = y1 + (y2 - y1)*t, y(x_zero) = 0 => t = -y1 / (y2 - y1)
                    t = -y1 / (y2 - y1)
                    x_zero = x[j] + t * (x[j + 1] - x[j])
                    xs_found.append(x_zero)

            # Remove approximate duplicates and sort
            xs_unique = []
            for xv in xs_found:
                if not any(np.isclose(xv, y, atol=1e-10) for y in xs_unique):
                    xs_unique.append(xv)
            xs_unique = sorted(xs_unique)

            if len(xs_unique) >= 1:
                x1 = xs_unique[0]
            else:
                x1 = np.nan

            if len(xs_unique) >= 2:
                x2 = xs_unique[1]
            else:
                x2 = np.nan

            spinodal_points.append((x1, x2))

        spinodal_join = []
        for iT, (x1, x2) in enumerate(spinodal_points):
            if not np.isnan(x1):
                spinodal_join.append((x1, self.T[iT]))
            if not np.isnan(x2):
                spinodal_join.append((x2, self.T[iT]))

        spinodal_join_df = pd.DataFrame(spinodal_join, columns=["x", "t"])
        # spinodal_join_df.set_index("t", inplace=True)
        spinodal_join_df.sort_values(by='x', inplace=True)

        return spinodal_join_df

        # spinodal_df = pd.DataFrame(spinodal_points, index=self.gibbs.index, columns=["x1", "x2"])
        # return spinodal_df


    # ------------------------------------------------------------------
    # Binodal curve
    # ------------------------------------------------------------------
    def binodal_curve(self, atol: float = 1e-8) -> pd.DataFrame:
        """
        Compute the binodal curve (coexistence curve) from G(x, T) using the common tangent construction.

        Algorithm
        ---------
        - For each T, scan pairs of compositions (x_i, x_j) with i < j.
        - Check if the line connecting (x_i, G_i) and (x_j, G_j) is a common tangent:
        * Slope m = (G_j - G_i) / (x_j - x_i)
        * Check if m is less than or equal to the local slope at x_i: m <= dGdx(T, x_i)
        * Check if m is greater than or equal to the local slope at x_j: m >= dGdx(T, x_j)
        * Also check if the line is below G(x) for all x in between (convexity condition).
        - Keep track of the pair (x_i, x_j) that satisfies these conditions and has the largest gap (x_j - x_i).
        - Return the pairs (x_i, x_j) for each T as the binodal points.

        Parameters
        ----------
        atol : float, optional
            Absolute tolerance for numerical comparisons. Default is 1e-8.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns "x" and "t", containing the binodal points (x, T).
            Each T may have up to two x values corresponding to the two sides of the coexistence curve.
        """

        G = self.gibbs.values  # shape: (nX, nT)
        x = self.x
        T = self.T
        dGdx_df = self.dGdx()
        dGdx = dGdx_df.values  # shape: (nX, nT)

        nX, nT = G.shape
        binodal_points = []

        for iT in range(nT):
            best_pair = (np.nan, np.nan)
            best_gap = -np.inf

            for i in range(nX):
                for j in range(i + 1, nX):
                    x_i, x_j = x[i], x[j]
                    G_i, G_j = G[i, iT], G[j, iT]
                    m = (G_j - G_i) / (x_j - x_i)

                    if m <= dGdx[i, iT] + atol and m >= dGdx[j, iT] - atol:
                        # Check convexity condition: line must be below G(x) for all x in (x_i, x_j)
                        x_mid = np.linspace(x_i, x_j, 10)
                        G_mid_line = G_i + m * (x_mid - x_i)
                        G_mid_actual = np.interp(x_mid, x, G[:, iT])

                        if np.all(G_mid_line <= G_mid_actual + atol):
                            gap = x_j - x_i
                            if gap > best_gap:
                                best_gap = gap
                                best_pair = (x_i, x_j)

            binodal_points.append(best_pair)

        # concatenate (T,x1) and (T, x2) in a single (T, x) DataFrame
        binodal_join = []
        for iT, (x1, x2) in enumerate(binodal_points):
            if not np.isnan(x1):
                binodal_join.append((x1, T[iT]))
            if not np.isnan(x2):
                binodal_join.append((x2, T[iT]))


        binodal_join_df = pd.DataFrame(binodal_join, columns=["x", "t"], index=True)
        # binodal_join_df.set_index("t", inplace=True)
        binodal_join_df.sort_values(by='x', inplace=True)

        return binodal_join_df