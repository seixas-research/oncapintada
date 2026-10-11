# -*- coding: utf-8 -*-
# file: phase_diagram.py

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

from ._validation import positive_integer


class PhaseDiagram:
    """
    Class to work with tabulated Gibbs free energy G(x,t) stored in a pandas DataFrame, and calculate spinodal and binodal curves.

    - Rows (index) are compositions x.
    - Columns are temperatures t.
    - Values are G(x, t).

    Main methods:
    - dGdx()                -> returns a DataFrame with the first derivative (∂G/∂x).
    - d2Gdx2()              -> returns a DataFrame with the second derivative (∂²G/∂x²).
    - spinodal_curve()      -> returns spinodal curve from sign changes in ∂²G/∂x².
    - binodal_curve()       -> returns binodal curve from common tangent construction on G(x, t).
    """

    def __init__(self, gibbs_df: pd.DataFrame, x_values=None, t_values=None):
        """
        Parameters
        ----------
        gibbs_df : pd.DataFrame
            DataFrame with G(x, t):
            - index: compositions x (float or convertible to float).
            - columns: temperatures t (float or int).
            - values: Gibbs free energy G(x, t).
        x_values : array-like, optional
            Values of x corresponding to the index.
            If None, attempts to convert the index labels to float.
        t_values : array-like, optional
            Values of temperatures corresponding to the columns.
            If None, attempts to convert the column labels to float.
        """

        if not isinstance(gibbs_df, pd.DataFrame):
            raise ValueError("gibbs_df must be a pandas DataFrame.")

        if gibbs_df.empty:
            raise ValueError("gibbs_df cannot be empty.")

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

        # Define temperature axis t
        if t_values is None:
            self.t = np.array(self.gibbs.columns, dtype=float)
            self.gibbs.columns = self.t
        else:
            self.t = np.array(t_values, dtype=float)
            if len(self.t) != self.gibbs.shape[1]:
                raise ValueError(
                    "t_values must have the same length as the number of columns in gibbs_df."
                )
            self.gibbs.columns = self.t
        self._validate_axes()

    def _validate_axes(self):
        if self.x.ndim != 1 or len(self.x) < 3:
            raise ValueError("At least three compositions are required.")
        if not np.isfinite(self.x).all() or np.any(np.diff(self.x) <= 0):
            raise ValueError("Compositions must be finite and strictly increasing.")
        if np.any((self.x < 0) | (self.x > 1)):
            raise ValueError("Compositions must lie in [0, 1].")
        if self.t.ndim != 1 or not np.isfinite(self.t).all() or np.any(self.t < 0):
            raise ValueError("Temperatures must be finite and non-negative.")
        if not np.isfinite(self.gibbs.values).all():
            raise ValueError("Gibbs energies must be finite.")

    # ------------------------------------------------------------------
    # First derivative ∂G/∂x
    # ------------------------------------------------------------------
    def dGdx(self) -> pd.DataFrame:
        """
        Compute the first derivative ∂G/∂x using np.gradient along the x axis.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same index (x) and columns (t),
            containing (∂G/∂x)(x, t).
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
        Compute the second derivative ∂²G/∂x² from G(x, t)
        using two calls to np.gradient along x.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same index (x) and columns (t),
            containing (∂²G/∂x²)(x, t).
        """
        G = self.gibbs.values  # (nX, nT)

        # First derivative
        dGdx = np.gradient(G, self.x, axis=0, edge_order=2)
        # Second derivative
        d2Gdx2_array = np.gradient(dGdx, self.x, axis=0, edge_order=2)

        d2Gdx2_df = pd.DataFrame(d2Gdx2_array, index=self.gibbs.index, columns=self.gibbs.columns)
        return d2Gdx2_df

    # ------------------------------------------------------------------
    # Spinodal curve
    # ------------------------------------------------------------------
    def spinodal_curve(self, atol: float = 1e-8) -> pd.DataFrame:
        """Return curvature zeros and sign changes, interpolated on the grid.

        All branches are retained. An entirely flat curvature column carries
        no identifiable spinodal boundary and contributes no points.
        """
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative.")
        points = []
        for temperature, curvature in zip(self.t, self.d2Gdx2().values.T):
            zero = np.isclose(curvature, 0, atol=atol, rtol=0)
            if zero.all():
                continue
            roots = list(self.x[zero])
            for j in range(len(self.x) - 1):
                y1, y2 = curvature[j : j + 2]
                if not zero[j] and not zero[j + 1] and np.signbit(y1) != np.signbit(y2):
                    roots.append(self.x[j] - y1 * (self.x[j + 1] - self.x[j]) / (y2 - y1))
            unique = []
            for root in sorted(roots):
                if not unique or not np.isclose(root, unique[-1], rtol=0, atol=1e-10):
                    unique.append(root)
            points.extend((root, temperature) for root in unique)
        return pd.DataFrame(points, columns=["x", "t"], dtype=float).sort_values(
            by="x", ignore_index=True
        )

    # ------------------------------------------------------------------
    # Binodal curve
    # ------------------------------------------------------------------
    def binodal_curve(self, atol: float = 1e-8, ninterp: int = 10) -> pd.DataFrame:
        """Coexistence endpoints from the discrete lower convex envelope.

        Tie-lines that skip sampled compositions delimit miscibility gaps.
        All grid values are checked, so a narrow instability cannot hide
        between interpolation probes. Multiple gaps per temperature are
        returned. Accuracy is limited by the supplied composition grid.
        ``ninterp`` is retained for call compatibility; it is no longer used.
        """
        positive_integer(ninterp, "ninterp")
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative.")
        points = []
        for temperature, values in zip(self.t, self.gibbs.values.T):
            hull = []
            for index in range(len(self.x)):
                while len(hull) >= 2:
                    a, b = hull[-2:]
                    left = (values[b] - values[a]) / (self.x[b] - self.x[a])
                    right = (values[index] - values[b]) / (self.x[index] - self.x[b])
                    if left <= right:
                        break
                    hull.pop()
                hull.append(index)
            for a, b in zip(hull, hull[1:]):
                if b <= a + 1:
                    continue
                interior = slice(a + 1, b)
                line = values[a] + (values[b] - values[a]) * (
                    (self.x[interior] - self.x[a]) / (self.x[b] - self.x[a])
                )
                if np.max(values[interior] - line) > atol:
                    points.extend(((self.x[a], temperature), (self.x[b], temperature)))
        return pd.DataFrame(points, columns=["x", "t"], dtype=float).sort_values(
            by="x", ignore_index=True
        )

    # ------------------------------------------------------------------
    # Critical points
    # ------------------------------------------------------------------
    def critical_point(self, atol: float = 1e-8, curve: str = "spinodal") -> dict:
        """Highest-temperature boundary on the tabulated grid.

        Return NaNs when no boundary exists. This is a grid estimate, not an
        extrapolation to an unsampled thermodynamic critical temperature.
        """
        if curve not in ("spinodal", "binodal"):
            raise ValueError("curve must be 'spinodal' or 'binodal'.")
        boundary = getattr(self, f"{curve}_curve")(atol=atol)
        if boundary.empty:
            return {"x_c": np.nan, "t_c": np.nan}
        highest = boundary[boundary["t"] == boundary["t"].max()]
        return {"x_c": float(highest["x"].mean()), "t_c": float(highest["t"].iloc[0])}
