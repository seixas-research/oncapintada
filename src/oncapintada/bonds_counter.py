# -*- coding: utf-8 -*-
# file: bonds_counter.py

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

# from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

# Defining Pair type
Pair = Tuple[str, str]


@dataclass
class BondCountResult:
    counts: Dict[Pair, int]
    total: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return the pair frequency and probabilities as a pandas DataFrame.
        The resulting DataFrame contains rows for each observed pair in lexical order
        with their counts and normalized probabilities, plus a final “TOTAL” row whose
        count equals the sum of all pair occurrences and whose probability is 1.0.
        Returns:
            pd.DataFrame: A DataFrame with columns "pair", "count", and "probability".
        """

        df = pd.DataFrame(
            [
                {
                    "pair": f"{a}-{b}",
                    "count": n,
                    "probability": n / self.total if self.total > 0 else 0,
                }
                for (a, b), n in sorted(self.counts.items())
            ],
            columns=["pair", "count", "probability"],
        )
        df.loc[len(df)] = {
            "pair": "TOTAL",
            "count": self.total,
            "probability": 1.0 if self.total else 0.0,
        }
        return df


class BondsCounter:
    """Count bonded atom pairs within an ASE Atoms object or XYZ file, optionally restricting
    the analysis to a subset of atoms defined by element symbols or explicit indices. The class
    initializes the atomic structure, enforces a cutoff distance for bond detection, configures
    the subset mask, and precomputes the target atom pairs that satisfy the bonding criteria."""

    def __init__(
        self,
        xyz_path: Optional[str] = None,  # xyz file
        atoms: Optional[
            Atoms
        ] = None,  # Pre-loaded ASE Atoms object. If provided, xyz_path is ignored.
        cutoff: float = 3.5,  # Cutoff radius in angstroms (default: 3.5 Å).
        subset_symbols: Optional[
            str
        ] = None,  # Element symbols defining the subset of atoms to consider.
        subset_indices: Optional[Union[Iterable[int], np.ndarray]] = None,
    ):
        if atoms is None and xyz_path is None:
            raise ValueError("Provide `atoms` or `xyz_path`.")
        self.atoms: Atoms = atoms if atoms is not None else read(xyz_path)
        self.cutoff: float = float(cutoff)  # Cutoff distance for bond detection
        if not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError("cutoff must be finite and positive.")
        self._subset_mask: Optional[np.ndarray] = None  # Mask for subset of atoms

        if subset_symbols is not None:
            self.set_subset_from_symbols(subset_symbols)
        elif subset_indices is not None:
            self.set_subset(subset_indices)
        else:
            self.set_subset(None)

        self.target_pairs: set[Pair] = self.get_target_pairs()

        # self.results: Optional[Dict] = None

    def set_subset(self, subset: Optional[Union[Iterable[int], np.ndarray]]) -> None:
        """Set or reset the subset mask for the current atoms selection.

        Parameters
        ----------
        subset : Optional[Union[Iterable[int], np.ndarray]]
            Indices of atoms to include in the subset. If ``None``, all atoms
            are included; otherwise, only the atoms at the provided indices
            are marked as part of the subset.
        """

        n = len(self.atoms)
        if subset is None:
            self._subset_mask = np.ones(n, dtype=bool)
        else:
            idx = np.asarray(list(subset))
            if idx.ndim != 1 or (idx.size and not np.issubdtype(idx.dtype, np.integer)):
                raise TypeError("Subset indices must be a 1D sequence of integers.")
            idx = idx.astype(int)
            if (idx < 0).any() or (idx >= n).any():
                raise ValueError("Subset indices out of range.")
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            self._subset_mask = mask
        self.target_pairs = self.get_target_pairs()

    def set_subset_from_symbols(self, symbols: Iterable[str]) -> None:
        """
        Update the internal subset mask to include only atoms whose chemical symbols
        appear in the provided iterable of symbols, effectively filtering the structure
        to those species of interest.
        """

        wanted = [symbols] if isinstance(symbols, str) else list(set(symbols))
        chem = np.array(self.atoms.get_chemical_symbols())
        mask = np.isin(chem, wanted)
        self._subset_mask = mask
        self.target_pairs = self.get_target_pairs()

    def get_target_pairs(self) -> set[Pair]:
        """Return unique unordered species pairs in the selected subset."""

        pairs = set()
        if self._subset_mask is None:
            chem_symbols = set(self.atoms.get_chemical_symbols())
        else:
            chem_symbols = set(np.array(self.atoms.get_chemical_symbols())[self._subset_mask])
        for a in chem_symbols:
            for b in chem_symbols:
                pair = tuple(sorted((a, b)))
                pairs.add(pair)
        return pairs

    def count_bonds(self) -> BondCountResult:
        """Count unique chemical bonds among the selected subset of atoms.

        Returns
        -------
        BondCountResult
            Result containing a dictionary with counts for each target pair and the total number of bonds counted.
        """

        chem = np.asarray(self.atoms.get_chemical_symbols())
        i_idx, j_idx, shifts = self._neighbors()
        # Count each undirected bond once, retaining distinct periodic images.
        # Self-image bonds have i == j; select one of S and -S lexicographically.
        keep = (i_idx < j_idx) | np.asarray(
            [i == j and tuple(shift) > (0, 0, 0) for i, j, shift in zip(i_idx, j_idx, shifts)],
            dtype=bool,
        )
        counts = {p: 0 for p in self.get_target_pairs()}
        for i, j in zip(i_idx[keep], j_idx[keep]):
            counts[tuple(sorted((chem[i], chem[j])))] += 1
        return BondCountResult(counts=counts, total=sum(counts.values()))

    def _neighbors(self):
        i, j, shifts = neighbor_list("ijS", self.atoms, self.cutoff, self_interaction=False)
        selected = self._subset_mask[i] & self._subset_mask[j]
        return i[selected], j[selected], shifts[selected]

    def bonds_dataframe(self) -> pd.DataFrame:
        """Return the count of bonds as a pandas DataFrame produced by `count_bonds`."""
        return self.count_bonds().to_dataframe()

    def counter_concentrations(self) -> pd.DataFrame:
        """Return the concentrations of each element in the selected subset of atoms as a pandas DataFrame."""
        if self._subset_mask is None:
            chem = np.array(self.atoms.get_chemical_symbols())
        else:
            chem = np.array(self.atoms.get_chemical_symbols())[self._subset_mask]
        unique, counts = np.unique(chem, return_counts=True)
        total = len(chem)
        data = {
            "element": unique,
            "count": counts,
            "concentration": counts / total if total > 0 else 0,
        }
        df = pd.DataFrame(data)
        return df

    def warren_cowley_parameters(self) -> pd.DataFrame:
        """Return α_ab = 1 - P(b | a) / c_b for selected neighbor shells.

        For each lexically sorted pair a-b, a is the central species. Conditional
        probabilities count directed neighbors, including periodic images. On
        lattices with unequal coordination α_ab need not equal α_ba. A species
        with no neighbors receives 1, retaining the historical no-bond convention.
        """
        concentrations_df = self.counter_concentrations()
        concentrations = dict(zip(concentrations_df["element"], concentrations_df["concentration"]))
        chem = np.asarray(self.atoms.get_chemical_symbols())
        i, j, _ = self._neighbors()
        data = []
        for a, b in sorted(self.get_target_pairs()):
            centers = chem[i] == a
            total = int(centers.sum())
            count = int(np.count_nonzero(centers & (chem[j] == b)))
            conditional = count / total if total else 0.0
            alpha = 1 - conditional / concentrations[b]
            data.append({"pair": f"{a}-{b}", "warren_cowley": alpha})
        return pd.DataFrame(data, columns=["pair", "warren_cowley"])
