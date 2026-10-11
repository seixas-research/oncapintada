# -*- coding: utf-8 -*-
# file: bonds_model.py

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write

from ._validation import chemical_species, exact_counts, fractions, positive_integer


class BondsModel:
    """
    Classical bond-interaction Hamiltonian on a 2D lattice with PBC.

    - Square lattice with z = 4 neighbors per site (up, down, left, right).
    - Binary or multicomponent alloys with specified concentrations.
    - Nearest-neighbor pair interactions J_{alpha,beta}.
    """

    def __init__(
        self,
        nx,
        ny,
        a=2.0,
        topology="square",
        species=("Au", "Pt"),
        concentrations=(0.5, 0.5),
        J=None,
        pbc=(True, True, False),
        seed=None,
    ):
        self.nx = positive_integer(nx, "nx")
        self.ny = positive_integer(ny, "ny")
        self.N = self.nx * self.ny
        self.a = float(a)
        if not np.isfinite(self.a) or self.a <= 0:
            raise ValueError("a must be finite and positive.")
        self.topology = topology.lower()
        self.species = list(chemical_species(species))
        self.concentrations = fractions(concentrations, len(self.species))
        if len(pbc) != 3:
            raise ValueError("pbc must contain three boolean values.")
        self.pbc = tuple(bool(x) for x in pbc)
        self.rng = np.random.default_rng(seed)

        # Build lattice positions
        self.positions = self._build_lattice()

        # Random alloy configuration with given concentrations
        self.occupation = self._generate_occupation()

        # ASE Atoms object
        self.atoms = self._build_atoms()

        # Nearest-neighbor list (4 neighbors per site)
        self.neighbors = self._build_neighbor_list()

        # Interaction matrix J_{alpha,beta} as pandas DataFrame
        self.J = self._prepare_J(J)
        if not np.isfinite(self.J.values).all() or not np.allclose(self.J.values, self.J.values.T):
            raise ValueError("J must be finite and symmetric.")

    # ------------------------------------------------------------------
    # Lattice construction
    # ------------------------------------------------------------------
    def _build_lattice(self):
        if self.topology == "square":
            return self._build_square_lattice()
        else:
            raise NotImplementedError(
                f"Topology '{self.topology}' not implemented yet (add _build_<topology>_lattice)."
            )

    def _build_square_lattice(self):
        xs = np.arange(self.nx) * self.a
        ys = np.arange(self.ny) * self.a
        positions = []
        for j in range(self.ny):
            for i in range(self.nx):
                positions.append([xs[i], ys[j], 0.0])
        return np.array(positions, dtype=float)

    # ------------------------------------------------------------------
    # Alloy configuration
    # ------------------------------------------------------------------
    def _generate_occupation(self):
        counts = exact_counts(self.concentrations, self.N)

        labels = []
        for sp, n in zip(self.species, counts):
            labels.extend([sp] * n)
        labels = np.array(labels, dtype=object)
        self.rng.shuffle(labels)
        return labels

    # ------------------------------------------------------------------
    # ASE Atoms
    # ------------------------------------------------------------------
    def _build_atoms(self):
        cell = np.diag([self.nx * self.a, self.ny * self.a, self.a])
        atoms = Atoms(
            symbols=list(self.occupation),
            positions=self.positions,
            cell=cell,
            pbc=self.pbc,
        )
        return atoms

    # ------------------------------------------------------------------
    # Neighbor list with PBC, z = 4
    # ------------------------------------------------------------------
    def _site_index(self, i, j):
        return j * self.nx + i

    def _build_neighbor_list(self):
        """
        Build list of 4 nearest neighbors per site for a square lattice with PBC.

        For each site i:
            neighbors[i] = [right, left, up, down]
        Periodic directions wrap by modulo; missing open-boundary neighbors are -1.
        """
        if self.topology != "square":
            raise NotImplementedError("Neighbor list only implemented for square lattice")

        neighbors = np.full((self.N, 4), -1, dtype=int)
        for j in range(self.ny):
            for i in range(self.nx):
                idx = self._site_index(i, j)
                for direction, (di, dj) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                    ii, jj = i + di, j + dj
                    if not self.pbc[0] and not 0 <= ii < self.nx:
                        continue
                    if not self.pbc[1] and not 0 <= jj < self.ny:
                        continue
                    neighbors[idx, direction] = self._site_index(ii % self.nx, jj % self.ny)
        return neighbors

    # ------------------------------------------------------------------
    # Interaction matrix J_{alpha,beta}
    # ------------------------------------------------------------------
    def _prepare_J(self, J):
        sp = self.species

        if J is None:
            data = np.ones((len(sp), len(sp)), dtype=float)
            return pd.DataFrame(data, index=sp, columns=sp)

        if isinstance(J, (int, float)):
            data = np.full((len(sp), len(sp)), float(J))
            return pd.DataFrame(data, index=sp, columns=sp)

        if isinstance(J, dict):
            data = np.zeros((len(sp), len(sp)), dtype=float)
            df = pd.DataFrame(data, index=sp, columns=sp)
            for (a, b), val in J.items():
                if a not in sp or b not in sp:
                    raise ValueError(f"Unknown species in J key: {(a, b)}")
                df.loc[a, b] = val
                df.loc[b, a] = val  # symmetric
            return df

        if isinstance(J, pd.DataFrame):
            if not set(sp).issubset(J.index) or not set(sp).issubset(J.columns):
                raise ValueError("J DataFrame must contain all species as index and columns")
            return J.reindex(index=sp, columns=sp).copy()

        raise TypeError("Unsupported type for J")

    # ------------------------------------------------------------------
    # Hamiltonian / energy
    # ------------------------------------------------------------------
    def bond_energy(self):
        """H = -Σ_<ij> J_ij, counting each periodic image bond once."""
        return float(self.site_energies().sum())

    def site_energies(self):
        """Half the directed neighbor sum, including periodic self-images.

        A 1×1 periodic square has four directed neighbors and two bonds per
        cell, so its energy is -2 J, consistent with any larger periodic cell.
        Open-boundary directions carry no wrapping neighbors (marked -1).
        """
        energy = np.zeros(self.N)
        for i in range(self.N):
            for j in self.neighbors[i]:
                if j >= 0:
                    energy[i] -= 0.5 * self.J.loc[self.occupation[i], self.occupation[j]]
        return energy

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_lattice(self, n_images=(1, 1), show_bonds=True, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        if len(n_images) != 2:
            raise ValueError("n_images must contain two positive integers.")
        nx_rep, ny_rep = (positive_integer(n, "n_images") for n in n_images)
        nx_half, ny_half = nx_rep // 2, ny_rep // 2

        species_unique = self.species
        cmap = plt.get_cmap("tab10")
        color_map = {sp: cmap(i % 10) for i, sp in enumerate(species_unique)}

        # Periodic images (for visualization only)
        for dx in range(-nx_half, nx_rep - nx_half):
            for dy in range(-ny_half, ny_rep - ny_half):
                shift = np.array([dx * self.nx * self.a, dy * self.ny * self.a, 0.0])
                pos_shifted = self.positions + shift
                for sp in species_unique:
                    mask = self.occupation == sp
                    ax.scatter(
                        pos_shifted[mask, 0],
                        pos_shifted[mask, 1],
                        marker="o",
                        s=50,
                        label=sp if (dx == 0 and dy == 0) else None,
                        color=color_map[sp],
                        edgecolors="k",
                        linewidths=0.5,
                    )

        # Bonds in the central cell
        if show_bonds:
            for i in range(self.N):
                x1, y1, _ = self.positions[i]
                # Right/up bonds count each image once and draw short segments.
                for direction, delta in ((0, (self.a, 0)), (2, (0, self.a))):
                    if self.neighbors[i, direction] >= 0:
                        ax.plot(
                            [x1, x1 + delta[0]],
                            [y1, y1 + delta[1]],
                            linestyle="-",
                            linewidth=1,
                            alpha=0.5,
                        )

        ax.set_aspect("equal")
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_title(
            f"{self.topology.capitalize()} lattice ({self.nx}x{self.ny}), PBC={self.pbc[:2]}"
        )
        ax.legend()
        plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, filename, format=None):
        write(filename, self.atoms, format=format)


# Example usage
if __name__ == "__main__":
    J_binary = {("Au", "Au"): 1.0, ("Pt", "Pt"): 0.5, ("Au", "Pt"): 0.8}
    model_3x3 = BondsModel(
        5, 5, species=["Au", "Pt"], concentrations=[0.3, 0.7], J=J_binary, seed=42
    )

    for i in range(model_3x3.N):
        print(f"Site {i} has {len(model_3x3.neighbors[i])} neighbors")
    E = model_3x3.bond_energy()
    print(f"Total bond energy: {E} eV")

    model_3x3.plot_lattice(n_images=(1, 1), show_bonds=True)
    plt.show()
    # model_3x3.export("lattice.xyz")
