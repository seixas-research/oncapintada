"""Special quasirandom supercells via composition-preserving annealing.

Species-tuple probabilities are matched to independent random-alloy targets
on distance-grouped periodic pair, triplet and quadruplet clusters. This native
histogram objective follows Calango Studio's SQS approach; it is not an ICET
orthogonal cluster basis or a proof of global optimality.
"""

from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from math import factorial
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

from ._validation import chemical_species, exact_counts, fractions, positive_integer


@dataclass(frozen=True)
class SQSOptions:
    """SQS geometry, composition, and search controls.

    Composition fractions must sum to one. They refer to all sites when
    ``active_element=None``, or only to the chosen element's sublattice.
    ``n_steps`` is the swap budget per restart. Annealing temperatures are
    dimensionless objective scales, not physical temperatures in kelvin.
    """

    composition: Mapping[str, float]
    supercell: tuple[int, int, int] = (2, 2, 2)
    active_element: str | None = None
    pair_cutoff: float = 4.0
    triplet_cutoff: float = 0.0
    quadruplet_cutoff: float = 0.0
    distance_tolerance: float = 0.05
    n_steps: int = 10_000
    restarts: int = 3
    seed: int | None = 42
    start_temperature: float = 0.1
    end_temperature: float = 1e-4
    tolerance: float = 1e-8
    pair_weight: float = 1.0
    triplet_weight: float = 0.5
    quadruplet_weight: float = 0.25
    max_clusters: int = 200_000

    def __post_init__(self):
        if not isinstance(self.composition, Mapping):
            raise TypeError("composition must map chemical symbols to fractions.")
        species = chemical_species(self.composition, minimum=2)
        values = fractions([self.composition[s] for s in species], len(species))
        # Copy the caller's mapping; species order determines rounding ties.
        from types import MappingProxyType

        object.__setattr__(self, "composition", MappingProxyType(dict(zip(species, values))))
        if self.active_element is not None:
            chemical_species((self.active_element,))
        if len(self.supercell) != 3:
            raise ValueError("supercell must contain three positive integers.")
        object.__setattr__(
            self,
            "supercell",
            tuple(positive_integer(n, "supercell repetition") for n in self.supercell),
        )
        for name in ("n_steps", "restarts", "max_clusters"):
            positive_integer(getattr(self, name), name)
        for name in ("pair_cutoff", "triplet_cutoff", "quadruplet_cutoff", "tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if max(self.pair_cutoff, self.triplet_cutoff, self.quadruplet_cutoff) == 0:
            raise ValueError("At least one cluster cutoff must be positive.")
        for name in (
            "distance_tolerance",
            "pair_weight",
            "triplet_weight",
            "quadruplet_weight",
            "start_temperature",
            "end_temperature",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.end_temperature > self.start_temperature:
            raise ValueError("end_temperature must not exceed start_temperature.")
        np.random.default_rng(self.seed)


@dataclass(frozen=True)
class SQSOrbit:
    """A periodic geometric cluster group and its objective weight."""

    order: int
    radius: float
    distance_signature: tuple[int, ...]
    multiplicity: int
    weight: float


@dataclass
class SQSResult:
    """Best supercell found, random targets, and search diagnostics.

    ``converged`` means the objective met the requested tolerance. It does not
    assert global optimality. Correlations are normalized unordered species
    tuple probabilities; target fractions use the realized integer composition.
    """

    atoms: Atoms
    species: tuple[str, ...]
    species_counts: tuple[int, ...]
    requested_composition: dict[str, float]
    active_indices: tuple[int, ...]
    orbits: tuple[SQSOrbit, ...]
    correlation_labels: tuple[str, ...]
    correlations: np.ndarray
    target_correlations: np.ndarray
    objective: float
    initial_objective: float
    iterations: int
    accepted_moves: int
    converged: bool
    history: tuple[float, ...]

    @property
    def composition(self):
        """Realized active-site fractions, including any zero-count species."""
        total = sum(self.species_counts)
        return {s: count / total for s, count in zip(self.species, self.species_counts)}

    @property
    def max_error(self):
        return float(np.max(np.abs(self.correlations - self.target_correlations)))

    def correlation_table(self):
        """One row per species tuple, with target, observation and signed error."""
        return pd.DataFrame(
            {
                "label": self.correlation_labels,
                "correlation": self.correlations,
                "target": self.target_correlations,
                "error": self.correlations - self.target_correlations,
            }
        )

    def write(self, filename: str | Path, format: str = "extxyz"):
        """Export the best ASE structure with objective and composition metadata."""
        atoms = self.atoms.copy()
        atoms.info.update(
            sqs_species=list(self.species),
            sqs_species_counts=list(self.species_counts),
            sqs_requested_composition=self.requested_composition,
            sqs_active_indices=list(self.active_indices),
            sqs_objective=self.objective,
            sqs_initial_objective=self.initial_objective,
            sqs_converged=self.converged,
            sqs_iterations=self.iterations,
            sqs_correlation_labels=list(self.correlation_labels),
            sqs_correlations=self.correlations.copy(),
            sqs_target_correlations=self.target_correlations.copy(),
        )
        write(filename, atoms, format=format)


class SQSGenerator:
    """Optimize occupations on a repeated parent geometry.

    Only ``options.active_element`` is substituted when specified. Spectator
    atoms, geometry and partial periodicity are preserved. Every periodic
    image contributes its actual multiplicity. Cutoffs that include a site's
    own periodic image are refused: enlarge the supercell for that range.
    """

    def __init__(self, parent: Atoms | str | Path, options: SQSOptions):
        if not isinstance(options, SQSOptions):
            raise TypeError("options must be SQSOptions.")
        atoms = parent if isinstance(parent, Atoms) else read(parent)
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise ValueError("parent must contain atoms.")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError("parent positions and cell must be finite.")
        periodic = atoms.cell.array[atoms.pbc]
        if not len(periodic) or np.linalg.matrix_rank(periodic) != len(periodic):
            raise ValueError("SQS requires nonzero, independent periodic cell vectors.")
        self.options = options
        self.template = atoms.repeat(options.supercell)
        symbols = np.asarray(self.template.get_chemical_symbols())
        selected = (
            np.ones(len(symbols), dtype=bool)
            if options.active_element is None
            else symbols == options.active_element
        )
        self.active_indices = tuple(int(i) for i in np.flatnonzero(selected))
        if not self.active_indices:
            raise ValueError("No active sites of the chosen element in the parent structure.")
        self.species = tuple(options.composition)
        self.species_counts = exact_counts(
            list(options.composition.values()), len(self.active_indices)
        )
        if np.count_nonzero(self.species_counts) < 2:
            raise ValueError(
                "At least two species must occupy active sites after rounding; enlarge the supercell."
            )
        self.concentrations = self.species_counts / len(self.active_indices)
        self._prepare_basis()

    def _prepare_basis(self):
        options = self.options
        cutoffs = {2: options.pair_cutoff, 3: options.triplet_cutoff, 4: options.quadruplet_cutoff}
        active = self.template[list(self.active_indices)]
        largest = max(cutoffs.values())
        # ASE uses a strict cutoff; nextafter includes pairs exactly on the boundary.
        i, j, distances, vectors = neighbor_list("ijdD", active, np.nextafter(largest, np.inf))
        if np.any(i == j):
            raise ValueError(
                "Cluster cutoff includes a site's own periodic image; enlarge the supercell."
            )
        if np.any(distances < 1e-8):
            raise ValueError("Active sites overlap, including across periodic boundaries.")
        grouped, radii = {}, {}
        counts = {order: 0 for order in cutoffs}

        def keep(order, sites, lengths):
            counts[order] += 1
            if counts[order] > options.max_clusters:
                raise ValueError(f"Order {order} exceeds max_clusters; reduce cutoff or supercell.")
            signature = tuple(
                sorted(int(np.floor(d / options.distance_tolerance + 0.5)) for d in lengths)
            )
            key = order, signature
            grouped.setdefault(key, []).append(sites)
            radii[key] = max(radii.get(key, 0), max(lengths))

        # Images of a distinct base-index pair are separate geometric clusters.
        for a, b, distance in zip(i, j, distances):
            if a < b and distance <= options.pair_cutoff:
                keep(2, (int(a), int(b)), (distance,))
        if options.triplet_cutoff or options.quadruplet_cutoff:
            neighbors = [[] for _ in self.active_indices]
            for a, b, distance, vector in zip(i, j, distances, vectors):
                if a < b:
                    neighbors[a].append((int(b), float(distance), vector))
            for order in (3, 4):
                cutoff = cutoffs[order]
                if cutoff == 0:
                    continue
                for anchor, neighbors_at in enumerate(neighbors):
                    near = [neighbor for neighbor in neighbors_at if neighbor[1] <= cutoff]
                    for tail in combinations(near, order - 1):
                        sites = (anchor,) + tuple(neighbor[0] for neighbor in tail)
                        if len(set(sites)) != order:
                            continue
                        lengths = [neighbor[1] for neighbor in tail]
                        lengths.extend(
                            float(np.linalg.norm(a[2] - b[2])) for a, b in combinations(tail, 2)
                        )
                        if max(lengths) <= cutoff:
                            keep(order, sites, lengths)
        enabled = [order for order, cutoff in cutoffs.items() if cutoff > 0]
        if any(counts[order] == 0 for order in enabled):
            raise ValueError(
                "An enabled cluster cutoff contains no clusters; increase it or disable that order."
            )

        orbits, clusters, orbit_ids, buckets, labels = [], [], [], [], []
        for orbit_id, key in enumerate(
            sorted(grouped, key=lambda key: (key[0], radii[key], key[1]))
        ):
            order, signature = key
            weight = getattr(
                options, {2: "pair_weight", 3: "triplet_weight", 4: "quadruplet_weight"}[order]
            )
            # Per-order shorter-range orbits have greater weight.
            rank = sum(orbit.order == order for orbit in orbits)
            orbit = SQSOrbit(order, radii[key], signature, len(grouped[key]), weight / (rank + 1))
            orbits.append(orbit)
            tuples = tuple(combinations_with_replacement(range(len(self.species)), order))
            buckets.append(tuples)
            kind = {2: "pair", 3: "triplet", 4: "quadruplet"}[order]
            labels.extend(
                f"{kind} orbit={orbit_id} r={orbit.radius:.6g} "
                + "-".join(self.species[s] for s in bucket)
                for bucket in tuples
            )
            clusters.extend(grouped[key])
            orbit_ids.extend([orbit_id] * len(grouped[key]))
        self.orbits = tuple(orbits)
        self.correlation_labels = tuple(labels)
        self._clusters = tuple(clusters)
        self._orbit_ids = np.asarray(orbit_ids)
        self._offsets = np.cumsum([0] + [len(b) for b in buckets])
        self._bucket_indices = [
            {bucket: offset + index for index, bucket in enumerate(tuples)}
            for offset, tuples in zip(self._offsets, buckets)
        ]
        normalizers, weights, targets = [], [], []
        for orbit, tuples in zip(orbits, buckets):
            for bucket in tuples:
                repetitions = np.bincount(bucket, minlength=len(self.species))
                multiplicity = factorial(orbit.order)
                for count in repetitions:
                    multiplicity //= factorial(int(count))
                targets.append(multiplicity * float(np.prod(self.concentrations[list(bucket)])))
                normalizers.append(orbit.multiplicity)
                weights.append(orbit.weight)
        self.target_correlations = np.asarray(targets)
        self._normalizers = np.asarray(normalizers)
        self._weights = np.asarray(weights)
        at_site = [[] for _ in self.active_indices]
        for cluster_id, sites in enumerate(self._clusters):
            for site in sites:
                at_site[site].append(cluster_id)
        self._at_site = tuple(np.asarray(ids, dtype=int) for ids in at_site)

    def _bins(self, occupation, cluster_ids):
        return np.asarray(
            [
                self._bucket_indices[self._orbit_ids[index]][
                    tuple(sorted(occupation[s] for s in self._clusters[index]))
                ]
                for index in cluster_ids
            ],
            dtype=int,
        )

    def _histogram(self, occupation):
        bins = self._bins(occupation, range(len(self._clusters)))
        return np.bincount(bins, minlength=len(self.target_correlations))

    def _objective(self, histogram):
        return float(
            np.sum(self._weights * np.abs(histogram / self._normalizers - self.target_correlations))
        )

    def _occupation(self, atoms):
        if (
            len(atoms) != len(self.template)
            or not np.array_equal(atoms.pbc, self.template.pbc)
            or not np.allclose(atoms.cell.array, self.template.cell.array, rtol=0, atol=1e-8)
            or not np.allclose(atoms.positions, self.template.positions, rtol=0, atol=1e-8)
        ):
            raise ValueError("Decoration must preserve template cell, positions and atom ordering.")
        symbols = atoms.get_chemical_symbols()
        active = set(self.active_indices)
        if any(
            s != self.template[index].symbol
            for index, s in enumerate(symbols)
            if index not in active
        ):
            raise ValueError("Spectator species must remain unchanged.")
        lookup = {s: index for index, s in enumerate(self.species)}
        if any(symbols[index] not in lookup for index in self.active_indices):
            raise ValueError("Active sites contain species outside this SQS basis.")
        occupation = np.asarray([lookup[symbols[index]] for index in self.active_indices])
        if not np.array_equal(
            np.bincount(occupation, minlength=len(self.species)), self.species_counts
        ):
            raise ValueError("Decoration composition must match the realized SQS site counts.")
        return occupation

    def calculate_correlation_functions(self, atoms):
        """Normalized correlations on the ideal template, in feature-label order."""
        return self._histogram(self._occupation(atoms)) / self._normalizers

    def evaluate(self, atoms):
        """Recompute this SQS objective for an independently supplied decoration."""
        return self._objective(self._histogram(self._occupation(atoms)))

    def generate(self):
        """Anneal exact-composition swaps and return the best state visited.

        A local update touches only clusters incident to either swapped site;
        integer histograms make accepted/rejected updates free of drift.
        Independent restarts share a reproducible seeded RNG stream.
        """
        options = self.options
        rng = np.random.default_rng(options.seed)
        base = np.repeat(np.arange(len(self.species)), self.species_counts)
        best_objective = np.inf
        best_occupation = None
        history = []
        iterations = accepted = 0
        initial_objective = None
        for _ in range(options.restarts):
            occupation = rng.permutation(base)
            histogram = self._histogram(occupation)
            current = self._objective(histogram)
            if initial_objective is None:
                initial_objective = current
            if current < best_objective:
                best_objective, best_occupation = current, occupation.copy()
                history.append(current)
            if best_objective <= options.tolerance:
                break
            for step in range(options.n_steps):
                # Select unlike species directly; no unbounded retry loop.
                a = int(rng.integers(len(occupation)))
                b = int(rng.choice(np.flatnonzero(occupation != occupation[a])))
                touched = np.union1d(self._at_site[a], self._at_site[b])
                old_bins = self._bins(occupation, touched)
                occupation[a], occupation[b] = occupation[b], occupation[a]
                new_bins = self._bins(occupation, touched)
                np.add.at(histogram, old_bins, -1)
                np.add.at(histogram, new_bins, 1)
                proposed = self._objective(histogram)
                fraction = step / max(options.n_steps - 1, 1)
                temperature = np.exp(
                    (1 - fraction) * np.log(options.start_temperature)
                    + fraction * np.log(options.end_temperature)
                )
                delta = proposed - current
                iterations += 1
                if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                    current = proposed
                    accepted += 1
                    if current < best_objective:
                        best_objective, best_occupation = current, occupation.copy()
                        history.append(current)
                else:
                    np.add.at(histogram, new_bins, -1)
                    np.add.at(histogram, old_bins, 1)
                    occupation[a], occupation[b] = occupation[b], occupation[a]
                if best_objective <= options.tolerance:
                    break
            if best_objective <= options.tolerance:
                break
        # Independently retally the returned best state rather than using the
        # final annealing state's incremental histogram.
        best_histogram = self._histogram(best_occupation)
        objective = self._objective(best_histogram)
        atoms = self.template.copy()
        for site, label in zip(self.active_indices, best_occupation):
            atoms[site].symbol = self.species[label]
        return SQSResult(
            atoms=atoms,
            species=self.species,
            species_counts=tuple(int(count) for count in self.species_counts),
            requested_composition=dict(options.composition),
            active_indices=self.active_indices,
            orbits=self.orbits,
            correlation_labels=self.correlation_labels,
            correlations=best_histogram / self._normalizers,
            target_correlations=self.target_correlations.copy(),
            objective=objective,
            initial_objective=initial_objective,
            iterations=iterations,
            accepted_moves=accepted,
            converged=objective <= options.tolerance,
            history=tuple(history),
        )


def generate_sqs(parent, composition, **kwargs):
    """Generate an SQS from ASE atoms (or a filename) and composition fractions.

    Keyword arguments are the fields of :class:`SQSOptions`, such as
    ``supercell``, ``active_element``, ``pair_cutoff``, ``n_steps``, and ``seed``.
    """
    return SQSGenerator(parent, SQSOptions(composition=composition, **kwargs)).generate()
