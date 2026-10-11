"""Calango-style, finite-supercell occupational cluster expansion builder.

Geometric orbits use binned minimum-image distances. Fingerprint equality is
correlation equivalence at the selected cutoffs, not proof of space-group
symmetry equivalence. No ICET dependency is required.
"""

from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement, product
from math import comb
from pathlib import Path
from typing import Mapping

import numpy as np
from ase import Atoms
from ase.io import read, write

from ._validation import chemical_species, exact_counts, fractions, positive_integer


@dataclass(frozen=True)
class ClusterExpansionOptions:
    """Controls for decorating a parent element's active sublattice.

    Cutoffs and ``distance_tolerance`` are in Å. ``composition`` optionally
    fixes active-site fractions, as a sequence in species order or a mapping
    containing every species. Fractions must sum to one; integer counts use
    largest remainder, breaking ties in species order. All budgets are hard
    limits. Unrestricted generation reserves one endpoint for each species.
    """

    active_element: str
    species: tuple[str, ...]
    supercell: tuple[int, int, int] = (2, 2, 2)
    pair_cutoff: float = 4.0
    triplet_cutoff: float = 0.0
    quadruplet_cutoff: float = 0.0
    composition: tuple[float, ...] | Mapping[str, float] | None = None
    max_configs: int = 200
    max_enumeration: int = 200_000
    max_clusters: int = 200_000
    seed: int | None = 42
    distance_tolerance: float = 0.05

    def __post_init__(self):
        chemical_species((self.active_element,))
        species = chemical_species(self.species, minimum=2)
        object.__setattr__(self, "species", species)
        if len(self.supercell) != 3:
            raise ValueError("supercell must contain three positive integers.")
        object.__setattr__(
            self,
            "supercell",
            tuple(positive_integer(n, "supercell repetition") for n in self.supercell),
        )
        for name in ("max_configs", "max_enumeration", "max_clusters"):
            positive_integer(getattr(self, name), name)
        cutoffs = (self.pair_cutoff, self.triplet_cutoff, self.quadruplet_cutoff)
        if not np.isfinite(cutoffs).all() or any(c < 0 for c in cutoffs):
            raise ValueError("Cluster cutoffs must be finite and non-negative.")
        if max(cutoffs) == 0:
            raise ValueError("At least one cluster cutoff must be positive.")
        if not np.isfinite(self.distance_tolerance) or self.distance_tolerance <= 0:
            raise ValueError("distance_tolerance must be finite and positive.")
        if self.composition is not None:
            values = self.composition
            if isinstance(values, Mapping):
                if set(values) != set(species):
                    raise ValueError(
                        "composition must specify every substitution species exactly once."
                    )
                values = [values[s] for s in species]
            values = fractions(values, len(species))
            object.__setattr__(self, "composition", tuple(values))
        elif min(self.max_configs, self.max_enumeration) < len(species):
            raise ValueError("Budgets must accommodate the pure endpoint for every species.")
        # Validate seeds even if no sampling is needed.
        np.random.default_rng(self.seed)


@dataclass(frozen=True)
class ClusterOrbit:
    """Clusters sharing a rounded, sorted pair-distance signature.

    ``sites`` contains indices into the active-site list, not the full atoms.
    Multiplicity is the number of finite-supercell clusters in this orbit.
    """

    order: int
    radius: float
    distance_signature: tuple[int, ...]
    sites: tuple[tuple[int, ...], ...]

    @property
    def multiplicity(self):
        return len(self.sites)


@dataclass
class ClusterExpansionConfig:
    """Representative decoration and raw integer correlation histogram.

    ``degeneracy`` is the number of examined decorations with this fingerprint.
    It is an exact class population only when the result is exhaustive.
    """

    atoms: Atoms
    species_counts: tuple[int, ...]
    correlation: np.ndarray
    degeneracy: int = 1

    @property
    def composition(self):
        """Active-site fractions in the declared species order."""
        return np.asarray(self.species_counts, dtype=float) / sum(self.species_counts)


@dataclass
class ClusterExpansionResult:
    """Generated ensemble, feature labels, and coverage diagnostics."""

    configs: list[ClusterExpansionConfig]
    species: tuple[str, ...]
    active_indices: tuple[int, ...]
    orbits: tuple[ClusterOrbit, ...]
    correlation_labels: tuple[str, ...]
    enumerated: int
    total_decorations: int
    sampled: bool
    truncated: bool

    @property
    def active_sites(self):
        return len(self.active_indices)

    @property
    def exhaustive(self):
        return not self.sampled and not self.truncated

    @property
    def correlations(self):
        """Design matrix, one integer fingerprint per configuration."""
        return np.stack([config.correlation for config in self.configs])

    def write(self, filename: str | Path, format: str = "extxyz"):
        """Export ASE frames with species counts, correlations and observed hits."""
        frames = []
        for config in self.configs:
            atoms = config.atoms.copy()
            atoms.info.update(
                ce_species=list(self.species),
                ce_species_counts=list(config.species_counts),
                ce_correlation=config.correlation.copy(),
                ce_degeneracy=config.degeneracy,
                ce_exhaustive=self.exhaustive,
            )
            frames.append(atoms)
        write(filename, frames, format=format)


def _multiset_permutations(counts):
    """Unique multiset permutations in lexicographic order, without recursion."""
    values = [s for s, count in enumerate(counts) for _ in range(count)]
    while True:
        yield tuple(values)
        i = len(values) - 2
        while i >= 0 and values[i] >= values[i + 1]:
            i -= 1
        if i < 0:
            return
        j = len(values) - 1
        while values[j] <= values[i]:
            j -= 1
        values[i], values[j] = values[j], values[i]
        values[i + 1 :] = reversed(values[i + 1 :])


def _cliques(distances, order, cutoff):
    """Enumerate bounded-order cliques, pruning incompatible sites early."""
    adjacency = [set(np.flatnonzero(row <= cutoff)) - {i} for i, row in enumerate(distances)]

    def extend(prefix, candidates):
        if len(prefix) == order:
            yield prefix
            return
        for i, site in enumerate(candidates):
            tail = [s for s in candidates[i + 1 :] if s in adjacency[site]]
            if len(tail) >= order - len(prefix) - 1:
                yield from extend(prefix + (site,), tail)

    yield from extend((), list(range(len(distances))))


class ClusterExpansionGenerator:
    """Build a reusable cluster basis on a copied ASE parent structure.

    ``parent`` may also be a structure filename supported by ASE. Only atoms
    matching ``options.active_element`` are substituted; spectators, cell,
    positions and boundary conditions are preserved after supercell repetition.
    Clusters span distinct active sites, with every mutual minimum-image
    distance within the order's cutoff. Periodic self-image clusters are not
    part of this finite-supercell basis.
    """

    def __init__(self, parent: Atoms | str | Path, options: ClusterExpansionOptions):
        if not isinstance(options, ClusterExpansionOptions):
            raise TypeError("options must be ClusterExpansionOptions.")
        atoms = parent if isinstance(parent, Atoms) else read(parent)
        if len(atoms) == 0 or not np.isfinite(atoms.positions).all():
            raise ValueError("parent must contain atoms with finite positions.")
        if not np.isfinite(atoms.cell.array).all():
            raise ValueError("parent cell must be finite.")
        periodic_vectors = atoms.cell.array[atoms.pbc]
        if len(periodic_vectors) and np.linalg.matrix_rank(periodic_vectors) < len(
            periodic_vectors
        ):
            raise ValueError("Periodic directions require independent, nonzero cell vectors.")
        self.options = options
        self.template = atoms.repeat(options.supercell)
        self.active_indices = tuple(
            int(i)
            for i in np.flatnonzero(
                np.asarray(self.template.get_chemical_symbols()) == options.active_element
            )
        )
        if not self.active_indices:
            raise ValueError("No active sites of the chosen element in the parent structure.")
        self.orbits = self._build_orbits()
        self._buckets = [
            tuple(combinations_with_replacement(range(len(options.species)), orbit.order))
            for orbit in self.orbits
        ]
        self._bucket_indices = [
            {bucket: i for i, bucket in enumerate(buckets)} for buckets in self._buckets
        ]
        labels = [f"point {s}" for s in options.species]
        for i, (orbit, buckets) in enumerate(zip(self.orbits, self._buckets)):
            kind = {2: "pair", 3: "triplet", 4: "quadruplet"}[orbit.order]
            labels.extend(
                f"{kind} orbit={i} r={orbit.radius:.6g} "
                + "-".join(options.species[s] for s in bucket)
                for bucket in buckets
            )
        self.correlation_labels = tuple(labels)

    def _build_orbits(self):
        active = self.template[list(self.active_indices)]
        distances = active.get_all_distances(mic=True)
        orbits = []
        for order, cutoff in (
            (2, self.options.pair_cutoff),
            (3, self.options.triplet_cutoff),
            (4, self.options.quadruplet_cutoff),
        ):
            if cutoff == 0:
                continue
            grouped = {}
            radii = {}
            for count, sites in enumerate(_cliques(distances, order, cutoff), 1):
                if count > self.options.max_clusters:
                    raise ValueError(
                        f"Order {order} exceeds max_clusters; reduce cutoff or supercell."
                    )
                lengths = [distances[a, b] for a, b in combinations(sites, 2)]
                signature = tuple(
                    sorted(
                        int(np.floor(d / self.options.distance_tolerance + 0.5)) for d in lengths
                    )
                )
                grouped.setdefault(signature, []).append(sites)
                radii[signature] = max(radii.get(signature, 0.0), max(lengths))
            for signature in sorted(grouped):
                orbits.append(
                    ClusterOrbit(order, radii[signature], signature, tuple(grouped[signature]))
                )
        return tuple(orbits)

    def _fingerprint(self, occupation):
        values = list(np.bincount(occupation, minlength=len(self.options.species)))
        for orbit, buckets, indices in zip(self.orbits, self._buckets, self._bucket_indices):
            histogram = [0] * len(buckets)
            for sites in orbit.sites:
                bucket = tuple(sorted(occupation[s] for s in sites))
                histogram[indices[bucket]] += 1
            values.extend(histogram)
        return tuple(int(v) for v in values)

    def correlation(self, atoms: Atoms):
        """Fingerprint a decoration on exactly this supercell and site ordering.

        Relaxed or reordered structures require mapping back to the parent;
        this method refuses changed geometry rather than guessing that map.
        """
        if (
            len(atoms) != len(self.template)
            or not np.array_equal(atoms.pbc, self.template.pbc)
            or not np.allclose(atoms.cell.array, self.template.cell.array, rtol=0, atol=1e-8)
            or not np.allclose(atoms.positions, self.template.positions, rtol=0, atol=1e-8)
        ):
            raise ValueError("Decoration must preserve template cell, positions and atom ordering.")
        symbols = atoms.get_chemical_symbols()
        active = set(self.active_indices)
        if any(s != self.template[i].symbol for i, s in enumerate(symbols) if i not in active):
            raise ValueError("Spectator species must remain unchanged.")
        lookup = {s: i for i, s in enumerate(self.options.species)}
        if any(symbols[i] not in lookup for i in self.active_indices):
            raise ValueError("Active sites contain species outside this cluster basis.")
        return np.asarray(self._fingerprint(tuple(lookup[symbols[i]] for i in self.active_indices)))

    def generate(self):
        """Enumerate a small occupation space, or sample it reproducibly.

        Caps may stop exploration before complete class populations are known.
        ``sampled`` and ``truncated`` distinguish random draws and budget stops.
        No pure endpoints are added under a fixed composition.
        """
        options = self.options
        n, k = len(self.active_indices), len(options.species)
        counts = None
        total = k**n
        if options.composition is not None:
            counts = exact_counts(options.composition, n)
            total, remaining = 1, n
            for count in counts:
                total *= comb(remaining, int(count))
                remaining -= int(count)
        sampled = total > options.max_enumeration
        configs, seen = [], {}
        enumerated = 0
        truncated = False

        def consider(occupation):
            nonlocal enumerated
            enumerated += 1
            fingerprint = self._fingerprint(occupation)
            if fingerprint in seen:
                configs[seen[fingerprint]].degeneracy += 1
                return True
            if len(configs) >= options.max_configs:
                return False
            atoms = self.template.copy()
            for site, symbol_index in zip(self.active_indices, occupation):
                atoms[site].symbol = options.species[symbol_index]
            seen[fingerprint] = len(configs)
            configs.append(
                ClusterExpansionConfig(
                    atoms, tuple(fingerprint[:k]), np.asarray(fingerprint, dtype=np.int64)
                )
            )
            return True

        if counts is None:
            for s in range(k):
                consider((s,) * n)
        if sampled:
            rng = np.random.default_rng(options.seed)
            base = None if counts is None else np.repeat(np.arange(k), counts)
            # Bound draws as well as fingerprints; pure draws are already seeded.
            for _ in range(options.max_enumeration - enumerated):
                occupation = tuple(
                    rng.integers(k, size=n) if base is None else rng.permutation(base)
                )
                if counts is None and len(set(occupation)) == 1:
                    continue
                if not consider(occupation):
                    break
            truncated = True
        else:
            occupations = (
                product(range(k), repeat=n) if counts is None else _multiset_permutations(counts)
            )
            for occupation in occupations:
                if counts is None and len(set(occupation)) == 1:
                    continue
                if not consider(occupation):
                    truncated = True
                    break
        return ClusterExpansionResult(
            configs,
            options.species,
            self.active_indices,
            self.orbits,
            self.correlation_labels,
            enumerated,
            total,
            sampled,
            truncated,
        )


def generate_cluster_expansion(parent, options):
    """Convenience wrapper around :class:`ClusterExpansionGenerator`."""
    return ClusterExpansionGenerator(parent, options).generate()
