"""Shared validation and composition rounding for atomistic builders."""

from numbers import Integral

import numpy as np
from ase.data import atomic_numbers


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def chemical_species(values, minimum=1):
    if isinstance(values, str):
        raise TypeError("species must be a sequence of chemical symbols.")
    values = tuple(values)
    if len(values) < minimum or len(set(values)) != len(values):
        raise ValueError(f"species must contain at least {minimum} distinct chemical symbols.")
    if any(s not in atomic_numbers or atomic_numbers[s] == 0 for s in values):
        raise ValueError("species must contain valid chemical element symbols.")
    return values


def fractions(values, size, *, normalize=False):
    values = np.asarray(values, dtype=float)
    if values.shape != (size,):
        raise ValueError("species and concentrations must have the same length (1D sequences).")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("concentrations must be finite and non-negative.")
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("concentrations must sum to a positive value.")
    if not normalize and not np.isclose(total, 1.0, rtol=0, atol=1e-8):
        raise ValueError("concentrations must sum to 1.")
    return values / total


def exact_counts(concentrations, n_sites):
    """Largest-remainder apportionment; ties follow the input species order."""
    raw = np.asarray(concentrations) * n_sites
    counts = np.floor(raw).astype(int)
    remainder = n_sites - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def site_indices(mask, atoms):
    """Resolve a symbol, a boolean mask, or distinct non-negative indices."""
    if isinstance(mask, str):
        return np.flatnonzero(np.asarray(atoms.get_chemical_symbols()) == mask)
    mask = np.asarray(mask)
    if mask.ndim != 1:
        raise ValueError("Substitution mask must be one-dimensional.")
    if mask.dtype == bool:
        if len(mask) != len(atoms):
            raise ValueError("Boolean mask length must match number of atoms in template.")
        return np.flatnonzero(mask)
    if mask.size == 0:
        return np.array([], dtype=int)
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Substitution indices must be integers.")
    if (mask < 0).any() or (mask >= len(atoms)).any():
        raise ValueError("Substitution indices out of range.")
    if len(np.unique(mask)) != len(mask):
        raise ValueError("Substitution indices must be unique.")
    return mask.astype(int)
