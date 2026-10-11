"""Physical and enumeration invariants of the finite-supercell builder."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from oncapintada import (
    ClusterExpansionGenerator,
    ClusterExpansionOptions,
    generate_cluster_expansion,
)


@pytest.fixture
def square():
    return Atoms("Cu4", positions=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])


@pytest.fixture
def options():
    return ClusterExpansionOptions("Cu", ("Cu", "Au"), supercell=(1, 1, 1), pair_cutoff=1.01)


def test_binary_square_class_populations(square, options):
    result = generate_cluster_expansion(square, options)
    assert result.exhaustive
    assert not result.sampled and not result.truncated
    assert result.enumerated == result.total_decorations == 16
    assert len(result.configs) == 6
    assert sorted(c.degeneracy for c in result.configs) == [1, 1, 2, 4, 4, 4]
    assert result.orbits[0].multiplicity == 4
    # Independent known square: adjacent and diagonal equiatomic decorations.
    mixed = [c for c in result.configs if c.species_counts == (2, 2)]
    assert sorted(tuple(c.correlation) for c in mixed) == [(2, 2, 0, 4, 0), (2, 2, 1, 2, 1)]
    assert result.correlations.shape == (6, len(result.correlation_labels))
    assert [c.species_counts for c in result.configs[:2]] == [(4, 0), (0, 4)]
    assert square.get_chemical_symbols() == ["Cu"] * 4


def test_multicomponent_against_independent_brute_force(square, options):
    options = replace(options, species=("Cu", "Au", "Ag"))
    result = generate_cluster_expansion(square, options)
    expected = {}
    for occupation in product(range(3), repeat=4):
        counts = tuple(occupation.count(s) for s in range(3))
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        tuples = [tuple(sorted((occupation[a], occupation[b]))) for a, b in edges]
        histogram = tuple(
            tuples.count(pair) for pair in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
        )
        fingerprint = counts + histogram
        expected[fingerprint] = expected.get(fingerprint, 0) + 1
    assert {tuple(c.correlation): c.degeneracy for c in result.configs} == expected
    assert result.enumerated == 81
    assert sum(c.degeneracy for c in result.configs) == 81


def test_higher_orders_and_feature_labels(square, options):
    options = replace(
        options,
        species=("Cu", "Au", "Ag"),
        pair_cutoff=0,
        triplet_cutoff=1.5,
        quadruplet_cutoff=1.5,
    )
    builder = ClusterExpansionGenerator(square, options)
    result = builder.generate()
    assert [(o.order, o.multiplicity) for o in result.orbits] == [(3, 4), (4, 1)]
    assert len(result.correlation_labels) == 3 + 10 + 15
    assert len(set(result.correlation_labels)) == len(result.correlation_labels)
    assert result.correlation_labels[3].endswith("Cu-Cu-Cu")
    for config in result.configs:
        assert config.correlation[:3].sum() == 4
        assert config.correlation[3:13].sum() == 4
        assert config.correlation[13:].sum() == 1
        np.testing.assert_array_equal(builder.correlation(config.atoms), config.correlation)


def test_fixed_composition_exact_rounding_and_degeneracy(square, options):
    result = generate_cluster_expansion(square, replace(options, composition=(0.5, 0.5)))
    assert result.total_decorations == result.enumerated == 6
    assert len(result.configs) == 2
    assert sorted(c.degeneracy for c in result.configs) == [2, 4]
    assert all(c.species_counts == (2, 2) for c in result.configs)
    result = generate_cluster_expansion(
        square,
        replace(
            options,
            species=("Cu", "Au", "Ag"),
            composition={"Ag": 1 / 3, "Au": 1 / 3, "Cu": 1 / 3},
        ),
    )
    assert all(c.species_counts == (2, 1, 1) for c in result.configs)
    assert result.enumerated == result.total_decorations == 12


def test_fixed_space_uses_multinomial_not_full_space(square, options):
    result = generate_cluster_expansion(
        square,
        replace(
            options,
            composition=(1, 0),
            max_configs=1,
            max_enumeration=1,
        ),
    )
    assert result.exhaustive and result.enumerated == 1
    assert result.configs[0].species_counts == (4, 0)


def test_unrestricted_sampling_is_seeded_and_keeps_endpoints(square, options):
    options = replace(options, max_enumeration=8, species=("Cu", "Au", "Ag"))
    first, second = [generate_cluster_expansion(square, options) for _ in range(2)]
    assert first.sampled and first.truncated and not first.exhaustive
    assert first.enumerated <= 8
    np.testing.assert_array_equal(first.correlations, second.correlations)
    assert [c.species_counts for c in first.configs[:3]] == [(4, 0, 0), (0, 4, 0), (0, 0, 4)]
    assert all(c.degeneracy == 1 for c in first.configs[:3])


def test_fixed_sampling_preserves_counts(square, options):
    options = replace(options, composition=(0.5, 0.5), max_enumeration=3)
    result = generate_cluster_expansion(square, options)
    assert result.sampled and result.enumerated == 3
    assert all(c.species_counts == (2, 2) for c in result.configs)
    assert sum(c.degeneracy for c in result.configs) == 3


def test_configuration_cap_marks_partial_enumeration(square, options):
    result = generate_cluster_expansion(square, replace(options, max_configs=3))
    assert len(result.configs) == 3
    assert result.truncated and not result.exhaustive and not result.sampled
    assert result.enumerated < result.total_decorations


def test_periodic_minimum_image_ring(options):
    parent = Atoms(
        "Cu4", positions=[[i, 0, 0] for i in range(4)], cell=[4, 5, 5], pbc=[True, False, False]
    )
    result = generate_cluster_expansion(parent, options)
    assert result.orbits[0].multiplicity == 4
    assert len(result.configs) == 6
    assert all(np.array_equal(c.atoms.pbc, parent.pbc) for c in result.configs)


def test_triclinic_cell_wraps_active_pairs(options):
    parent = Atoms(
        "Cu2",
        scaled_positions=[[0, 0, 0], [0.9, 0.9, 0]],
        cell=[[4, 0, 0], [1, 3, 0], [0, 0, 8]],
        pbc=[True, True, False],
    )
    result = generate_cluster_expansion(parent, options)
    assert len(result.orbits) == 1
    assert result.orbits[0].radius == pytest.approx(np.linalg.norm([0.5, 0.3, 0]))


def test_supercell_and_spectators_preserved(options, tmp_path):
    parent = Atoms("CuO", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[2, 2, 2], pbc=True)
    builder = ClusterExpansionGenerator(
        parent, replace(options, supercell=(2, 1, 1), pair_cutoff=2.1)
    )
    result = builder.generate()
    assert result.active_sites == 2
    for config in result.configs:
        assert [config.atoms[i].symbol for i in (1, 3)] == ["O", "O"]
        np.testing.assert_allclose(config.atoms.cell.lengths(), [4, 2, 2])
        np.testing.assert_allclose(config.atoms.positions, parent.repeat((2, 1, 1)).positions)
    path = tmp_path / "configs.extxyz"
    result.write(path)
    frames = read(path, index=":")
    assert len(frames) == len(result.configs)
    for frame, config in zip(frames, result.configs):
        np.testing.assert_array_equal(frame.info["ce_correlation"], config.correlation)
        assert frame.info["ce_degeneracy"] == config.degeneracy
        assert frame.info["ce_exhaustive"]
    from_path = ClusterExpansionGenerator(path, replace(options, supercell=(1, 1, 1))).generate()
    assert from_path.active_sites >= 1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"species": ("Cu",)}, "at least 2"),
        ({"species": ("Cu", "Cu")}, "distinct"),
        ({"species": ("Cu", "Fake")}, "valid"),
        ({"active_element": "X"}, "valid"),
        ({"supercell": (0, 1, 1)}, "positive integer"),
        ({"supercell": (1.5, 1, 1)}, "integer"),
        ({"supercell": (1, 1)}, "three"),
        ({"pair_cutoff": -1}, "non-negative"),
        ({"pair_cutoff": np.nan}, "finite"),
        ({"pair_cutoff": 0}, "At least one"),
        ({"distance_tolerance": 0}, "positive"),
        ({"max_configs": 1}, "endpoint"),
        ({"max_enumeration": 1}, "endpoint"),
        ({"max_clusters": 0}, "positive"),
        ({"composition": (0.2, 0.2)}, "sum to 1"),
        ({"composition": (1.1, -0.1)}, "non-negative"),
        ({"composition": (np.nan, 1)}, "finite"),
        ({"composition": {"Cu": 1}}, "every substitution species"),
    ],
)
def test_options_validation(options, change, message):
    with pytest.raises((ValueError, TypeError), match=message):
        replace(options, **change)


def test_cluster_cap_fails_without_biased_partial_orbits(square, options):
    with pytest.raises(ValueError, match="max_clusters"):
        ClusterExpansionGenerator(square, replace(options, max_clusters=2))


def test_empty_sublattice_and_invalid_periodicity(options):
    with pytest.raises(ValueError, match="No active sites"):
        ClusterExpansionGenerator(Atoms("O"), options)
    with pytest.raises(ValueError, match="nonzero cell vectors"):
        ClusterExpansionGenerator(Atoms("Cu", pbc=True), options)


def test_correlation_rejects_changes_outside_occupations(square, options):
    builder = ClusterExpansionGenerator(square, options)
    changed = square.copy()
    changed.positions[0, 0] += 0.1
    with pytest.raises(ValueError, match="preserve template"):
        builder.correlation(changed)
    changed = square.copy()
    changed[0].symbol = "Ag"
    with pytest.raises(ValueError, match="outside this cluster basis"):
        builder.correlation(changed)
