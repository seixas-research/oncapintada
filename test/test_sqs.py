"""Known correlations, search invariants, and SQS supercell regression tests."""

from dataclasses import replace
from itertools import product
from math import factorial

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read

from oncapintada import SQSGenerator, SQSOptions, generate_sqs


def ring(size=4):
    return Atoms(
        "Cu" * size,
        positions=[[i, 0, 0] for i in range(size)],
        cell=[size, 8, 8],
        pbc=[True, False, False],
    )


@pytest.fixture
def options():
    return SQSOptions(
        {"Cu": 0.5, "Au": 0.5},
        supercell=(1, 1, 1),
        pair_cutoff=1.01,
        n_steps=300,
        restarts=2,
        seed=42,
    )


def test_binary_ring_known_random_and_ordered_correlations(options):
    generator = SQSGenerator(ring(), options)
    assert len(generator.orbits) == 1
    assert generator.orbits[0].multiplicity == 4
    grouped = generator.template.copy()
    grouped.set_chemical_symbols(["Cu", "Cu", "Au", "Au"])
    np.testing.assert_allclose(
        generator.calculate_correlation_functions(grouped), [0.25, 0.5, 0.25]
    )
    assert generator.evaluate(grouped) == 0
    alternating = generator.template.copy()
    alternating.set_chemical_symbols(["Cu", "Au", "Cu", "Au"])
    np.testing.assert_allclose(generator.calculate_correlation_functions(alternating), [0, 1, 0])
    assert generator.evaluate(alternating) == 1
    # The unordered pair statistic corresponds to conventional binary spin
    # correlation P_AA + P_BB - P_AB, whose random target is (c_A-c_B)^2.
    observed = generator.calculate_correlation_functions(grouped)
    assert observed[0] + observed[2] - observed[1] == 0


def test_annealing_reaches_known_zero_and_is_reproducible(options):
    generator = SQSGenerator(ring(), replace(options, seed=5))
    first, second = generator.generate(), generator.generate()
    assert first.objective == 0 and first.converged
    assert first.objective <= first.initial_objective
    assert first.atoms.get_chemical_symbols() == second.atoms.get_chemical_symbols()
    assert first.iterations == second.iterations
    assert first.iterations <= options.n_steps * options.restarts
    assert first.accepted_moves <= first.iterations
    assert first.species_counts == (2, 2)
    assert generator.evaluate(first.atoms) == first.objective
    assert np.all(np.diff(first.history) < 0)


def test_multicomponent_matches_independent_exhaustive_minimum(options):
    composition = {"Cu": 1 / 3, "Au": 1 / 3, "Ag": 1 / 3}
    options = replace(options, composition=composition, n_steps=1000)
    generator = SQSGenerator(ring(6), options)
    buckets = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    target = np.array([1 / 9, 2 / 9, 2 / 9, 1 / 9, 2 / 9, 1 / 9])
    minimum = np.inf
    for occupation in product(range(3), repeat=6):
        if [occupation.count(s) for s in range(3)] != [2, 2, 2]:
            continue
        pairs = [tuple(sorted((occupation[i], occupation[(i + 1) % 6]))) for i in range(6)]
        observed = np.array([pairs.count(bucket) / 6 for bucket in buckets])
        minimum = min(minimum, np.abs(observed - target).sum())
    result = generator.generate()
    np.testing.assert_allclose(result.target_correlations, target)
    assert result.species_counts == (2, 2, 2)
    assert result.objective == pytest.approx(minimum)
    assert not result.converged  # Random targets cannot all fit this tiny ring.
    assert generator.evaluate(result.atoms) == result.objective


def test_periodic_images_retain_fcc_coordination():
    generator = SQSGenerator(
        bulk("Cu", "fcc", a=3.6, cubic=True),
        SQSOptions({"Cu": 0.5, "Au": 0.5}, pair_cutoff=2.7, n_steps=10),
    )
    assert len(generator.template) == 32
    assert generator.orbits[0].multiplicity == 32 * 12 // 2
    result = generator.generate()
    assert result.species_counts == (16, 16)
    assert result.objective <= result.initial_objective
    np.testing.assert_allclose(result.correlations.sum(), 1)


def test_sublattice_spectators_rounding_and_export(options, tmp_path):
    parent = Atoms(
        "CuO", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[1, 4, 4], pbc=[True, False, False]
    )
    options = replace(
        options,
        composition={"Cu": 1 / 3, "Au": 1 / 3, "Ag": 1 / 3},
        active_element="Cu",
        supercell=(4, 1, 1),
    )
    generator = SQSGenerator(parent, options)
    result = generator.generate()
    assert result.species_counts == (2, 1, 1)
    assert result.composition == {"Cu": 0.5, "Au": 0.25, "Ag": 0.25}
    np.testing.assert_allclose(list(result.requested_composition.values()), [1 / 3] * 3)
    for index in (1, 3, 5, 7):
        assert result.atoms[index].symbol == "O"
    np.testing.assert_array_equal(result.atoms.positions, parent.repeat((4, 1, 1)).positions)
    np.testing.assert_array_equal(result.atoms.pbc, parent.pbc)
    assert parent.get_chemical_symbols() == ["Cu", "O"]
    path = tmp_path / "sqs.extxyz"
    result.write(path)
    restored = read(path)
    assert restored.info["sqs_objective"] == pytest.approx(result.objective)
    assert restored.info["sqs_converged"] == result.converged
    np.testing.assert_allclose(restored.info["sqs_target_correlations"], result.target_correlations)
    assert list(restored.info["sqs_correlation_labels"]) == list(result.correlation_labels)
    np.testing.assert_allclose(
        generator.calculate_correlation_functions(restored), result.correlations
    )
    table = result.correlation_table()
    np.testing.assert_allclose(table["error"], table["correlation"] - table["target"])
    np.testing.assert_allclose(result.max_error, table["error"].abs().max())


def test_higher_order_periodic_multiplicities_and_multinomial_targets():
    # 2x2x2 conventional FCC cell: 32 sites, 192 NN pairs,
    # 256 equilateral NN triangles and 64 NN tetrahedra per periodic cell.
    options = SQSOptions(
        {"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
        pair_cutoff=2.7,
        triplet_cutoff=2.7,
        quadruplet_cutoff=2.7,
        n_steps=30,
        restarts=1,
    )
    generator = SQSGenerator(bulk("Cu", "fcc", a=3.6, cubic=True), options)
    assert [(o.order, o.multiplicity) for o in generator.orbits] == [(2, 192), (3, 256), (4, 64)]
    result = generator.generate()
    offset = 0
    for orbit in generator.orbits:
        n_buckets = factorial(3 + orbit.order - 1) // (factorial(orbit.order) * factorial(3 - 1))
        section = slice(offset, offset + n_buckets)
        assert result.correlations[section].sum() == pytest.approx(1)
        assert result.target_correlations[section].sum() == pytest.approx(1)
        # All-Cu tuple probability is c_Cu**order.
        assert result.target_correlations[offset] == pytest.approx(0.5**orbit.order)
        offset += n_buckets
    assert result.objective == pytest.approx(generator.evaluate(result.atoms))
    np.testing.assert_allclose(
        result.correlations, generator.calculate_correlation_functions(result.atoms)
    )


def test_higher_order_mixed_tuple_targets(options):
    square = Atoms(
        "Cu4", positions=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], cell=[8, 8, 8], pbc=True
    )
    options = replace(
        options,
        composition={"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
        pair_cutoff=0,
        triplet_cutoff=1.5,
        quadruplet_cutoff=1.5,
    )
    generator = SQSGenerator(square, options)
    labels = generator.correlation_labels
    target = generator.target_correlations
    index = next(
        i
        for i, label in enumerate(labels)
        if label.startswith("triplet") and label.endswith("Cu-Au-Ag")
    )
    assert target[index] == pytest.approx(6 * 0.5 * 0.25 * 0.25)
    index = next(
        i
        for i, label in enumerate(labels)
        if label.startswith("quadruplet") and label.endswith("Cu-Cu-Au-Ag")
    )
    assert target[index] == pytest.approx(12 * 0.5**2 * 0.25 * 0.25)


def test_incident_histogram_updates_match_full_retally(options):
    # Verify overlapping clusters touching both swap endpoints are updated once.
    generator = SQSGenerator(ring(8), options)
    occupation = np.array([0, 1] * 4)
    histogram = generator._histogram(occupation)
    for a, b in ((0, 1), (0, 7), (1, 5), (2, 6)):
        touched = np.union1d(generator._at_site[a], generator._at_site[b])
        old = generator._bins(occupation, touched)
        occupation[a], occupation[b] = occupation[b], occupation[a]
        new = generator._bins(occupation, touched)
        np.add.at(histogram, old, -1)
        np.add.at(histogram, new, 1)
        np.testing.assert_array_equal(histogram, generator._histogram(occupation))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"composition": {"Cu": 1}}, "at least 2"),
        ({"composition": {"Cu": 1.1, "Au": -0.1}}, "non-negative"),
        ({"composition": {"Cu": 0.2, "Au": 0.2}}, "sum to 1"),
        ({"composition": {"Cu": np.nan, "Au": 1}}, "finite"),
        ({"composition": {"Cu": 0.5, "Fake": 0.5}}, "valid"),
        ({"composition": [0.5, 0.5]}, "map"),
        ({"active_element": "Fake"}, "valid"),
        ({"supercell": (0, 1, 1)}, "positive"),
        ({"supercell": (1.5, 1, 1)}, "integer"),
        ({"supercell": (1, 1)}, "three"),
        ({"n_steps": 0}, "positive"),
        ({"restarts": True}, "integer"),
        ({"pair_cutoff": 0}, "At least one"),
        ({"pair_cutoff": -1}, "non-negative"),
        ({"pair_cutoff": np.inf}, "finite"),
        ({"distance_tolerance": 0}, "positive"),
        ({"pair_weight": 0}, "positive"),
        ({"tolerance": -1}, "non-negative"),
        ({"start_temperature": 0}, "positive"),
        ({"end_temperature": 1}, "must not exceed"),
        ({"max_clusters": 0}, "positive"),
    ],
)
def test_options_validation(options, change, message):
    with pytest.raises((ValueError, TypeError), match=message):
        replace(options, **change)


def test_options_copy_composition(options):
    composition = {"Cu": 0.5, "Au": 0.5}
    options = replace(options, composition=composition)
    composition["Cu"] = 0
    assert options.composition["Cu"] == 0.5
    with pytest.raises(TypeError):
        options.composition["Cu"] = 0


def test_inadequate_geometry_is_refused(options):
    with pytest.raises(ValueError, match="periodic"):
        SQSGenerator(Atoms("Cu4"), options)
    with pytest.raises(ValueError, match="own periodic image"):
        SQSGenerator(ring(2), replace(options, pair_cutoff=2.1))
    with pytest.raises(ValueError, match="No active sites"):
        SQSGenerator(ring(), replace(options, active_element="O"))
    with pytest.raises(ValueError, match="after rounding"):
        SQSGenerator(ring(), replace(options, composition={"Cu": 0.99, "Au": 0.01}))
    with pytest.raises(ValueError, match="contains no clusters"):
        SQSGenerator(ring(), replace(options, pair_cutoff=0.5))
    with pytest.raises(ValueError, match="max_clusters"):
        SQSGenerator(ring(), replace(options, max_clusters=3))


def test_evaluation_requires_same_ideal_geometry_and_composition(options):
    generator = SQSGenerator(ring(), options)
    result = generator.generate()
    changed = result.atoms.copy()
    changed.positions[0, 0] += 0.01
    with pytest.raises(ValueError, match="preserve template"):
        generator.evaluate(changed)
    changed = result.atoms.copy()
    changed.set_chemical_symbols(["Cu"] * 4)
    with pytest.raises(ValueError, match="composition"):
        generator.evaluate(changed)
    changed[0].symbol = "Ag"
    with pytest.raises(ValueError, match="outside this SQS basis"):
        generator.evaluate(changed)


def test_convenience_function_and_historical_import_path(options):
    from oncapintada.generator import SQS
    from oncapintada.generator import SQSGenerator as LegacyGenerator

    assert LegacyGenerator is SQSGenerator and SQS is SQSGenerator
    result = generate_sqs(
        ring(), {"Cu": 0.5, "Au": 0.5}, supercell=(1, 1, 1), pair_cutoff=1.01, seed=42
    )
    assert result.converged


def test_incremental_higher_order_swaps_and_rejection_restore_histograms():
    options = SQSOptions(
        {"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
        pair_cutoff=2.7,
        triplet_cutoff=2.7,
        quadruplet_cutoff=2.7,
        n_steps=1,
    )
    generator = SQSGenerator(bulk("Cu", "fcc", a=3.6, cubic=True), options)
    rng = np.random.default_rng(87)
    occupation = rng.permutation(np.repeat(np.arange(3), generator.species_counts))
    histogram = generator._histogram(occupation)
    for a, b in ((0, 1), (0, 31), (1, 5), (2, 6)):
        before_occupation, before_histogram = occupation.copy(), histogram.copy()
        touched = np.union1d(generator._at_site[a], generator._at_site[b])
        old = generator._bins(occupation, touched)
        occupation[a], occupation[b] = occupation[b], occupation[a]
        new = generator._bins(occupation, touched)
        np.add.at(histogram, old, -1)
        np.add.at(histogram, new, 1)
        np.testing.assert_array_equal(histogram, generator._histogram(occupation))
        # Simulate rejecting the move; integer counts must restore exactly.
        np.add.at(histogram, new, -1)
        np.add.at(histogram, old, 1)
        occupation[a], occupation[b] = occupation[b], occupation[a]
        np.testing.assert_array_equal(histogram, before_histogram)
        np.testing.assert_array_equal(occupation, before_occupation)


def test_sqs_can_supply_ce_features_on_the_same_supercell():
    from oncapintada import ClusterExpansionGenerator, ClusterExpansionOptions

    parent = bulk("Cu", "fcc", a=3.6, cubic=True)
    sqs = generate_sqs(
        parent, {"Cu": 0.5, "Au": 0.25, "Ag": 0.25}, pair_cutoff=3.7, n_steps=1000, seed=42
    )
    ce = ClusterExpansionGenerator(
        parent,
        ClusterExpansionOptions(
            active_element="Cu",
            species=("Cu", "Au", "Ag"),
            supercell=(2, 2, 2),
            pair_cutoff=3.7,
        ),
    )
    fingerprint = ce.correlation(sqs.atoms)
    np.testing.assert_array_equal(fingerprint[:3], [16, 8, 8])
    assert len(fingerprint) == len(ce.correlation_labels)
    for orbit in sqs.orbits:
        assert orbit.order == 2


def test_partial_periodicity_and_skew_cell():
    # The short lattice vector gives two NN images per site in this 4-site ring.
    parent = Atoms(
        "Cu4",
        scaled_positions=[[i / 4, 0, 0] for i in range(4)],
        cell=[[4, 0, 0], [1, 8, 0], [0, 0, 9]],
        pbc=[True, False, False],
    )
    generator = SQSGenerator(
        parent, SQSOptions({"Cu": 0.5, "Au": 0.5}, supercell=(1, 1, 1), pair_cutoff=1.01)
    )
    result = generator.generate()
    assert result.objective == 0
    np.testing.assert_allclose(result.atoms.cell.array, parent.cell.array)
    np.testing.assert_array_equal(result.atoms.pbc, parent.pbc)
