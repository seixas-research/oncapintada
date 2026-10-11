"""Regression tests for numerical and validation defects found during review."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk
from scipy.integrate import trapezoid

from oncapintada import BinaryAlloy, BondsCounter, DisorderedAlloyGenerator, MultiComponentAlloy
from oncapintada.bonds_counter import BondCountResult
from oncapintada.bonds_model import BondsModel
from oncapintada.constants import R, h, kJmol
from oncapintada.phase_diagram import PhaseDiagram
from oncapintada.qca import QCABinary
from oncapintada.vibrational import Vibrational


def test_package_import_is_silent_and_does_not_hide_user_warnings():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import warnings; warnings.simplefilter("always"); '
            'import oncapintada; warnings.warn("visible warning", UserWarning)',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
    assert "visible warning" in result.stderr


def test_all_modules_import():
    source = Path(__file__).resolve().parents[1] / "src" / "oncapintada"
    for module in source.glob("*.py"):
        importlib.import_module(f"oncapintada.{module.stem}")


def test_banner_without_user_environment(monkeypatch, capsys):
    import oncapintada

    monkeypatch.delenv("USER", raising=False)
    oncapintada.banner()
    assert "Onça-pintada" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf, [0.1, 0.2, 1.5]])
def test_binary_entropy_rejects_invalid_composition(bad):
    with pytest.raises(ValueError):
        BinaryAlloy().configurational_entropy(bad)


def test_entropy_exact_endpoints_and_list_energy_input():
    binary = BinaryAlloy([[1, 1.1], [2.1, 2]], dilution=0.1)
    np.testing.assert_array_equal(binary.configurational_entropy([0, 1]), [0, 0])
    ternary = MultiComponentAlloy(np.eye(3))
    np.testing.assert_array_equal(ternary.configurational_entropy(np.eye(3)), [0, 0, 0])
    assert ternary.enthalpy_of_mixing([0, 0, 1]) == 0


def test_energy_matrix_is_copied_and_binary_dimension_validated():
    energies = np.eye(2)
    model = BinaryAlloy(energies)
    energies[0, 0] = 99
    assert model.energy_matrix[0, 0] == 1
    with pytest.raises(ValueError, match="2 by 2"):
        BinaryAlloy(np.eye(3))
    with pytest.raises(ValueError, match="finite"):
        MultiComponentAlloy([[1, np.inf], [0, 1]])


@pytest.mark.parametrize("temperature", [-1, np.inf, np.nan, [[1, 2]]])
def test_gibbs_invalid_temperatures(temperature):
    with pytest.raises(ValueError, match="Temperatures"):
        BinaryAlloy(np.eye(2)).gibbs_free_energy_of_mixing(0.5, temperature)


def test_periodic_fcc_bonds_retain_all_twelve_neighbors():
    primitive = bulk("Cu", "fcc", a=3.6)
    result = BondsCounter(atoms=primitive, cutoff=2.6).count_bonds()
    assert result.total == 6
    assert result.counts == {("Cu", "Cu"): 6}
    conventional = primitive.repeat((2, 2, 2))
    assert BondsCounter(atoms=conventional, cutoff=2.6).count_bonds().total == 6 * len(conventional)


def test_periodic_binary_chain_counts_multiple_images():
    chain = Atoms(
        "CuAu", positions=[[0, 0, 0], [1, 0, 0]], cell=[2, 5, 5], pbc=[True, False, False]
    )
    result = BondsCounter(atoms=chain, cutoff=1.1).count_bonds()
    assert result.total == 2 and result.counts[("Au", "Cu")] == 2


def test_subset_updates_target_pairs_and_multi_letter_symbols():
    atoms = Atoms("CuAu", positions=[[0, 0, 0], [1, 0, 0]])
    counter = BondsCounter(atoms=atoms)
    counter.set_subset_from_symbols("Cu")
    assert counter.target_pairs == {("Cu", "Cu")}
    counter.set_subset([1])
    assert counter.count_bonds().counts == {("Au", "Au"): 0}
    counter.set_subset([])
    assert counter.count_bonds().total == 0
    assert counter.bonds_dataframe()["pair"].tolist() == ["TOTAL"]


def test_empty_bond_results_are_valid_tables():
    table = BondCountResult({}, 0).to_dataframe()
    assert table["count"].tolist() == [0]
    assert table["probability"].tolist() == [0]


@pytest.mark.parametrize("subset", [[-1], [3], [0.5], [[0]]])
def test_invalid_bond_subset(subset):
    with pytest.raises((TypeError, ValueError)):
        BondsCounter(atoms=Atoms("Cu2"), subset_indices=subset)


def test_warren_cowley_uses_species_conditional_neighbors():
    # Au has degree 1, Cu has degree 2 on this open three-site chain.
    atoms = Atoms("AuCuCu", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    table = BondsCounter(atoms=atoms, cutoff=1.1).warren_cowley_parameters()
    alpha = dict(zip(table["pair"], table["warren_cowley"]))
    assert alpha["Au-Cu"] == pytest.approx(1 - 1 / (2 / 3))
    assert alpha["Cu-Cu"] == pytest.approx(1 - (2 / 3) / (2 / 3))


def test_disordered_alloy_accepts_atoms_and_rounds_exact_counts():
    parent = Atoms("Cu5O", positions=np.zeros((6, 3)))
    generator = DisorderedAlloyGenerator(parent, "Cu", ["Cu", "Au", "Ag"], [1, 1, 1], seed=42)
    first = generator.generate_configuration(seed=12)
    second = generator.generate_configuration(seed=12)
    assert first.atoms.get_chemical_symbols() == second.atoms.get_chemical_symbols()
    assert first.site_table["new_symbol"].value_counts().to_dict() == {"Cu": 2, "Au": 2, "Ag": 1}
    assert first.atoms[-1].symbol == "O" and parent[0].symbol == "Cu"
    assert len(generator.generate_multiple(3, base_seed=42)) == 3


@pytest.mark.parametrize(
    "mask, species, concentration",
    [
        ([0, 0], ["Cu", "Au"], [0.5, 0.5]),
        ([[0, 1]], ["Cu", "Au"], [0.5, 0.5]),
        ([0, 1], ["Cu", "Au"], [-1, 2]),
        ([0, 1], ["Cu", "Au"], [np.nan, 1]),
        ([0, 1], ["Cu", "Cu"], [0.5, 0.5]),
        ([True], ["Cu", "Au"], [0.5, 0.5]),
        ([-1], ["Cu", "Au"], [0.5, 0.5]),
    ],
)
def test_disordered_alloy_validation(mask, species, concentration):
    with pytest.raises((ValueError, TypeError)):
        DisorderedAlloyGenerator(Atoms("Cu2"), mask, species, concentration)


@pytest.mark.parametrize("size", [(1, 1), (1, 3), (2, 2), (3, 4)])
def test_square_lattice_energy_is_extensive_even_with_self_images(size):
    model = BondsModel(*size, species=["Cu"], concentrations=[1], J=2)
    assert model.bond_energy() == -4 * model.N
    assert model.site_energies().sum() == model.bond_energy()


def test_square_lattice_honors_open_boundaries():
    model = BondsModel(3, 2, species=["Cu"], concentrations=[1], J=1, pbc=(False, False, False))
    assert model.bond_energy() == -(2 * 2 + 3 * 1)
    assert np.count_nonzero(model.neighbors == -1) == 10


def test_square_lattice_rejects_asymmetric_interactions():
    matrix = pd.DataFrame([[0, 1], [2, 0]], index=["Cu", "Au"], columns=["Cu", "Au"])
    with pytest.raises(ValueError, match="symmetric"):
        BondsModel(2, 2, species=["Cu", "Au"], J=matrix)


def test_qca_random_mixing_is_exact_and_methods_are_lazy():
    x = np.array([0, 0.25, 0.5, 0.75, 1])
    qca = QCABinary(12, pd.DataFrame(np.zeros((5, 2)), index=x, columns=[300, 1000]))
    gibbs = qca.get_gibbs_free_energy_of_mixing()
    ab, aa, bb = qca.get_probability()
    np.testing.assert_allclose(ab[300], 2 * x * (1 - x))
    np.testing.assert_allclose(aa[300], x**2)
    np.testing.assert_allclose(bb[300], (1 - x) ** 2)
    expected_entropy = BinaryAlloy().configurational_entropy(x)
    np.testing.assert_allclose(gibbs[300], -300 * expected_entropy, atol=1e-12)
    np.testing.assert_allclose(qca.get_entropy_of_mixing().iloc[[0, -1]], 0, atol=0)
    alpha = qca.get_warren_cowley_parameters()
    assert np.isnan(alpha[0].iloc[0, 0])
    np.testing.assert_allclose(alpha[0].iloc[1:-1], 0, atol=1e-15)


@pytest.mark.parametrize("interaction", [-10000, -1, 0, 1, 10000])
def test_qca_ordering_limits_and_probability_conservation(interaction):
    x = np.array([0, 0.1, 0.5, 0.9, 1])
    table = pd.DataFrame(interaction * x * (1 - x), index=x, columns=[1])
    qca = QCABinary(4, table)
    ab, aa, bb = [p.values for p in qca.get_probability()]
    assert np.isfinite(ab + aa + bb).all()
    np.testing.assert_allclose(ab + aa + bb, 1)
    assert min(ab.min(), aa.min(), bb.min()) >= 0
    if interaction == -10000:
        np.testing.assert_allclose(ab[:, 0], 2 * np.minimum(x, 1 - x))
    if interaction == 10000:
        np.testing.assert_array_equal(ab, 0)


def test_qca_rejects_zero_temperature():
    with pytest.raises(ValueError, match="positive"):
        QCABinary(4, pd.DataFrame([[0]], index=[0.5], columns=[0]))


def test_phase_diagram_lower_envelope_finds_gap_and_convex_data_has_none():
    x = np.linspace(0, 1, 101)
    double_well = (x - 0.25) ** 2 * (x - 0.75) ** 2
    phase = PhaseDiagram(pd.DataFrame({100: double_well, 200: (x - 0.5) ** 2}, index=x))
    result = phase.binodal_curve()
    np.testing.assert_allclose(result["x"], [0.25, 0.75])
    assert result["t"].tolist() == [100, 100]
    critical = phase.critical_point(curve="binodal")
    assert critical == {"x_c": 0.5, "t_c": 100}
    convex = PhaseDiagram(pd.DataFrame({100: x**2}, index=x))
    assert convex.binodal_curve().empty
    assert np.isnan(convex.critical_point()["t_c"])


def test_phase_diagram_resolves_narrow_gap_using_all_samples():
    x = np.linspace(0, 1, 101)
    energy = x**2
    energy[49:52] += 1
    phase = PhaseDiagram(pd.DataFrame({100: energy}, index=x))
    np.testing.assert_allclose(phase.binodal_curve(ninterp=2)["x"], [0.48, 0.52])


def test_phase_derivatives_on_nonuniform_grid():
    x = np.array([0, 0.05, 0.2, 0.55, 1.0])
    phase = PhaseDiagram(pd.DataFrame({100: x**2}, index=x))
    np.testing.assert_allclose(phase.dGdx()[100], 2 * x, atol=1e-12)
    np.testing.assert_allclose(phase.d2Gdx2()[100], 2, atol=1e-12)


@pytest.mark.parametrize("x", [[0, 1], [0, 0, 1], [1, 0.5, 0], [0, np.nan, 1]])
def test_phase_diagram_rejects_invalid_composition_axis(x):
    with pytest.raises(ValueError):
        PhaseDiagram(pd.DataFrame({100: np.zeros(len(x))}, index=x))


@pytest.fixture
def dos():
    nu = np.linspace(1, 10, 501)
    return pd.DataFrame({"Frequency": nu, "VDOS": np.ones(len(nu)) / 3})


def test_vibrational_zero_temperature_and_temperature_arrays(dos):
    vibration = Vibrational(dos, "THz")
    assert vibration.number_of_modes == pytest.approx(3)
    zero_point = trapezoid(dos["VDOS"] * h * dos["Frequency"] / 2, dos["Frequency"]) * kJmol
    assert vibration.get_vibrational_enthalpy(0) == pytest.approx(zero_point)
    assert vibration.get_vibrational_entropy(0) == 0
    temperature = np.array([0, 100, 300, 1000])
    enthalpy = vibration.get_vibrational_enthalpy(temperature)
    entropy = vibration.get_vibrational_entropy(temperature)
    assert enthalpy.shape == entropy.shape == (4,)
    assert np.all(np.diff(enthalpy) > 0)
    assert np.all(np.diff(entropy) > 0)
    np.testing.assert_allclose(
        vibration.get_vibrational_free_energy(temperature), enthalpy - temperature * entropy
    )
    assert vibration.get_vibrational_free_energy(500) == pytest.approx(
        Vibrational(dos, "THz").get_vibrational_free_energy(500)
    )


def test_vibrational_units_agree(dos):
    from oncapintada.constants import cm_to_THz, eV_to_THz

    values = []
    for unit, scale in [("THz", 1), ("cm^-1", cm_to_THz), ("eV", eV_to_THz)]:
        converted = pd.DataFrame(
            {"Frequency": dos["Frequency"] / scale, "VDOS": dos["VDOS"] * scale}
        )
        vibration = Vibrational(converted, unit)
        values.append(vibration.get_vibrational_free_energy([0, 300, 1000]))
        assert vibration.number_of_modes == pytest.approx(3)
    np.testing.assert_allclose(values, np.broadcast_to(values[0], (3, 3)), atol=1e-12)


def test_vibrational_high_temperature_limit_and_missing_dos(dos):
    vibration = Vibrational(dos, "THz")
    temperature = 1e7
    assert vibration.get_vibrational_enthalpy(temperature) / temperature == pytest.approx(
        3 * R, rel=1e-7
    )
    empty = Vibrational()
    with pytest.raises(ValueError, match="VDOS is not set"):
        empty.get_vibrational_free_energy(300)
    with pytest.raises(ValueError, match="non-negative"):
        vibration.get_vibrational_entropy(-1)


def test_spinodal_retains_multiple_instability_regions():
    x = np.linspace(0, 1, 601)
    phase = PhaseDiagram(pd.DataFrame({300: np.sin(3 * np.pi * x) ** 2}, index=x))
    roots = phase.spinodal_curve()
    np.testing.assert_allclose(roots["x"], np.arange(1, 12, 2) / 12, atol=1e-4)
    flat = PhaseDiagram(pd.DataFrame({300: np.zeros_like(x)}, index=x))
    assert flat.spinodal_curve().empty


def test_lattice_plot_uses_requested_image_count_and_short_periodic_bonds():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = BondsModel(2, 2, species=["Cu"], concentrations=[1])
    axes = model.plot_lattice(n_images=(2, 2))
    assert sum(len(collection.get_offsets()) for collection in axes.collections) == 4 * model.N
    assert len(axes.lines) == 2 * model.N
    for line in axes.lines:
        dx = np.diff(line.get_xdata())[0]
        dy = np.diff(line.get_ydata())[0]
        assert np.hypot(dx, dy) == pytest.approx(model.a)
    plt.close(axes.figure)


def test_unimplemented_monte_carlo_fails_explicitly():
    from oncapintada.monte_carlo import MDMC, BasinHopping, ChemicalSwapMC

    for algorithm in (BasinHopping, ChemicalSwapMC, MDMC):
        with pytest.raises(NotImplementedError, match="not implemented"):
            algorithm()


def test_dos_refresh_updates_units_modes_and_thermal_state(monkeypatch):
    from unittest.mock import MagicMock

    import oncapintada.vibrational as module

    sampled = MagicMock()
    sampled.get_energies.return_value = np.linspace(0.01, 0.03, 21)
    sampled.get_weights.return_value = np.ones(21) * 100
    phonons = MagicMock()
    phonons.get_dos.return_value.sample_grid.return_value = sampled
    monkeypatch.setattr(module, "Phonons", MagicMock(return_value=phonons))
    monkeypatch.setattr(module, "BFGS", MagicMock())
    atoms = bulk("Cu")
    original = atoms.copy()
    vibration = Vibrational(frequency_unit="THz")
    table = vibration.get_vdos(atoms, MagicMock())
    assert vibration.number_of_modes == pytest.approx(2)
    assert vibration.get_vibrational_enthalpy(0) == pytest.approx(2 * 0.02 / 2 * kJmol)
    np.testing.assert_array_equal(atoms.positions, original.positions)
    assert atoms.calc is None
    phonons.get_dos.return_value.sample_grid.assert_called_once()
    assert table["Frequency"].iloc[0] == pytest.approx(0.01 / h)
    # Refresh after a prior thermal evaluation must invalidate cached state.
    vibration.get_vdos(atoms, MagicMock())
    assert vibration.vibrational_enthalpy is None
