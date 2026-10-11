import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from pandas.testing import assert_frame_equal

from oncapintada.bonds_counter import BondsCounter  # Onça Pintada


@pytest.fixture  # fixture to create a simple water molecule
def water_atoms() -> Atoms:
    return Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.74], [0.96, 0, 0]])


def test_bondcounter_init_requires_atoms_or_path():
    with pytest.raises(ValueError, match="Provide `atoms` or `xyz_path`."):
        BondsCounter()


def test_bondcounter_init_with_atoms_sets_full_subset_and_target_pairs(water_atoms):
    counter = BondsCounter(atoms=water_atoms)
    assert np.array_equal(counter._subset_mask, np.ones(len(water_atoms), dtype=bool))
    assert counter.target_pairs == {("H", "H"), ("H", "O"), ("O", "O")}


def test_bondcounter_init_subset_symbols_filters_mask_and_pairs(water_atoms):
    counter = BondsCounter(atoms=water_atoms, subset_symbols=["H"])
    assert np.array_equal(counter._subset_mask, np.array([True, True, False]))
    assert counter.target_pairs == {("H", "H")}


def test_bondcounter_init_subset_indices_filters_mask(water_atoms):
    counter = BondsCounter(atoms=water_atoms, subset_indices=[0, 2])
    assert np.array_equal(counter._subset_mask, np.array([True, False, True]))
    assert counter.target_pairs == {("H", "H"), ("H", "O"), ("O", "O")}


def test_set_subset_none_resets_to_all_atoms(water_atoms):
    counter = BondsCounter(atoms=water_atoms, subset_indices=[0])
    counter.set_subset(None)
    assert np.array_equal(counter._subset_mask, np.ones(len(water_atoms), dtype=bool))


def test_set_subset_with_iterable(water_atoms):
    counter = BondsCounter(atoms=water_atoms)
    counter.set_subset([0, 2])
    assert np.array_equal(counter._subset_mask, np.array([True, False, True]))


def test_set_subset_with_numpy_array_and_duplicates(water_atoms):
    counter = BondsCounter(atoms=water_atoms)
    counter.set_subset(np.array([1, 1]))
    assert np.array_equal(counter._subset_mask, np.array([False, True, False]))


def _atoms_from_symbols(symbols):
    positions = [(float(i), 0.0, 0.0) for i in range(len(symbols))]
    return Atoms(symbols=symbols, positions=positions)


def test_counter_concentrations_full_structure():
    atoms = _atoms_from_symbols(["C", "O", "O", "H", "H"])
    counter = BondsCounter(atoms=atoms, cutoff=1.0)
    result = counter.counter_concentrations().sort_values("element").reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "element": ["C", "H", "O"],
            "count": [1, 2, 2],
            "concentration": [0.2, 0.4, 0.4],
        }
    )
    assert_frame_equal(result, expected)


def test_counter_concentrations_subset_indices():
    atoms = _atoms_from_symbols(["C", "O", "H", "O"])
    counter = BondsCounter(atoms=atoms, cutoff=1.0, subset_indices=[1, 3])
    result = counter.counter_concentrations().reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "element": ["O"],
            "count": [2],
            "concentration": [1.0],
        }
    )
    assert_frame_equal(result, expected)


def test_warren_cowley_parameters_balanced_structure():
    atoms = Atoms(
        symbols=["Cu", "Zn", "Cu", "Zn"],
        positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 2.0),
            (0.0, 2.0, 0.0),
            (0.0, 2.0, 2.0),
        ],
        cell=[10, 10, 10],
        pbc=False,
    )
    counter = BondsCounter(atoms=atoms, cutoff=2.1)
    df = counter.warren_cowley_parameters()
    result = dict(zip(df["pair"], df["warren_cowley"]))
    assert pytest.approx(0.0) == result["Cu-Cu"]
    assert pytest.approx(0.0) == result["Cu-Zn"]
    assert pytest.approx(0.0) == result["Zn-Zn"]


def test_warren_cowley_parameters_no_bonds_returns_one():
    atoms = Atoms(
        symbols=["Cu", "Cu"],
        positions=[
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        ],
        cell=[10, 10, 10],
        pbc=False,
    )
    counter = BondsCounter(atoms=atoms, cutoff=1.0)
    df = counter.warren_cowley_parameters()
    assert list(df["pair"]) == ["Cu-Cu"]
    assert pytest.approx(1.0) == df.loc[0, "warren_cowley"]
