import numpy as np
import pytest

from oncapintada.constants import R, kJmol
from oncapintada.subregular_model import BinaryAlloy, MultiComponentAlloy

# ──────────────────────────────── Fixtures ────────────────────────────────────


@pytest.fixture
def simple_energy_matrix():
    return np.array([[1.0, 1.1], [2.1, 2.0]])


@pytest.fixture
def composition():
    return np.linspace(0, 1, 11)


@pytest.fixture
def simple_alloy(simple_energy_matrix):
    return BinaryAlloy(energy_matrix=simple_energy_matrix, dilution=0.10)


@pytest.fixture
def symmetric_energy_matrix():
    return np.array([[1.0, 1.5], [1.5, 2.0]])


@pytest.fixture
def ternary_energy_matrix():
    return np.array(
        [
            [1.0, 1.1, 1.2],
            [2.1, 2.0, 2.2],
            [3.2, 3.1, 3.0],
        ]
    )


@pytest.fixture
def ternary_alloy(ternary_energy_matrix):
    return MultiComponentAlloy(energy_matrix=ternary_energy_matrix, dilution=0.0)


# ─────────────────────────── BinaryAlloy – Init ───────────────────────────────


class TestBinaryAlloyInit:
    def test_initialization(self, simple_energy_matrix, simple_alloy):
        assert np.array_equal(simple_alloy.energy_matrix, simple_energy_matrix)
        assert simple_alloy.dilution == 0.10

    def test_default_initialization(self):
        alloy = BinaryAlloy()
        assert alloy.energy_matrix is None
        assert alloy.dilution == 0.0

    def test_non_square_energy_matrix_raises(self):
        with pytest.raises(ValueError, match="square"):
            BinaryAlloy(energy_matrix=np.array([[1.0, 1.1, 1.2], [2.1, 2.0, 2.2]]))

    def test_dilution_below_zero_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            BinaryAlloy(dilution=-0.1)

    def test_dilution_above_one_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            BinaryAlloy(dilution=1.1)

    def test_dilution_boundary_values(self, simple_energy_matrix):
        alloy0 = BinaryAlloy(energy_matrix=simple_energy_matrix, dilution=0.0)
        alloy1 = BinaryAlloy(energy_matrix=simple_energy_matrix, dilution=1.0)
        assert alloy0.dilution == 0.0
        assert alloy1.dilution == 1.0


# ────────────────────────── BinaryAlloy – Properties ─────────────────────────


class TestBinaryAlloyProperties:
    def test_get_energy_matrix(self, simple_alloy, simple_energy_matrix):
        assert np.array_equal(simple_alloy.energy_matrix, simple_energy_matrix)

    def test_set_energy_matrix(self, simple_alloy):
        new_matrix = np.array([[2.0, 2.5], [3.5, 3.0]])
        simple_alloy.energy_matrix = new_matrix
        assert np.array_equal(simple_alloy.energy_matrix, new_matrix)

    def test_set_non_square_energy_matrix_raises(self, simple_alloy):
        with pytest.raises(ValueError, match="square"):
            simple_alloy.energy_matrix = np.array([[1.0, 1.1, 1.2], [2.1, 2.0, 2.2]])

    def test_get_dilution(self, simple_alloy):
        assert simple_alloy.dilution == 0.10

    def test_set_dilution(self, simple_alloy):
        simple_alloy.dilution = 0.5
        assert simple_alloy.dilution == 0.5

    def test_set_dilution_out_of_range_raises(self, simple_alloy):
        with pytest.raises(ValueError, match="between 0 and 1"):
            simple_alloy.dilution = 1.5


# ──────────────────────────── BinaryAlloy – Mij ──────────────────────────────


class TestBinaryAlloyMij:
    def test_Mij_shape(self, simple_alloy):
        assert simple_alloy.Mij().shape == (2, 2)

    def test_Mij_diagonal_is_zero(self, simple_alloy):
        M = simple_alloy.Mij()
        assert np.allclose(np.diag(M), 0.0)

    def test_Mij_values(self, simple_alloy):
        # M_ij = E_ij - (x0 * E_ii + (1-x0) * E_jj), i = solute (row), j = solvent (column)
        M = simple_alloy.Mij()
        expected = np.array([[0.0, -0.8], [1.0, 0.0]])
        assert np.allclose(M, expected)

    def test_Mij_no_energy_matrix_raises(self):
        with pytest.raises(ValueError, match="not set"):
            BinaryAlloy().Mij()


# ────────────────────── BinaryAlloy – Enthalpy of Mixing ─────────────────────


class TestBinaryAlloyEnthalpy:
    def test_enthalpy_length(self, simple_alloy, composition):
        H = simple_alloy.enthalpy_of_mixing(composition)
        assert len(H) == len(composition)

    def test_enthalpy_zero_at_pure_components(self, simple_alloy):
        x = np.array([0.0, 1.0])
        H = simple_alloy.enthalpy_of_mixing(x)
        assert np.allclose(H, 0.0)

    def test_enthalpy_midpoint_value(self, simple_alloy):
        x = np.array([0.5])
        H_eV = simple_alloy.enthalpy_of_mixing(x, unit="eV/atom")
        M = simple_alloy.Mij()
        expected = (M[0, 1] * 0.5 + M[1, 0] * 0.5) * 0.5 * 0.5
        assert np.allclose(H_eV, expected)

    def test_enthalpy_matches_subregular_formula(self, simple_alloy):
        # H(x) = (M_01 * x + M_10 * (1-x)) * x * (1-x), with x the fraction of B
        x = np.linspace(0, 1, 11)
        H_eV = simple_alloy.enthalpy_of_mixing(x, unit="eV/atom")
        M = simple_alloy.Mij()
        expected = (M[0, 1] * x + M[1, 0] * (1 - x)) * x * (1 - x)
        assert np.allclose(H_eV, expected)

    def test_enthalpy_unit_conversion_kJmol_vs_eVatom(self, simple_alloy, composition):
        H_kJ = simple_alloy.enthalpy_of_mixing(composition, unit="kJ/mol")
        H_eV = simple_alloy.enthalpy_of_mixing(composition, unit="eV/atom")
        assert np.allclose(H_kJ, H_eV * kJmol)

    def test_enthalpy_invalid_unit_raises(self, simple_alloy, composition):
        with pytest.raises(ValueError, match="Invalid unit"):
            simple_alloy.enthalpy_of_mixing(composition, unit="J/mol")


# ──────────────────── BinaryAlloy – Configurational Entropy ──────────────────


class TestBinaryAlloyEntropy:
    def test_entropy_length(self, simple_alloy, composition):
        S = simple_alloy.configurational_entropy(composition)
        assert len(S) == len(composition)

    def test_entropy_non_negative(self, simple_alloy, composition):
        S = simple_alloy.configurational_entropy(composition)
        assert np.all(S >= 0)

    def test_entropy_max_at_equimolar(self, simple_alloy):
        x = np.linspace(0.01, 0.99, 199)
        S = simple_alloy.configurational_entropy(x)
        idx_max = np.argmax(S)
        assert np.isclose(x[idx_max], 0.5, atol=0.02)

    def test_entropy_midpoint_value(self, simple_alloy):
        x = np.array([0.5])
        S = simple_alloy.configurational_entropy(x, unit="kJ/(mol*K)")
        assert np.allclose(S, R * np.log(2))

    def test_entropy_unit_conversion_kJmol_vs_eVatom(self, simple_alloy, composition):
        S_kJ = simple_alloy.configurational_entropy(composition, unit="kJ/(mol*K)")
        S_eV = simple_alloy.configurational_entropy(composition, unit="eV/(atom*K)")
        assert np.allclose(S_kJ, S_eV * kJmol)

    def test_entropy_invalid_unit_raises(self, simple_alloy, composition):
        with pytest.raises(ValueError, match="Invalid unit"):
            simple_alloy.configurational_entropy(composition, unit="J/(mol*K)")


# ─────────────────── BinaryAlloy – Gibbs Free Energy of Mixing ───────────────


class TestBinaryAlloyGibbs:
    def test_gibbs_shape(self, simple_alloy, composition):
        t = np.array([300.0, 600.0, 900.0])
        G = simple_alloy.gibbs_free_energy_of_mixing(composition, t)
        assert G.shape == (len(composition), len(t))

    def test_gibbs_equals_enthalpy_at_zero_temperature(self, simple_alloy, composition):
        t = np.array([0.0])
        G = simple_alloy.gibbs_free_energy_of_mixing(composition, t)
        H = simple_alloy.enthalpy_of_mixing(composition, unit="kJ/mol")
        assert np.allclose(G[:, 0], H)

    def test_gibbs_decreases_with_temperature_at_midpoint(self, simple_alloy):
        x = np.array([0.5])
        t_low = np.array([100.0])
        t_high = np.array([1000.0])
        G_low = simple_alloy.gibbs_free_energy_of_mixing(x, t_low)
        G_high = simple_alloy.gibbs_free_energy_of_mixing(x, t_high)
        assert G_high[0, 0] < G_low[0, 0]

    def test_gibbs_matches_formula(self, simple_alloy):
        x = np.linspace(0.01, 0.99, 20)
        t = np.array([300.0, 1000.0])
        G = simple_alloy.gibbs_free_energy_of_mixing(x, t)
        H = simple_alloy.enthalpy_of_mixing(x, unit="kJ/mol")
        S = simple_alloy.configurational_entropy(x, unit="kJ/(mol*K)")
        for j, temp in enumerate(t):
            assert np.allclose(G[:, j], H - temp * S)

    def test_gibbs_invalid_unit_raises(self, simple_alloy, composition):
        t = np.array([300.0])
        with pytest.raises(ValueError, match="Invalid unit"):
            simple_alloy.gibbs_free_energy_of_mixing(composition, t, unit="J/mol")


# ─────────────────── MultiComponentAlloy – Init ──────────────────────────────


class TestMultiComponentAlloyInit:
    def test_initialization(self, ternary_energy_matrix, ternary_alloy):
        assert np.array_equal(ternary_alloy.energy_matrix, ternary_energy_matrix)
        assert ternary_alloy.dilution == 0.0

    def test_default_initialization(self):
        alloy = MultiComponentAlloy()
        assert alloy.energy_matrix is None
        assert alloy.dilution == 0.0

    def test_non_square_matrix_raises(self):
        with pytest.raises(ValueError, match="square"):
            MultiComponentAlloy(energy_matrix=np.ones((2, 3)))

    def test_invalid_dilution_raises(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            MultiComponentAlloy(dilution=2.0)


# ─────────────────── MultiComponentAlloy – Properties ────────────────────────


class TestMultiComponentAlloyProperties:
    def test_get_energy_matrix(self, ternary_alloy, ternary_energy_matrix):
        assert np.array_equal(ternary_alloy.energy_matrix, ternary_energy_matrix)

    def test_set_energy_matrix(self, ternary_alloy):
        new_matrix = np.eye(3)
        ternary_alloy.energy_matrix = new_matrix
        assert np.array_equal(ternary_alloy.energy_matrix, new_matrix)

    def test_set_non_square_matrix_raises(self, ternary_alloy):
        with pytest.raises(ValueError, match="square"):
            ternary_alloy.energy_matrix = np.ones((2, 3))


# ─────────────────── MultiComponentAlloy – simplex_grid ─────────────────────


class TestSimplexGrid:
    def test_simplex_grid_columns_equal_components(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid(resolution=5)
        assert grid.shape[1] == 3

    def test_simplex_grid_rows_sum_to_one(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid(resolution=5)
        assert np.allclose(grid.sum(axis=1), 1.0)

    def test_simplex_grid_non_negative(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid(resolution=5)
        assert np.all(grid >= 0)

    def test_simplex_grid_uses_matrix_size_when_n_is_none(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid()
        assert grid.shape[1] == 3

    def test_simplex_grid_no_matrix_no_n_raises(self):
        alloy = MultiComponentAlloy()
        with pytest.raises(ValueError, match="Energy matrix must be set"):
            alloy.simplex_grid()

    def test_simplex_grid_n_less_than_2_raises(self, ternary_alloy):
        with pytest.raises(ValueError, match="at least 2"):
            ternary_alloy.simplex_grid(n_components=1)

    def test_simplex_grid_non_integer_resolution_raises(self, ternary_alloy):
        with pytest.raises(TypeError, match="integer"):
            ternary_alloy.simplex_grid(resolution=5.5)

    def test_simplex_grid_zero_resolution_raises(self, ternary_alloy):
        with pytest.raises(ValueError, match="positive integer"):
            ternary_alloy.simplex_grid(resolution=0)


# ──────────────────── MultiComponentAlloy – line_profile ────────────────────


class TestLineProfile:
    def test_line_profile_shape(self, ternary_alloy):
        y1 = np.array([1.0, 0.0, 0.0])
        y2 = np.array([0.0, 1.0, 0.0])
        line = ternary_alloy.line_profile(npoints=11, y1=y1, y2=y2)
        assert line.shape == (11, 3)

    def test_line_profile_endpoints(self, ternary_alloy):
        y1 = np.array([1.0, 0.0, 0.0])
        y2 = np.array([0.0, 1.0, 0.0])
        line = ternary_alloy.line_profile(npoints=11, y1=y1, y2=y2)
        assert np.allclose(line[0], y1)
        assert np.allclose(line[-1], y2)

    def test_line_profile_rows_sum_to_one(self, ternary_alloy):
        y1 = np.array([1.0, 0.0, 0.0])
        y2 = np.array([0.0, 0.5, 0.5])
        line = ternary_alloy.line_profile(npoints=7, y1=y1, y2=y2)
        assert np.allclose(line.sum(axis=1), 1.0)

    def test_line_profile_non_array_raises(self, ternary_alloy):
        with pytest.raises(ValueError, match="numpy arrays"):
            ternary_alloy.line_profile(npoints=5, y1=[1.0, 0.0, 0.0], y2=np.array([0.0, 1.0, 0.0]))

    def test_line_profile_y1_not_sum_to_one_raises(self, ternary_alloy):
        with pytest.raises(ValueError, match="sum to 1"):
            ternary_alloy.line_profile(
                npoints=5, y1=np.array([0.5, 0.0, 0.0]), y2=np.array([0.0, 1.0, 0.0])
            )

    def test_line_profile_negative_composition_raises(self, ternary_alloy):
        with pytest.raises(ValueError, match="non-negative"):
            ternary_alloy.line_profile(
                npoints=5, y1=np.array([1.2, -0.2, 0.0]), y2=np.array([0.0, 1.0, 0.0])
            )

    def test_line_profile_npoints_less_than_2_raises(self, ternary_alloy):
        with pytest.raises(ValueError, match="at least 2"):
            ternary_alloy.line_profile(
                npoints=1, y1=np.array([1.0, 0.0, 0.0]), y2=np.array([0.0, 1.0, 0.0])
            )

    def test_line_profile_mismatched_shapes_raises(self, ternary_alloy):
        with pytest.raises(ValueError):
            ternary_alloy.line_profile(
                npoints=5, y1=np.array([1.0, 0.0]), y2=np.array([0.0, 1.0, 0.0])
            )


# ─────────────────── MultiComponentAlloy – Mij ────────────────────────────


class TestMultiComponentAlloyMij:
    def test_Mij_shape(self, ternary_alloy):
        assert ternary_alloy.Mij().shape == (3, 3)

    def test_Mij_diagonal_is_zero(self, ternary_alloy):
        M = ternary_alloy.Mij()
        assert np.allclose(np.diag(M), 0.0)

    def test_Mij_solute_solvent_reference(self, ternary_energy_matrix):
        # With i = solute (row) and j = solvent (column):
        # M_ij = E_ij - (x0 * E_ii + (1-x0) * E_jj)
        x0 = 0.1
        alloy = MultiComponentAlloy(energy_matrix=ternary_energy_matrix, dilution=x0)
        M = alloy.Mij()
        E = ternary_energy_matrix
        for i in range(3):
            for j in range(3):
                expected = E[i, j] - (x0 * E[i, i] + (1 - x0) * E[j, j])
                assert np.isclose(M[i, j], expected)


# ─────────────────── MultiComponentAlloy – Enthalpy of Mixing ────────────────


class TestMultiComponentAlloyEnthalpy:
    def test_enthalpy_length_matches_grid(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid(resolution=5)
        H = ternary_alloy.enthalpy_of_mixing(grid)
        assert len(H) == len(grid)

    def test_enthalpy_pure_components_near_zero(self, ternary_alloy):
        pure = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        H = ternary_alloy.enthalpy_of_mixing(pure)
        assert np.allclose(H, 0.0, atol=1e-6)

    def test_enthalpy_1d_input_returns_scalar(self, ternary_alloy):
        H = ternary_alloy.enthalpy_of_mixing(np.array([0.2, 0.3, 0.5]))
        assert np.ndim(H) == 0

    def test_enthalpy_matches_pairwise_loop(self, ternary_alloy):
        # Vectorized result must equal the explicit pairwise sum
        grid = ternary_alloy.simplex_grid(resolution=4)
        H = ternary_alloy.enthalpy_of_mixing(grid, unit="eV/atom")
        M = ternary_alloy.Mij()
        for row, h in zip(grid, H):
            expected = 0.0
            for i in range(3):
                for j in range(i + 1, 3):
                    if row[i] + row[j] == 0:
                        continue
                    expected += (
                        (M[i, j] * row[j] + M[j, i] * row[i]) * row[i] * row[j] / (row[i] + row[j])
                    )
            assert np.isclose(h, expected)

    def test_enthalpy_unit_conversion_kJmol_vs_eVatom(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid(resolution=5)
        H_kJ = ternary_alloy.enthalpy_of_mixing(grid, unit="kJ/mol")
        H_eV = ternary_alloy.enthalpy_of_mixing(grid, unit="eV/atom")
        assert np.allclose(H_kJ, H_eV * kJmol)

    def test_enthalpy_wrong_number_of_columns_raises(self, ternary_alloy):
        x = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError):
            ternary_alloy.enthalpy_of_mixing(x)

    def test_enthalpy_rows_not_sum_to_one_raises(self, ternary_alloy):
        x = np.array([[0.5, 0.3, 0.1]])  # sums to 0.9
        with pytest.raises(ValueError, match="sum to 1"):
            ternary_alloy.enthalpy_of_mixing(x)

    def test_enthalpy_negative_composition_raises(self, ternary_alloy):
        x = np.array([[1.1, -0.1, 0.0]])
        with pytest.raises(ValueError, match="non-negative"):
            ternary_alloy.enthalpy_of_mixing(x)

    def test_enthalpy_invalid_unit_raises(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid(resolution=3)
        with pytest.raises(ValueError, match="Invalid unit"):
            ternary_alloy.enthalpy_of_mixing(grid, unit="J/mol")


# ─────────────────── MultiComponentAlloy – Gibbs Free Energy ─────────────────


class TestMultiComponentAlloyGibbs:
    def test_gibbs_shape(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid(resolution=4)
        t = np.array([300.0, 600.0])
        G = ternary_alloy.gibbs_free_energy_of_mixing(grid, t)
        assert G.shape == (len(grid), len(t))

    def test_gibbs_matches_formula(self, ternary_alloy):
        grid = ternary_alloy.simplex_grid(resolution=4)
        t = np.array([300.0, 1000.0])
        G = ternary_alloy.gibbs_free_energy_of_mixing(grid, t)
        H = ternary_alloy.enthalpy_of_mixing(grid, unit="kJ/mol")
        S = ternary_alloy.configurational_entropy(grid, unit="kJ/(mol*K)")
        for j, temp in enumerate(t):
            assert np.allclose(G[:, j], H - temp * S)
