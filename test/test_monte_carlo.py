"""Physical Metropolis limits, chain bookkeeping, and ASE calculator integration."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read

from oncapintada import (
    BasinHopping,
    ChemicalSwapMC,
    MetropolisMonteCarlo,
    RelaxationError,
    TemperatureSchedule,
    metropolis_probability,
)
from oncapintada.constants import kB


class Harmonic(Calculator):
    """Analytic energy and forces permit a genuine geometry-optimizer test."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.5 * np.sum(self.atoms.positions**2),
            "forces": -self.atoms.positions.copy(),
        }


class SiteEnergy(Calculator):
    """Two occupations have energies 0 and 0.1 eV on a fixed two-site lattice."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = 0.1 if self.atoms[0].symbol == "Au" else 0.0
        self.results = {"energy": energy, "forces": np.zeros_like(self.atoms.positions)}


class FailsOnAuFirst(SiteEnergy):
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if atoms[0].symbol == "Au":
            raise RuntimeError("intentional failed proposal")
        super().calculate(atoms, properties, system_changes)


class NonFinite(SiteEnergy):
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms[0].symbol == "Au":
            self.results["energy"] = np.nan


def alloy():
    return Atoms("CuAu", positions=[[0, 0, 0], [1, 0, 0]])


def test_metropolis_physical_limits_and_total_energy_units():
    assert metropolis_probability(-1, 300) == 1
    assert metropolis_probability(0, 0) == 1
    assert metropolis_probability(1, 0) == 0
    assert metropolis_probability(kB * 300, 300) == pytest.approx(np.exp(-1))
    assert metropolis_probability(10000, 1) == 0
    for delta, temperature in ((np.inf, 300), (1, -1), (1, np.nan)):
        with pytest.raises(ValueError):
            metropolis_probability(delta, temperature)


def test_zero_temperature_rejection_preserves_chain_and_input():
    original = alloy()
    sampler = MetropolisMonteCarlo(original, SiteEnergy, temperature=0)
    result = sampler.run(5)
    assert result.acceptance_rate == 0
    np.testing.assert_array_equal(result.energies, np.zeros(6))
    assert len(result.trajectory) == 6
    for frame in result.trajectory:
        assert frame.get_chemical_symbols() == ["Cu", "Au"]
        assert frame.get_potential_energy() == 0
    assert all(step.proposal_energy == 0.1 and not step.accepted for step in result.steps)
    assert original.calc is None
    assert original.get_chemical_symbols() == ["Cu", "Au"]


def test_downhill_acceptance_and_best_state():
    original = alloy()
    original.set_chemical_symbols(["Au", "Cu"])
    sampler = ChemicalSwapMC(original, SiteEnergy, temperature=0)
    result = sampler.run(2)
    assert result.steps[0].accepted and not result.steps[1].accepted
    assert result.steps[0].delta_energy == -0.1
    assert result.energy == result.best_energy == 0
    assert result.best_atoms.info["mc_step"] == 1
    np.testing.assert_allclose(result.energies, [0.1, 0, 0])


def test_two_state_chain_matches_boltzmann_population():
    sampler = MetropolisMonteCarlo(alloy(), SiteEnergy, temperature=600, seed=18)
    # Every step proposes the other occupation. Keeping rejected states is
    # essential: storing only accepted states would alternate at 50/50.
    result = sampler.run(6000, store_trajectory=False)
    observed = np.mean(result.energies[501:] > 0)
    expected = 1 / (1 + np.exp(0.1 / (kB * 600)))
    assert observed == pytest.approx(expected, abs=0.025)
    assert 0.1 < result.acceptance_rate < 0.4
    assert result.trajectory == ()


def test_seeded_chain_continuation_matches_single_run():
    first = MetropolisMonteCarlo(alloy(), SiteEnergy(), temperature=900, seed=37)
    second = MetropolisMonteCarlo(alloy(), SiteEnergy(), temperature=900, seed=37)
    complete = first.run(40)
    a, b = second.run(15), second.run(25)
    np.testing.assert_array_equal(complete.energies, np.r_[a.energies, b.energies[1:]])
    assert b.steps[0].step == 16
    assert b.steps[-1].step == 40


def test_swap_rule_exchanges_exactly_two_unlike_species_and_preserves_spectators():
    atoms = Atoms("CuAuAgO", positions=np.arange(12).reshape(4, 3))
    sampler = MetropolisMonteCarlo(atoms, Harmonic, swap_indices=[0, 1, 2], seed=42)
    proposal = sampler.propose()
    changed = np.flatnonzero(proposal.numbers != atoms.numbers)
    assert len(changed) == 2
    assert atoms.numbers[changed[0]] != atoms.numbers[changed[1]]
    assert sorted(proposal.get_chemical_symbols()) == sorted(atoms.get_chemical_symbols())
    assert proposal[-1].symbol == "O"
    np.testing.assert_array_equal(proposal.positions, atoms.positions)
    np.testing.assert_array_equal(sampler.atoms.numbers, atoms.numbers)


@pytest.mark.parametrize("distribution", ["uniform", "gaussian"])
def test_displacement_rule_perturbs_all_atoms_with_expected_noise(distribution):
    original = Atoms("CuAuAg", positions=np.zeros((3, 3)))
    sampler = MetropolisMonteCarlo(
        original,
        Harmonic,
        move="displacement",
        displacement=0.2,
        distribution=distribution,
        seed=42,
    )
    proposal = sampler.propose()
    rng = np.random.default_rng(42)
    expected = (
        rng.uniform(-0.2, 0.2, (3, 3)) if distribution == "uniform" else rng.normal(0, 0.2, (3, 3))
    )
    np.testing.assert_allclose(proposal.positions, expected)
    assert np.all(np.linalg.norm(proposal.positions - original.positions, axis=1) > 0)
    np.testing.assert_array_equal(proposal.numbers, original.numbers)
    np.testing.assert_array_equal(sampler.atoms.positions, original.positions)


def test_displacements_respect_ase_constraints():
    original = alloy()
    original.set_constraint(FixAtoms(indices=[0]))
    sampler = MetropolisMonteCarlo(original, Harmonic, move="displacement", seed=42)
    proposal = sampler.propose()
    np.testing.assert_array_equal(proposal.positions[0], original.positions[0])
    assert not np.array_equal(proposal.positions[1], original.positions[1])


def test_mixed_moves_visit_both_rules_and_preserve_atom_counts():
    sampler = MetropolisMonteCarlo(alloy(), Harmonic, move="mixed", seed=11, temperature=1000)
    result = sampler.run(30)
    assert {step.move for step in result.steps} == {"swap", "displacement"}
    for frame in result.trajectory:
        assert sorted(frame.get_chemical_symbols()) == ["Au", "Cu"]


def test_real_ase_geometry_relaxation_initial_and_proposed_states():
    original = Atoms("CuAu", positions=[[1, 0, 0], [0, 1, 0]])
    sampler = BasinHopping(
        original,
        Harmonic,
        move="displacement",
        fmax=1e-6,
        relax_steps=100,
        displacement=0.2,
        seed=42,
    )
    assert sampler.energy < 1e-12
    result = sampler.run(3)
    assert result.relaxed_proposals
    assert all(step.relaxation_converged for step in result.steps)
    assert all(step.proposal_energy < 1e-12 for step in result.steps)
    np.testing.assert_array_equal(original.positions, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(result.atoms.cell.array, original.cell.array)


def test_calculator_attached_to_input_is_supported():
    atoms = alloy()
    atoms.calc = EMT()
    sampler = MetropolisMonteCarlo(atoms, temperature=0, seed=2)
    assert np.isfinite(sampler.energy)
    assert sampler.atoms.get_potential_energy() == pytest.approx(atoms.get_potential_energy())


def test_failed_proposal_can_be_rejected_and_logged():
    sampler = MetropolisMonteCarlo(alloy(), FailsOnAuFirst, on_error="reject")
    record = sampler.step()
    assert not record.accepted and record.proposal_energy is None
    assert "intentional failed proposal" in record.error
    assert sampler.energy == 0
    assert sampler.atoms.get_chemical_symbols() == ["Cu", "Au"]
    nonfinite = MetropolisMonteCarlo(alloy(), NonFinite, on_error="reject")
    assert "non-finite energy" in nonfinite.step().error


def test_failed_proposal_raises_by_default_and_keeps_state():
    sampler = MetropolisMonteCarlo(alloy(), FailsOnAuFirst)
    with pytest.raises(RuntimeError, match="intentional"):
        sampler.step()
    assert sampler.energy == 0
    assert sampler.atoms.get_chemical_symbols() == ["Cu", "Au"]


def test_unconverged_relaxation_is_recorded_as_rejection():
    sampler = MetropolisMonteCarlo(
        Atoms("Cu", positions=[[0, 0, 0]]),
        Harmonic,
        move="displacement",
        relax=True,
        fmax=1e-12,
        relax_steps=1,
        displacement=1,
        on_error="reject",
    )
    record = sampler.step()
    assert not record.accepted and record.relaxation_converged is False
    assert "Geometry did not converge" in record.error
    assert sampler.energy == 0
    with pytest.raises(RelaxationError, match="did not converge"):
        MetropolisMonteCarlo(
            Atoms("Cu", positions=[[2, 0, 0]]),
            Harmonic,
            move="displacement",
            relax=True,
            fmax=1e-12,
            relax_steps=1,
        )


def test_result_export_keeps_energy_snapshots_and_rejected_states(tmp_path):
    sampler = MetropolisMonteCarlo(alloy(), SiteEnergy, temperature=0)
    result = sampler.run(4, sample_interval=2)
    assert len(result.trajectory) == 3
    assert result.to_dataframe()["accepted"].tolist() == [False] * 4
    path = tmp_path / "chain.extxyz"
    result.write(path)
    restored = read(path, index=":")
    assert [frame.info["mc_step"] for frame in restored] == [0, 2, 4]
    assert all(frame.get_potential_energy() == 0 for frame in restored)
    result.write(tmp_path / "final.extxyz", trajectory=False)
    assert read(tmp_path / "final.extxyz").get_potential_energy() == result.energy
    no_frames = sampler.run(1, store_trajectory=False)
    with pytest.raises(ValueError, match="No trajectory"):
        no_frames.write(tmp_path / "missing.extxyz")


def test_current_state_snapshots_cannot_mutate_the_chain():
    sampler = MetropolisMonteCarlo(alloy(), SiteEnergy, temperature=0)
    exposed = sampler.atoms
    exposed.positions[:] = 999
    exposed.set_chemical_symbols(["Au", "Au"])
    np.testing.assert_array_equal(sampler.atoms.positions, alloy().positions)
    assert sampler.energy == 0


def test_source_correlation_metadata_is_preserved_as_provenance():
    original = alloy()
    original.info.update(sqs_objective=0.0, ce_correlation=[1, 2], mc_step=9)
    result = MetropolisMonteCarlo(original, SiteEnergy, temperature=0).run(1)
    assert "sqs_objective" not in result.atoms.info
    assert result.atoms.info["source_sqs_objective"] == 0
    assert result.atoms.info["source_ce_correlation"] == [1, 2]
    assert result.atoms.info["source_mc_step"] == 9
    assert result.atoms.info["mc_step"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -1},
        {"temperature": np.nan},
        {"move": "unknown"},
        {"distribution": "unknown"},
        {"displacement": 0},
        {"fmax": 0},
        {"relax_steps": 0},
        {"relax": "yes"},
        {"optimizer": "unknown"},
        {"optimizer": 7},
        {"on_error": "ignore"},
        {"swap_indices": []},
        {"swap_indices": [0, 0]},
        {"swap_indices": [2]},
    ],
)
def test_invalid_options_are_refused(kwargs):
    with pytest.raises((TypeError, ValueError)):
        MetropolisMonteCarlo(alloy(), SiteEnergy, **kwargs)


def test_missing_or_invalid_calculators_and_single_species_swaps():
    with pytest.raises(TypeError, match="ASE calculator"):
        MetropolisMonteCarlo(alloy())
    with pytest.raises(TypeError, match="factory must return"):
        MetropolisMonteCarlo(alloy(), lambda: None)
    with pytest.raises(ValueError, match="two chemical species"):
        MetropolisMonteCarlo(Atoms("Cu"), Harmonic)
    with pytest.raises(ValueError, match="swap moves only"):
        ChemicalSwapMC(alloy(), SiteEnergy, move="mixed")
    with pytest.raises(ValueError, match="requires relaxed"):
        BasinHopping(alloy(), Harmonic, relax=False)


@pytest.mark.parametrize(
    "scheme,expected",
    [
        ("linear", [800, 550, 300, 50]),
        ("exponential", [800, 800 * (1 / 16) ** (1 / 3), 800 * (1 / 16) ** (2 / 3), 50]),
        ("logarithmic", [800, 800 / 8.5, 800 / (1 + 15 * np.log(3) / np.log(4)), 50]),
    ],
)
def test_cooling_laws_known_temperatures_and_endpoint_hold(scheme, expected):
    schedule = TemperatureSchedule(800, 50, n_steps=4, scheme=scheme)
    np.testing.assert_allclose([schedule(i) for i in range(4)], expected)
    assert schedule(4) == schedule(1000) == 50


def test_constant_and_zero_ending_linear_schedules():
    constant = TemperatureSchedule(0)
    assert constant(0) == constant(100) == 0
    linear = TemperatureSchedule(100, 0, n_steps=3, scheme="linear")
    assert [linear(i) for i in range(4)] == [100, 50, 0, 0]


@pytest.mark.parametrize("scheme", ["linear", "exponential", "logarithmic"])
def test_annealing_continuation_and_stepwise_metropolis_acceptance(scheme, tmp_path):
    schedule = TemperatureSchedule(1200, 100, n_steps=8, scheme=scheme)
    single = MetropolisMonteCarlo(alloy(), SiteEnergy, temperature=schedule, seed=17)
    split = MetropolisMonteCarlo(alloy(), SiteEnergy, temperature=schedule, seed=17)
    complete = single.run(10)
    first, second = split.run(3), split.run(7)
    np.testing.assert_array_equal(complete.energies, np.r_[first.energies, second.energies[1:]])
    np.testing.assert_array_equal(
        complete.temperatures, np.r_[first.temperatures, second.temperatures]
    )
    assert complete.initial_temperature == 1200
    assert complete.temperature == 100
    assert complete.temperatures[0] == 1200
    assert complete.temperatures[7] == 100
    assert second.initial_temperature == first.temperature
    for step, frame in zip(complete.steps, complete.trajectory[1:]):
        expected = 1 if step.delta_energy <= 0 else np.exp(-0.1 / (kB * step.temperature))
        assert step.acceptance_probability == pytest.approx(expected)
        assert frame.info["mc_temperature"] == step.temperature
        assert frame.info["mc_temperature_schedule"] == scheme
    assert complete.to_dataframe()["temperature"].tolist() == complete.temperatures.tolist()
    path = tmp_path / "annealing.extxyz"
    complete.write(path)
    frames = read(path, ":")
    np.testing.assert_allclose(
        [f.info["mc_temperature"] for f in frames[1:]], complete.temperatures
    )


def test_rejected_and_failed_proposals_advance_cooling_and_preserve_best_temperature():
    schedule = TemperatureSchedule(1, 0, n_steps=3, scheme="linear")
    rejected = MetropolisMonteCarlo(alloy(), SiteEnergy, temperature=schedule).run(4)
    assert rejected.accepted_moves == 0
    np.testing.assert_array_equal(rejected.temperatures, [1, 0.5, 0, 0])
    assert rejected.best_atoms.info["mc_temperature"] == 1
    failed = MetropolisMonteCarlo(
        alloy(), FailsOnAuFirst, temperature=schedule, on_error="reject"
    ).run(4)
    assert all(step.error for step in failed.steps)
    np.testing.assert_array_equal(failed.temperatures, [1, 0.5, 0, 0])
    assert failed.energy == 0
    downhill = alloy()
    downhill.set_chemical_symbols(["Au", "Cu"])
    result = ChemicalSwapMC(downhill, SiteEnergy, temperature=schedule).run(4)
    assert result.best_atoms.info["mc_step"] == 1
    assert result.best_atoms.info["mc_temperature"] == 1
    assert result.atoms.info["mc_temperature"] == 0


@pytest.mark.parametrize("move", ["swap", "displacement", "mixed"])
def test_annealing_supports_relaxed_and_unrelaxed_proposal_rules(move):
    for relax in (False, True):
        sampler = MetropolisMonteCarlo(
            alloy(),
            Harmonic,
            move=move,
            distribution="gaussian",
            relax=relax,
            temperature=TemperatureSchedule(900, 0, n_steps=4, scheme="linear"),
            fmax=1e-6,
            relax_steps=100,
        )
        result = sampler.run(4)
        np.testing.assert_allclose(result.temperatures, [900, 600, 300, 0])
        assert sorted(result.atoms.get_chemical_symbols()) == ["Au", "Cu"]
        if relax:
            assert all(step.relaxation_converged for step in result.steps)
        else:
            assert all(step.relaxation_converged is None for step in result.steps)


def test_custom_temperature_schedule_and_invalid_temperatures_do_not_advance_chain():
    values = [600, -1, 0]
    sampler = MetropolisMonteCarlo(
        alloy(), SiteEnergy, temperature=lambda step: values[step], on_error="reject"
    )
    first = sampler.step()
    with pytest.raises(ValueError, match="non-negative"):
        sampler.step()
    assert sampler.temperature == first.temperature
    values[1] = 300
    remaining = sampler.run(2)
    assert [step.step for step in remaining.steps] == [2, 3]
    np.testing.assert_array_equal(remaining.temperatures, [300, 0])
    assert remaining.atoms.info["mc_temperature_schedule"] == "custom"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_temperature": -1},
        {"initial_temperature": np.inf},
        {"initial_temperature": True},
        {"initial_temperature": [300]},
        {"scheme": "unknown"},
        {"n_steps": 0},
        {"n_steps": 2.5},
        {"final_temperature": 200},
        {"scheme": "linear"},
        {"scheme": "linear", "final_temperature": 10},
        {"scheme": "linear", "final_temperature": 400, "n_steps": 2},
        {"scheme": "exponential", "final_temperature": 0, "n_steps": 2},
        {"scheme": "logarithmic", "final_temperature": 0, "n_steps": 2},
    ],
)
def test_invalid_temperature_schedules_are_refused(kwargs):
    options = {"initial_temperature": 300, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        TemperatureSchedule(**options)


def test_schedule_index_validation_and_extreme_temperature_ranges():
    constant = TemperatureSchedule(300)
    for step in (True, 1.5, -1):
        with pytest.raises((TypeError, ValueError)):
            constant(step)
    for scheme in ("linear", "exponential", "logarithmic"):
        schedule = TemperatureSchedule(1e300, 1e-300, n_steps=5, scheme=scheme)
        temperatures = [schedule(i) for i in range(5)]
        assert np.isfinite(temperatures).all()
        assert all(a >= b >= 0 for a, b in zip(temperatures, temperatures[1:]))
