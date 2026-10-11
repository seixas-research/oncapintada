# -*- coding: utf-8 -*-
# file: monte_carlo.py

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


"""ASE-backed Metropolis chains, simulated annealing, and basin hopping."""

from dataclasses import asdict, dataclass
from math import exp
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.optimize import BFGS, FIRE, LBFGS

from ._validation import positive_integer, site_indices
from .constants import kB


def _temperature(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("temperature must be a real scalar in kelvin.")
    if not np.isfinite(value) or value < 0:
        raise ValueError("temperature must be finite and non-negative.")
    return float(value)


@dataclass(frozen=True)
class TemperatureSchedule:
    """Cooling temperatures in kelvin, indexed by zero-based proposal attempt.

    ``scheme`` is constant, linear, exponential, or logarithmic. Cooling starts
    at ``initial_temperature`` on attempt 0 and reaches ``final_temperature``
    on attempt ``n_steps - 1``; subsequent attempts stay at the final value.
    Exponential and logarithmic endpoints must be positive. For logarithmic
    cooling, T(n) = Ti / [1 + (Ti/Tf - 1) * ln(n+1)/ln(n_steps)], a
    regularized inverse-log law. Rejected proposals also advance the schedule.
    """

    initial_temperature: float
    final_temperature: float | None = None
    n_steps: int = 1
    scheme: str = "constant"

    def __post_init__(self):
        initial = _temperature(self.initial_temperature)
        final = initial if self.final_temperature is None else _temperature(self.final_temperature)
        steps = positive_integer(self.n_steps, "n_steps")
        if self.scheme not in ("constant", "linear", "exponential", "logarithmic"):
            raise ValueError("scheme must be constant, linear, exponential, or logarithmic.")
        if self.scheme == "constant":
            if final != initial:
                raise ValueError("A constant schedule requires equal endpoint temperatures.")
        else:
            if self.final_temperature is None:
                raise ValueError("A cooling schedule requires final_temperature.")
            if steps < 2:
                raise ValueError("A cooling schedule requires at least two steps.")
            if final > initial:
                raise ValueError("final_temperature must not exceed initial_temperature.")
            if self.scheme != "linear" and final <= 0:
                raise ValueError("Exponential and logarithmic temperatures must be positive.")
        object.__setattr__(self, "initial_temperature", initial)
        object.__setattr__(self, "final_temperature", final)
        object.__setattr__(self, "n_steps", steps)

    def __call__(self, step):
        """Temperature for a non-negative integer proposal index."""
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral):
            raise TypeError("step must be an integer.")
        if step < 0:
            raise ValueError("step must be non-negative.")
        initial, final = self.initial_temperature, self.final_temperature
        if self.scheme == "constant" or step == 0:
            return initial
        if step >= self.n_steps - 1:
            return final
        fraction = step / (self.n_steps - 1)
        if self.scheme == "linear":
            return (1 - fraction) * initial + fraction * final
        if self.scheme == "exponential":
            return float(np.exp((1 - fraction) * np.log(initial) + fraction * np.log(final)))
        fraction = np.log1p(step) / np.log(self.n_steps)
        # Algebraically equivalent to the documented inverse-log formula,
        # avoiding an overflowing initial/final temperature ratio.
        return final / ((1 - fraction) * (final / initial) + fraction)


def metropolis_probability(delta_energy, temperature):
    """min(1, exp(-ΔE/kBT)), with total ΔE in eV and temperature in kelvin.

    At T=0, accept downhill and equal-energy moves and reject uphill moves.
    """
    if not np.isfinite(delta_energy):
        raise ValueError("Energy difference must be finite.")
    temperature = _temperature(temperature)
    if delta_energy <= 0:
        return 1.0
    thermal_energy = kB * temperature
    return exp(-delta_energy / thermal_energy) if thermal_energy > 0 else 0.0


class RelaxationError(RuntimeError):
    """An ASE optimizer exhausted the geometry-relaxation budget."""


@dataclass(frozen=True)
class MCStep:
    """One proposal and its acceptance outcome; energies are total eV."""

    step: int
    move: str
    accepted: bool
    energy: float
    proposal_energy: float | None
    delta_energy: float | None
    acceptance_probability: float
    swapped_indices: tuple[int, int] | None
    relaxation_converged: bool | None
    temperature: float
    error: str | None = None


@dataclass
class MonteCarloResult:
    """Final state, lowest accepted state, sampled chain, and complete step log.

    Trajectory frames include rejected-state repetitions at the requested
    sampling interval. They carry independent single-point energy snapshots.
    The best state is tracked across all runs of the same sampler.
    ``temperature`` is the last attempted temperature, in kelvin;
    ``temperatures`` contains one value per recorded proposal.
    """

    atoms: Atoms
    best_atoms: Atoms
    initial_energy: float
    energy: float
    best_energy: float
    steps: tuple[MCStep, ...]
    trajectory: tuple[Atoms, ...]
    temperature: float
    relaxed_proposals: bool
    initial_temperature: float

    @property
    def accepted_moves(self):
        return sum(step.accepted for step in self.steps)

    @property
    def acceptance_rate(self):
        return self.accepted_moves / len(self.steps) if self.steps else 0.0

    @property
    def energies(self):
        """Full chain energy trace, including initial and repeated states."""
        return np.asarray([self.initial_energy] + [step.energy for step in self.steps])

    @property
    def temperatures(self):
        """Kelvin used for each proposal, including rejected or failed attempts."""
        return np.asarray([step.temperature for step in self.steps])

    def to_dataframe(self):
        """One diagnostic row for every proposal, including failed proposals."""
        return pd.DataFrame([asdict(step) for step in self.steps])

    def write(self, filename: str | Path, format: str = "extxyz", *, trajectory=True):
        """Export sampled states, or just the final state with trajectory=False."""
        if trajectory and not self.trajectory:
            raise ValueError(
                "No trajectory was stored; use trajectory=False to export the final state."
            )
        write(filename, list(self.trajectory) if trajectory else self.atoms, format=format)


class MetropolisMonteCarlo:
    """Markov chain over alloy occupations and/or Cartesian coordinates.

    ``calculator`` is an ASE calculator, a zero-argument calculator factory,
    or None to use the input atoms' calculator. ``move`` is 'swap',
    'displacement', or 'mixed'; mixed chooses either rule with probability 1/2.
    Swaps exchange chemical species on two distinct unlike-species sites.
    Displacements perturb every atom independently. Uniform noise uses
    [-displacement, displacement] per coordinate; Gaussian noise uses this
    amplitude as its standard deviation, in Å. ASE constraints are respected.

    ``temperature`` is a constant in kelvin, a TemperatureSchedule, or a
    callable mapping the zero-based global proposal index to kelvin. Cooling
    continues across run() calls and advances on all recorded attempts.

    ``relax=True`` optimizes the initial state and every candidate before
    comparing energies. This is a basin-hopping search on minimized energies,
    rather than canonical sampling of the original unrelaxed energy surface.
    The cell stays fixed. Optimizers may be BFGS, FIRE, LBFGS, or an ASE-compatible
    factory. Failed calculations raise by default; on_error='reject' records
    failed proposals as rejected, preserving the current state.
    """

    def __init__(
        self,
        atoms: Atoms,
        calculator=None,
        *,
        temperature=300.0,
        move="swap",
        displacement=0.05,
        distribution="uniform",
        swap_indices=None,
        relax=False,
        optimizer="BFGS",
        fmax=0.05,
        relax_steps=200,
        on_error="raise",
        seed=42,
    ):
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise ValueError("atoms must be a non-empty ASE Atoms object.")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError("Atom positions and cell must be finite.")
        periodic = atoms.cell.array[atoms.pbc]
        if len(periodic) and np.linalg.matrix_rank(periodic) != len(periodic):
            raise ValueError("Periodic directions require independent, nonzero cell vectors.")
        schedule = temperature if callable(temperature) else None
        initial_temperature = _temperature(schedule(0) if schedule is not None else temperature)
        if move not in ("swap", "displacement", "mixed"):
            raise ValueError("move must be 'swap', 'displacement', or 'mixed'.")
        if distribution not in ("uniform", "gaussian"):
            raise ValueError("distribution must be 'uniform' or 'gaussian'.")
        if not np.isfinite(displacement) or displacement <= 0:
            raise ValueError("displacement must be finite and positive.")
        if not np.isfinite(fmax) or fmax <= 0:
            raise ValueError("fmax must be finite and positive.")
        positive_integer(relax_steps, "relax_steps")
        if not isinstance(relax, (bool, np.bool_)):
            raise TypeError("relax must be a boolean.")
        if on_error not in ("raise", "reject"):
            raise ValueError("on_error must be 'raise' or 'reject'.")
        optimizers = {"BFGS": BFGS, "FIRE": FIRE, "LBFGS": LBFGS}
        if isinstance(optimizer, str):
            if optimizer not in optimizers:
                raise ValueError("optimizer must be BFGS, FIRE, LBFGS, or an optimizer factory.")
            optimizer = optimizers[optimizer]
        if not callable(optimizer):
            raise TypeError("optimizer must be a callable ASE-compatible optimizer factory.")
        source = calculator if calculator is not None else atoms.calc
        if not isinstance(source, BaseCalculator) and not callable(source):
            raise TypeError(
                "Provide an ASE calculator, a calculator factory, or atoms with a calculator."
            )
        indices = (
            np.arange(len(atoms)) if swap_indices is None else site_indices(swap_indices, atoms)
        )
        if move != "displacement" and len(np.unique(atoms.numbers[indices])) < 2:
            raise ValueError("Swap moves require at least two chemical species in swap_indices.")
        self.temperature = initial_temperature
        self._temperature_schedule = schedule
        self._schedule_name = (
            schedule.scheme
            if isinstance(schedule, TemperatureSchedule)
            else "custom"
            if schedule is not None
            else "constant"
        )
        self.move, self.displacement, self.distribution = move, float(displacement), distribution
        self.swap_indices = np.asarray(indices, dtype=int)
        self.relax, self.fmax, self.relax_steps = bool(relax), float(fmax), int(relax_steps)
        self.on_error = on_error
        self._optimizer = optimizer
        self._calculator = source
        self._rng = np.random.default_rng(seed)
        self._records = []
        self._atoms = atoms.copy()
        # Input correlation/chain diagnostics describe the source structure;
        # retain them as provenance rather than stale output-state properties.
        self._atoms.info = {
            (f"source_{key}" if key.startswith(("ce_", "sqs_", "mc_")) else key): value
            for key, value in self._atoms.info.items()
        }
        # Initialization always raises on failure: a chain needs a valid state.
        self._energy, _ = self._evaluate(self._atoms)
        self._best = self._atoms.copy()
        self._best_energy = self._energy
        self._best_step = 0
        self._best_temperature = initial_temperature

    def _evaluate(self, atoms):
        calculator = (
            self._calculator()
            if not isinstance(self._calculator, BaseCalculator)
            else self._calculator
        )
        if not isinstance(calculator, BaseCalculator):
            raise TypeError("Calculator factory must return an ASE calculator.")
        atoms.calc = calculator
        converged = None
        if self.relax:
            optimization = self._optimizer(atoms, logfile=None)
            converged = bool(optimization.run(fmax=self.fmax, steps=self.relax_steps))
            if not converged:
                raise RelaxationError(
                    f"Geometry did not converge to fmax={self.fmax} within {self.relax_steps} steps."
                )
        energy = float(atoms.get_potential_energy())
        if not np.isfinite(energy) or not np.isfinite(atoms.positions).all():
            raise ValueError("Calculator returned a non-finite energy or geometry.")
        return energy, converged

    def _snapshot(self, atoms, energy, record=None):
        snapshot = atoms.copy()
        # Replace stale source energy metadata with the evaluated chain energy.
        snapshot.info.pop("energy", None)
        snapshot.calc = SinglePointCalculator(snapshot, energy=energy)
        snapshot.info.update(
            mc_energy=energy,
            mc_temperature=self.temperature if record is None else record.temperature,
            mc_temperature_schedule=self._schedule_name,
            mc_relaxed_proposals=self.relax,
            mc_step=len(self._records),
        )
        if record is not None:
            snapshot.info.update(
                mc_move=record.move,
                mc_accepted=record.accepted,
                mc_acceptance_probability=record.acceptance_probability,
            )
            if record.error:
                snapshot.info["mc_error"] = record.error
        return snapshot

    @property
    def atoms(self):
        """Copy of the current state with its evaluated single-point energy."""
        return self._snapshot(self._atoms, self._energy)

    @property
    def energy(self):
        return self._energy

    @property
    def best_atoms(self):
        snapshot = self._snapshot(self._best, self._best_energy)
        snapshot.info["mc_step"] = self._best_step
        snapshot.info["mc_temperature"] = self._best_temperature
        return snapshot

    def _proposal(self):
        atoms = self._atoms.copy()
        move = self.move
        if move == "mixed":
            move = "swap" if self._rng.random() < 0.5 else "displacement"
        swapped = None
        if move == "swap":
            a = int(self._rng.choice(self.swap_indices))
            unlike = self.swap_indices[atoms.numbers[self.swap_indices] != atoms.numbers[a]]
            b = int(self._rng.choice(unlike))
            atoms.numbers[a], atoms.numbers[b] = atoms.numbers[b], atoms.numbers[a]
            swapped = a, b
        else:
            if self.distribution == "uniform":
                noise = self._rng.uniform(
                    -self.displacement, self.displacement, atoms.positions.shape
                )
            else:
                noise = self._rng.normal(0, self.displacement, atoms.positions.shape)
            atoms.set_positions(atoms.positions + noise)
        return atoms, move, swapped

    def propose(self):
        """Generate an unevaluated candidate without changing the current state.

        Consumes the sampler's RNG stream; use a separate seeded sampler when
        inspecting proposals without changing the subsequent chain sequence.
        """
        atoms, move, swapped = self._proposal()
        atoms.info["mc_move"] = move
        if swapped is not None:
            atoms.info["mc_swapped_indices"] = list(swapped)
        return atoms

    def step(self):
        """Propose, optionally relax, evaluate, and accept/reject one move."""
        # Validate scheduling errors before consuming RNG or evaluating a
        # candidate. on_error='reject' applies only to candidate calculations.
        temperature = _temperature(
            self._temperature_schedule(len(self._records))
            if self._temperature_schedule is not None
            else self.temperature
        )
        proposal, move, swapped = self._proposal()
        proposal_energy = delta = None
        probability, accepted, converged, error = 0.0, False, None, None
        try:
            proposal_energy, converged = self._evaluate(proposal)
            delta = proposal_energy - self._energy
            probability = metropolis_probability(delta, temperature)
            accepted = probability >= 1 or self._rng.random() < probability
        except Exception as exception:
            if self.on_error == "raise":
                raise
            error = f"{type(exception).__name__}: {exception}"
            converged = False if isinstance(exception, RelaxationError) else None
        if accepted:
            self._atoms, self._energy = proposal, proposal_energy
            if self._energy < self._best_energy:
                self._best, self._best_energy = proposal.copy(), self._energy
                self._best_step = len(self._records) + 1
                self._best_temperature = temperature
        self.temperature = temperature
        record = MCStep(
            len(self._records) + 1,
            move,
            accepted,
            self._energy,
            proposal_energy,
            delta,
            probability,
            swapped,
            converged,
            temperature,
            error,
        )
        self._records.append(record)
        return record

    def run(self, n_steps, *, sample_interval=1, store_trajectory=True):
        """Continue the chain, retaining rejected states and complete diagnostics.

        Sampling starts with the current state and then every sample_interval
        proposals. The final state is always accessible in result.atoms.
        """
        n_steps = positive_integer(n_steps, "n_steps")
        sample_interval = positive_integer(sample_interval, "sample_interval")
        initial_energy = self._energy
        initial_temperature = self.temperature
        frames = [self.atoms] if store_trajectory else []
        records = []
        for index in range(1, n_steps + 1):
            record = self.step()
            records.append(record)
            if store_trajectory and index % sample_interval == 0:
                frames.append(self._snapshot(self._atoms, self._energy, record))
        return MonteCarloResult(
            self.atoms,
            self.best_atoms,
            initial_energy,
            self._energy,
            self._best_energy,
            tuple(records),
            tuple(frames),
            self.temperature,
            self.relax,
            initial_temperature,
        )


class ChemicalSwapMC(MetropolisMonteCarlo):
    """Convenience sampler restricted to unlike-species swap moves."""

    def __init__(self, atoms, calculator=None, **kwargs):
        if kwargs.pop("move", "swap") != "swap":
            raise ValueError("ChemicalSwapMC supports swap moves only.")
        super().__init__(atoms, calculator, move="swap", **kwargs)


class BasinHopping(MetropolisMonteCarlo):
    """Convenience search with relaxed proposals and mixed moves by default."""

    def __init__(self, atoms, calculator=None, **kwargs):
        if not kwargs.pop("relax", True):
            raise ValueError("BasinHopping requires relaxed proposals.")
        kwargs.setdefault("move", "mixed")
        super().__init__(atoms, calculator, relax=True, **kwargs)


class MDMC:
    """Reserved interface for molecular-dynamics-assisted Monte Carlo."""

    def __init__(self):
        raise NotImplementedError("MDMC is not implemented yet.")
