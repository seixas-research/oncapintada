# Metropolis Monte Carlo, simulated annealing, and structure relaxation

`MetropolisMonteCarlo` builds a Markov chain of ASE structures. Each step proposes
a new configuration, calculates its energy with an ASE calculator, and accepts
or rejects it with the Metropolis criterion. Rejected configurations leave the
current state unchanged and repeat that state in the sampled trajectory.

It supports binary and multicomponent alloys, an existing structure, an SQS,
or a configuration produced by the Cluster Expansion builder. Particle count,
species counts, cell, and boundary conditions are preserved.
Temperature can stay constant or vary with a cooling schedule.

## Start with an SQS and an ASE calculator

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from oncapintada import generate_sqs, MetropolisMonteCarlo

parent = bulk("Cu", "fcc", a=3.9, cubic=True)
sqs = generate_sqs(
    parent, {"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
    supercell=(2, 2, 2), pair_cutoff=4.0, seed=42,
)

sampler = MetropolisMonteCarlo(
    sqs.atoms,
    calculator=EMT,              # factory: a fresh calculator per evaluation
    temperature=600,            # kelvin
    move="mixed",               # equal chance of swaps or displacements
    displacement=0.03,          # Å, per-coordinate Gaussian standard deviation
    distribution="gaussian",
    swap_indices=sqs.active_indices,
    seed=42,
)
result = sampler.run(100)
result.write("mc_trajectory.extxyz")
result.to_dataframe().to_csv("mc_steps.csv", index=False)
print(result.initial_energy, result.energy, result.best_energy)  # total eV
print(result.acceptance_rate)
```

EMT is an inexpensive illustrative calculator. For production calculations,
supply the ASE calculator appropriate to the alloy. A calculator instance is
also accepted: `calculator=EMT()`. If `calculator` is omitted, the input atoms
must already have a calculator attached. Energy-only calculators suffice for
unrelaxed sampling; relaxation requires forces.

File-based calculators can use a factory that assigns an evaluation directory
for each new instance. The sampler evaluates serially; configure calculators'
work directories and external execution settings as you would for an ASE job.

The input atoms are copied. `sampler.atoms`, `result.atoms`, `result.best_atoms`,
and trajectory frames are independent snapshots with single-point energies.
Attach a new calculator before further force evaluation or geometry changes;
the snapshot calculator records the evaluated energy rather than rerunning the
simulation backend.

## Structure-generation rules

| `move` | Proposal |
| --- | --- |
| `"swap"` | Choose a random site and a second random site of a different chemical species, then exchange their species at fixed positions |
| `"displacement"` | Independently displace every atom in Cartesian coordinates |
| `"mixed"` | Choose either proposal rule with probability 1/2 |

By default, swaps can use any atom. Set `swap_indices` to indices, a boolean mask,
or an element symbol to restrict swaps to a sublattice. At least two species
must be present in that selection. For an already decorated SQS, its
`active_indices` identifies the substitutable sites regardless of their current
chemical symbols. Swaps preserve exact composition for any species count.

For uniform displacement noise, each Cartesian component is sampled from
`[-displacement, +displacement]`. For Gaussian noise, each component has zero
mean and standard deviation `displacement`. The cell stays fixed and positions
are not automatically wrapped into it. ASE position constraints are honored;
fixed atoms remain fixed even when a displacement proposal is applied to all
atoms. Projection through nonlinear constraints can alter proposal symmetry;
canonical sampling with such constraints requires a separate treatment of the
proposal density and constrained measure.

`sampler.propose()` returns an unevaluated candidate without changing the chain
state. It consumes random numbers, so inspecting proposals changes the later
sequence of that sampler. `sampler.step()` performs one evaluated proposal and
returns an `MCStep` record. Repeated `run()` calls continue the same RNG stream
and current state.

## Metropolis acceptance

Let `delta_energy = E_proposal - E_current`, using **total potential energies in
eV**. At positive temperature, acceptance is `min(1, exp(-delta_energy / (kB*T)))`,
with `kB` in eV/K and `T` in kelvin. Downhill and equal-energy moves are accepted.
At zero temperature, uphill moves are rejected. The standalone
`metropolis_probability(delta_energy, temperature)` exposes this calculation.

For unrelaxed, unconstrained Cartesian displacements and the fixed-composition
swap rule, the proposal is symmetric and this is the usual configurational
Metropolis chain at fixed volume. SQS optimization supplies an initial state;
Monte Carlo acceptance is then governed by calculator energies, so subsequent
states need not retain the initial SQS correlations. Assess equilibration,
burn-in, correlation times, and proposal amplitude when extracting ensemble
averages.

## Constant temperature and simulated annealing

Passing a number, such as `temperature=600`, keeps the temperature constant.
Pass a `TemperatureSchedule` instead to cool during the same swap,
displacement, or mixed-move chain. Both binary and multicomponent alloys,
calculator factories, and geometry relaxation support these schedules.

```python
from oncapintada import TemperatureSchedule

schedule = TemperatureSchedule(
    initial_temperature=1200,
    final_temperature=100,
    n_steps=1000,
    scheme="exponential",       # also "linear" or "logarithmic"
)
annealer = MetropolisMonteCarlo(
    sqs.atoms, calculator=EMT, temperature=schedule,
    move="mixed", distribution="gaussian", displacement=0.03, seed=42,
)
annealed = annealer.run(1000)
annealed.write("annealing.extxyz")
annealed.to_dataframe().to_csv("annealing_steps.csv", index=False)
print(annealed.temperatures)     # one kelvin value per attempted proposal
print(annealed.best_energy)
```

Let `n` be the **zero-based global proposal index**, `N = n_steps`,
`Ti = initial_temperature`, `Tf = final_temperature`, and
`f = n / (N - 1)` within the cooling interval:

| Scheme | Temperature |
| --- | --- |
| `"constant"` | `Ti` |
| `"linear"` | `(1 - f)*Ti + f*Tf` |
| `"exponential"` | `exp((1 - f)*ln(Ti) + f*ln(Tf))` |
| `"logarithmic"` | `Ti / (1 + (Ti/Tf - 1)*ln(n + 1)/ln(N))` |

The logarithmic schedule is a regularized inverse-logarithm law: the additive
offset avoids a singularity at the first proposal. Its rate is chosen to
reach the requested final temperature after `N` attempts. This finite schedule
holds `Tf` thereafter. All cooling schedules start at `Ti` on the first
proposal and reach `Tf` on the `N`th proposal. They require `N >= 2` and
`0 <= Tf <= Ti`; exponential and logarithmic schedules require positive
endpoints. Linear cooling can end at zero, where uphill moves are rejected.
`TemperatureSchedule(600)` is equivalent to constant `temperature=600`.

Temperature changes on **every attempted proposal**, including rejected moves
and calculation failures recorded with `on_error="reject"`. Sampling intervals
affect only stored frames. Repeated `run()` calls and direct `step()` calls use
the same global schedule: `run(400)` followed by `run(600)` follows exactly the
same temperatures and RNG stream as `run(1000)`. A raised calculation error
does not add a step or advance cooling, though the proposal consumed RNG draws.

Custom schedules are callables receiving `n` and returning a finite,
non-negative scalar temperature in kelvin. For an unbounded classical
`1/ln(n)` variation with a nonsingular start, use:

```python
import numpy as np

inverse_log = lambda n: 1200 * np.log(2) / np.log(n + 2)
custom_annealer = MetropolisMonteCarlo(
    sqs.atoms, calculator=EMT, temperature=inverse_log, seed=42,
)
```

Custom callbacks should be deterministic functions of `n`: construction also
evaluates index zero. An invalid callback value raises before generating or
evaluating a candidate, even with `on_error="reject"`.

The current step's temperature determines Metropolis acceptance. Cooling
produces a time-dependent search, so its combined trajectory does not represent
one equilibrium temperature. Choose cooling rates and additional fixed-temperature
sampling according to the calculation. Finite schedules do not guarantee a
global optimum. See the [GSL simulated annealing description](https://www.gnu.org/software/gsl/doc/html/siman.html)
for the role of Boltzmann acceptance and cooling in structure search.

## Relax before comparing energies

```python
relaxed_sampler = MetropolisMonteCarlo(
    sqs.atoms,
    calculator=EMT,
    temperature=600,
    move="mixed",
    distribution="uniform",
    displacement=0.03,
    swap_indices=sqs.active_indices,
    relax=True,
    optimizer="BFGS",
    fmax=0.05,                  # eV/Å
    relax_steps=200,            # maximum optimizer steps per configuration
    seed=42,
)
relaxed = relaxed_sampler.run(20)
relaxed.write("relaxed_mc.extxyz")

from ase.io import write
write("lowest_energy_structure.extxyz", relaxed.best_atoms)
```

The initial configuration and each proposal are optimized before the energy
comparison, keeping their cell fixed. This is a **basin-hopping structure
search** on minimized energies, not a canonical distribution on the original
unrelaxed energy surface. Use `relax=False` for that configurational sampling.
`BasinHopping(atoms, calculator, ...)` enables relaxation and mixed moves by
default; `ChemicalSwapMC(...)` restricts the proposal rule to species swaps and
can also use `relax=True`.

Supported named optimizers are `BFGS`, `FIRE`, and `LBFGS`. An ASE-compatible
optimizer class or factory can be supplied instead. The optimizer receives
`atoms` and `logfile=None`; its `run(fmax=..., steps=...)` must report convergence.
See [ASE geometry optimization](https://docs.ase-lib.org/ase/optimize.html) for
optimizer behavior. Exhausting the step budget raises `RelaxationError`.

## Diagnostics, failures, and trajectory export

Each step records its move, swapped indices (if applicable), acceptance decision,
current and proposed total energies, energy difference, acceptance probability,
temperature, relaxation convergence, and any recorded failure. `result.energies` includes the
initial energy and the current energy after every attempt, including repetitions
for rejected proposals. `result.best_energy` and `best_atoms` track the lowest
accepted state over all runs of that sampler.
`result.temperature` and `sampler.temperature` give the last attempted
temperature; `result.initial_temperature` describes the starting frame of
that run. `result.temperatures` contains only proposal temperatures, matching
the diagnostic rows. The best structure retains the temperature at which
it was accepted, or the initial temperature if it was never improved.

```python
result = sampler.run(1000, sample_interval=10)
result.write("sampled_chain.extxyz")
result.write("final_state.extxyz", trajectory=False)
print(result.to_dataframe().tail())
```

The sampled trajectory contains its initial state and then every tenth chain
state. The final state is always available in `result.atoms`, even if the interval
does not divide the step count. Use `store_trajectory=False` to avoid retaining
structures; the complete per-step log and energy trace remain available.
Do not retain only accepted states for ensemble statistics: doing so removes
residence times and changes the empirical distribution.

Calculation errors, non-finite energies, and optimizer nonconvergence raise by
default. Set `on_error="reject"` to record failed proposals as rejected and
continue from the unchanged current state. Initialization always raises on
failure because no valid chain state exists yet. Check recorded errors before
interpreting a run with failures.

Outputs include `mc_energy`, `mc_temperature`, `mc_temperature_schedule`, `mc_step`, and
`mc_relaxed_proposals` metadata, plus acceptance metadata for sampled steps.
Input SQS, CE, and previous MC diagnostics are retained under `source_sqs_*`,
`source_ce_*`, and `source_mc_*` names so they are not mistaken for correlations
or metrics recomputed on the changed structure. Source provenance is left intact
on the caller's original atoms.

Run the [complete example](https://github.com/seixas-research/oncapintada/blob/main/examples/monte_carlo.py):

```sh
python examples/monte_carlo.py --output-dir results/mc --steps 20
python examples/monte_carlo.py --output-dir results/basin --steps 20 --relax
python examples/monte_carlo.py --output-dir results/annealing --steps 1000 --schedule exponential --temperature 1200 --final-temperature 100
```

The example writes an initial ternary SQS, an energy-bearing chain trajectory,
a diagnostic CSV, and the lowest-energy structure. `MDMC` remains a reserved
interface for molecular-dynamics-assisted proposals and is not implemented.
