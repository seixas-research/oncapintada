# Quick start

## Binary thermodynamics

Energy matrices use eV/atom. Row `i`, column `j` describes the dilute solute `i`
in solvent `j`; diagonal values describe the pure components. Binary composition
`x` is the fraction of the second component, so rows correspond to `[1-x, x]`.

```python
import numpy as np
from oncapintada import BinaryAlloy

alloy = BinaryAlloy(
    energy_matrix=[[-3.0, -3.9], [-3.1, -4.0]],
    dilution=0.1,
)
x = np.linspace(0, 1, 101)
temperatures = np.array([300, 600, 900])
enthalpy = alloy.enthalpy_of_mixing(x)  # kJ/mol
entropy = alloy.configurational_entropy(x)  # kJ/(mol K)
gibbs = alloy.gibbs_free_energy_of_mixing(x, temperatures)
# gibbs.shape == (101, 3)
```

Compositions must be finite, non-negative, and sum to one. Entropy is exactly
zero at pure compositions. Gibbs calculations accept finite temperatures at
or above zero; pair QCA requires temperatures strictly above zero. Alternative
energy units are `eV/atom`; corresponding entropy units are `eV/(atom*K)`.

## Multicomponent thermodynamics

```python
from oncapintada import MultiComponentAlloy

ternary = MultiComponentAlloy(
    [[-3.0, -3.9, -4.9], [-3.1, -4.0, -4.9], [-3.1, -4.1, -5.0]],
    dilution=0.1,
)
compositions = ternary.simplex_grid(resolution=10)
gibbs = ternary.gibbs_free_energy_of_mixing(compositions, [300, 600])
```

Columns follow the energy matrix's component order. A 1D multicomponent vector
returns scalar enthalpy and entropy; a 2D composition array returns one value
per row. Gibbs results always have `(n_compositions, n_temperatures)` shape.

## Phase boundaries

```python
import pandas as pd
from oncapintada import PhaseDiagram

binary_gibbs = alloy.gibbs_free_energy_of_mixing(x, temperatures)
phase = PhaseDiagram(pd.DataFrame(binary_gibbs, index=x, columns=temperatures))
binodal = phase.binodal_curve()
spinodal = phase.spinodal_curve()
```

`PhaseDiagram` is a **binary** composition-axis tool. Do not project a ternary grid onto one
column: repeated concentrations do not define a binary free-energy curve.
The composition axis must be strictly increasing and contain at least three
points. `binodal_curve()` uses the full discrete lower convex envelope, and
`spinodal_curve()` uses finite-difference curvature. Both are grid estimates;
refine composition and temperature grids for quantitative phase boundaries.
`critical_point()` returns NaNs when no boundary exists.

## Random and cluster-distinct structures

```python
from ase.build import bulk
from oncapintada import DisorderedAlloyGenerator

parent = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
generator = DisorderedAlloyGenerator(parent, "Cu", ["Cu", "Au", "Ag"], [0.5, 0.25, 0.25], seed=42)
config = generator.generate_configuration()
# config.atoms is an ASE structure; config.site_table records substitutions.
```

Random alloy generation normalizes positive composition weights and uses exact
largest-remainder site counts. It does not optimize SQS correlations. For
ensembles with distinct pair and higher-order correlations, use the
[Cluster Expansion builder](../cluster_expansion.md).

## SQS supercells for random alloys

```python
from oncapintada import generate_sqs

sqs_parent = bulk("Cu", "fcc", a=3.6, cubic=True)
binary_sqs = generate_sqs(
    sqs_parent, {"Cu": 0.5, "Au": 0.5},
    supercell=(2, 2, 2), pair_cutoff=3.7, n_steps=10_000, seed=42,
)
ternary_sqs = generate_sqs(
    sqs_parent, {"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
    supercell=(2, 2, 2), pair_cutoff=3.7, n_steps=10_000, seed=42,
)
binary_sqs.write("CuAu_sqs.extxyz")
ternary_sqs.write("CuAuAg_sqs.extxyz")
print(ternary_sqs.species_counts, ternary_sqs.objective, ternary_sqs.converged)
print(ternary_sqs.correlation_table())
```

Both cells contain 32 sites, with counts `(16, 16)` and `(16, 8, 8)` respectively.
The seed makes the search reproducible. SQS optimizes agreement with random
correlations at one composition; Cluster Expansion generates representatives
with distinct correlations for building an energy model. The [SQS guide](../sqs.md)
explains quality checks, higher-order clusters, sublattices, and calculator use.

## Calculator-backed Monte Carlo

```python
from ase.calculators.emt import EMT
from oncapintada import MetropolisMonteCarlo

mc = MetropolisMonteCarlo(
    ternary_sqs.atoms, calculator=EMT, temperature=600,
    move="mixed", distribution="gaussian", displacement=0.03,
    swap_indices=ternary_sqs.active_indices, seed=42,
)
chain = mc.run(100)
chain.write("mc_trajectory.extxyz")
print(chain.energy, chain.best_energy, chain.acceptance_rate)
```

Metropolis decisions compare total energies in eV. Rejected states remain in
the trajectory. For geometry relaxation before acceptance, set `relax=True`,
`optimizer="BFGS"`, `fmax=0.05`, and `relax_steps=200`; this searches minimized
energy basins. See the [Monte Carlo guide](../monte_carlo.md) for proposal rules,
ASE integration, convergence checks, and exporting the best structure.

For simulated annealing, replace the scalar temperature with a schedule:

```python
from oncapintada import TemperatureSchedule

cooling = TemperatureSchedule(1200, 100, n_steps=1000, scheme="logarithmic")
annealer = MetropolisMonteCarlo(
    ternary_sqs.atoms, calculator=EMT, temperature=cooling,
    move="mixed", distribution="gaussian", displacement=0.03, seed=42,
)
annealed = annealer.run(1000)
annealed.to_dataframe().to_csv("annealing_steps.csv", index=False)
```

Linear and exponential cooling are also available. The schedule advances on
every attempt and continues across repeated runs. After its specified cooling
interval, it holds the final temperature. The Monte Carlo guide defines each
law and shows custom `1/ln(n)` schedules.
