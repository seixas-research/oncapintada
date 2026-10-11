<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/oncapintada" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/oncapintada/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/oncapintada/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/oncapintada/refs/heads/main/logo/logo_light.png" alt="Onça-pintada logo" style="height: auto; width: auto; max-height: 200px;">
  </picture>
</a>
</h1> 

[![License: MIT](https://img.shields.io/github/license/seixas-research/oncapintada?color=green&style=for-the-badge)](LICENSE)    [![PyPI](https://img.shields.io/pypi/v/oncapintada?color=red&style=for-the-badge)](https://pypi.org/project/oncapintada/)

# Onça-pintada
**Onça-pintada** is a Python package designed to compute thermodynamic properties of binary and multicomponent alloys. It enables the calculation of enthalpy, entropy, and Gibbs free energy of mixing within the subregular solution model, as well as the determination of binodal and spinodal curves for phase diagrams.

The package also supports the evaluation of cluster interactions using the quasi-chemical approximation (QCA), allowing for the description of chemical short-range order (CSRO). Additionally, it provides tools to reconstruct atomistic geometries of alloys based on short-range order metrics, facilitating subsequent calculations of their electronic properties.

The Cluster Expansion builder follows Calango Studio's native approach: it
decorates an active sublattice, groups pair/triplet/quadruplet clusters by
distance, and retains configurations with distinct correlation fingerprints.
It supports exact compositions and seeded sampling for binary and
multicomponent alloys, with pure end members preserved during unrestricted
generation. The SQS module optimizes fixed-composition supercells against
random-alloy correlations using simulated annealing, including periodic pair
and optional triplet/quadruplet clusters. No ICET dependency is required.

# Installation

```sh
python -m pip install oncapintada
```

For development from a local checkout (Python 3.10 or later):

```sh
python -m pip install -e '.[dev,docs]'
python -m pytest
```

# Getting started

```python
import numpy as np
from ase.build import bulk
from oncapintada import BinaryAlloy, ClusterExpansionOptions, generate_cluster_expansion

alloy = BinaryAlloy([[-3.0, -3.9], [-3.1, -4.0]], dilution=0.1)
x = np.linspace(0, 1, 101)
gibbs = alloy.gibbs_free_energy_of_mixing(x, [300, 600])  # kJ/mol

parent = bulk("Cu", "fcc", a=3.6, cubic=True)
options = ClusterExpansionOptions(
    active_element="Cu",
    species=("Cu", "Au", "Ag"),
    supercell=(2, 1, 1),
    pair_cutoff=2.7,
    composition=(0.5, 0.25, 0.25),
    seed=42,
)
result = generate_cluster_expansion(parent, options)
result.write("CuAuAg.extxyz")
print(len(result.configs), result.exhaustive)
```

See the [quick start](docs/source/quick_start/index.md),
[Cluster Expansion guide](docs/source/cluster_expansion.md),
[SQS guide](docs/source/sqs.md),
[Monte Carlo guide](docs/source/monte_carlo.md),
[review notes](docs/source/development.md), and
[runnable binary/ternary example](examples/cluster_expansion.py).

Fingerprint equality means correlation equivalence at the selected cutoffs,
not proven space-group symmetry equivalence. Degeneracies are exact only when
`result.exhaustive` is true. A ridge energy fit with held-out validation is
available through `fit_effective_cluster_interactions`; compute consistent
energies for the generated configurations before fitting.

Imports are silent. Use `oncapintada.banner()` when environment diagnostics are
needed. Random alloy, Cluster Expansion, and SQS generation are supported
structure workflows. ASE-backed Metropolis chains support species swaps, random
displacements, and optional geometry relaxation.

# SQS supercells

```python
from oncapintada import generate_sqs

# The same API handles binary and multicomponent alloys.
for name, composition in {
    "CuAu": {"Cu": 0.5, "Au": 0.5},
    "CuAuAg": {"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
}.items():
    sqs = generate_sqs(
        parent,
        composition,
        supercell=(2, 2, 2),  # 32 atoms for this conventional FCC parent
        pair_cutoff=3.7,     # first two FCC neighbor shells, in Å
        n_steps=10_000,
        seed=42,
    )
    sqs.write(f"{name}_sqs.extxyz")
    print(name, sqs.species_counts, sqs.objective, sqs.converged)
```

Inspect the correlation mismatch before using the structure for calculator
runs. Targets use the composition actually realized after site-count rounding.
Annealing returns the best visited decoration; it does not guarantee a globally
optimal SQS. See [examples/sqs.py](examples/sqs.py) for an executable workflow.

# Energy-based Monte Carlo

```python
from ase.calculators.emt import EMT
from oncapintada import MetropolisMonteCarlo

mc = MetropolisMonteCarlo(
    sqs.atoms, calculator=EMT, temperature=600,
    move="mixed", distribution="gaussian", displacement=0.03,
    swap_indices=sqs.active_indices, seed=42,
)
chain = mc.run(100)
chain.write("mc_trajectory.extxyz")
print(chain.energy, chain.best_energy, chain.acceptance_rate)
```

Moves either exchange two unlike species or independently displace all atoms.
Metropolis acceptance uses total calculator energies in eV and temperature in
kelvin. Add `relax=True, optimizer="BFGS"` to optimize each proposal before
comparing energies; this performs a basin-hopping structure search. See the
[executable example](examples/monte_carlo.py) and
[Monte Carlo guide](docs/source/monte_carlo.md) for relaxation and diagnostics.

Use a temperature schedule for simulated annealing:

```python
from oncapintada import TemperatureSchedule

cooling = TemperatureSchedule(1200, 100, n_steps=1000, scheme="exponential")
annealer = MetropolisMonteCarlo(sqs.atoms, calculator=EMT, temperature=cooling)
annealed = annealer.run(1000)
print(annealed.temperatures, annealed.best_energy)
```

Schedules include constant, linear, exponential, and regularized inverse-log
cooling (`scheme="logarithmic"`). They advance on attempted proposals,
including rejections, and continue across successive `run()` calls. A callable
temperature law is also supported. All temperatures are in kelvin.

# License

This is open source code under the [MIT License](LICENSE).
