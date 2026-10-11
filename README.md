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
generation. No ICET dependency is required.

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
[review notes](docs/source/development.md), and
[runnable binary/ternary example](examples/cluster_expansion.py).

Fingerprint equality means correlation equivalence at the selected cutoffs,
not proven space-group symmetry equivalence. Degeneracies are exact only when
`result.exhaustive` is true. A ridge energy fit with held-out validation is
available through `fit_effective_cluster_interactions`; compute consistent
energies for the generated configurations before fitting.

Imports are silent. Use `oncapintada.banner()` when environment diagnostics are
needed. Existing SQS and Monte Carlo algorithms remain unimplemented; random
alloy and Cluster Expansion generation are the supported structure workflows.

# License

This is open source code under the [MIT License](LICENSE).
