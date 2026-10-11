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
