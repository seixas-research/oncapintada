# Special quasirandom structures (SQS)

SQS generation produces an ideal-lattice supercell whose selected occupational
correlations approximate a random solid solution. The native optimizer matches
species-tuple probabilities on periodic distance-grouped clusters. It follows
Calango Studio's [`SqsGenerator`](https://github.com/seixas-research/calango-studio/blob/main/src/core/SqsGenerator.hpp)
method: fixed-composition species swaps, a Metropolis acceptance rule, and a
geometric simulated-annealing schedule. NumPy and ASE are the only computational
dependencies of this implementation; ICET is not required.

Matching random-alloy correlation targets is the defining SQS task; the
[ICET SQS tutorial](https://icet.materialsmodeling.org/advanced_topics/sqs_generation.html)
provides background and alternative objective measures. Onça-pintada uses a
species-tuple histogram objective, rather than ICET's orthogonal cluster-vector
objective, and does not claim interchangeable coefficients or quality scores.

## Binary and multicomponent supercells

```python
from ase.build import bulk
from oncapintada import generate_sqs

parent = bulk("Cu", "fcc", a=3.6, cubic=True)

binary = generate_sqs(
    parent,
    {"Cu": 0.5, "Au": 0.5},
    supercell=(2, 2, 2),
    pair_cutoff=3.7,
    n_steps=10_000,
    restarts=3,
    seed=42,
)
ternary = generate_sqs(
    parent,
    {"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
    supercell=(2, 2, 2),
    pair_cutoff=3.7,
    n_steps=10_000,
    restarts=3,
    seed=42,
)

binary.write("CuAu_sqs.extxyz")
ternary.write("CuAuAg_sqs.extxyz")
print(binary.species_counts)   # (16, 16)
print(ternary.species_counts)  # (16, 8, 8)
print(ternary.initial_objective, ternary.objective, ternary.converged)
```

The conventional FCC parent has four atoms, so repeating it 2×2×2 produces
32 atoms. The 3.7 Å cutoff includes the first two FCC pair shells at this lattice
parameter. Species insertion order determines both output count order and tie
rounding. The same call supports four or more species; use actual chemical
symbols and fractions that sum to one. A structure filename can replace
`parent`, using the last frame under ASE's usual default read behavior.

See [examples/sqs.py](https://github.com/seixas-research/oncapintada/blob/main/examples/sqs.py)
for an executable workflow:

```sh
python examples/sqs.py --output-dir results/sqs --steps 10000 --seed 42
```

It writes an extended XYZ structure and correlation CSV for each alloy, and
prints the initial/final mismatch and nearest-neighbor Warren–Cowley parameters.

## Inspect quality and use the structure

```python
from oncapintada import BondsCounter

print(ternary.composition)       # realized active-site fractions
print(ternary.max_error)         # largest individual tuple mismatch
print(ternary.correlation_table())

sro = BondsCounter(
    atoms=ternary.atoms,
    cutoff=2.7,                 # nearest shell only
    subset_indices=ternary.active_indices,
).warren_cowley_parameters()
print(sro)

# The returned atoms are ready for an ASE calculator or a DFT input export.
from ase.calculators.emt import EMT

atoms = ternary.atoms.copy()
atoms.calc = EMT()               # inexpensive illustrative calculator
energy_per_atom = atoms.get_potential_energy() / len(atoms)
print(energy_per_atom)           # eV/atom for this ideal-lattice decoration
```

Choose the calculator and relaxation protocol appropriate to the alloy for
production energies. SQS generation optimizes occupations without changing
positions or cell vectors. Geometry relaxation is a subsequent calculator task.
Correlations should be inspected on the ideal occupational lattice; reordered
or relaxed structures must be mapped back to that lattice for comparison.

`converged` means the weighted correlation mismatch is at or below `tolerance`.
A finite cell can have an unavoidable residual; a completed run with
`converged=False` still returns the best decoration found. Compare several seeds,
cell sizes, cutoffs, and search budgets when selecting a production SQS.
Neither convergence nor a small training-range mismatch establishes global
optimality or agreement at unselected distances.

## Preserve a spectator sublattice

```python
from oncapintada import SQSGenerator, SQSOptions

oxide = bulk("NiO", "rocksalt", a=4.2, cubic=True)
options = SQSOptions(
    composition={"Ni": 0.5, "Co": 0.25, "Fe": 0.25},
    active_element="Ni",
    supercell=(2, 2, 2),
    pair_cutoff=3.1,
    seed=42,
)
generator = SQSGenerator(oxide, options)
oxide_sqs = generator.generate()
# Ni-sublattice sites are decorated; all oxygen sites retain their species.
print(generator.evaluate(oxide_sqs.atoms))  # recompute the same objective
oxide_sqs.write("NiCoFeO_sqs.extxyz")
```

With `active_element=None` (the default), all parent sites are active. Specify
an element to preserve other sublattices. Concentrations and reported correlations
refer only to active sites. Fractional counts are apportioned by largest
remainder, with ties following species order. The random targets use the
**realized** fractions after rounding, and both requested and realized
compositions are reported. At least two species must survive rounding; otherwise
use a larger cell or a representable composition.

## Higher-order correlations

```python
higher_order = generate_sqs(
    parent,
    {"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
    supercell=(2, 2, 2),
    pair_cutoff=3.7,
    triplet_cutoff=2.7,
    quadruplet_cutoff=2.7,
    n_steps=20_000,
    seed=42,
)
```

Pairs are enabled by default. A zero cutoff disables an order. Triplets and
quadruplets require every mutual distance in a compact periodic cluster to fit
the corresponding cutoff. Unlike the finite minimum-image Cluster Expansion
builder, SQS geometry retains multiple periodic images of a distinct site pair.
The [ASE neighbor-list documentation](https://docs.ase-lib.org/ase/neighborlist.html)
explains these image conventions. Higher-order clusters are anchored on their
lowest base-site index, counted once per translation, and grouped by rounded,
sorted distance signatures. These are distance groups, not certified space-group
orbits. An enabled order without any clusters is rejected.

The species tuples are unordered. For realized fractions `c_s`, the random
target of a tuple of order `m`, containing `n_s` occurrences of each species, is
`m! / product(n_s!) * product(c_s**n_s)`. For binary pairs these are `c_A**2`,
`2*c_A*c_B`, and `c_B**2`. Each orbit's histogram and target sum to one.
The objective sums the weighted absolute differences of observed and target
probabilities. Point terms already match the realized composition exactly.
Shorter-radius groups within each order receive `order_weight / (rank + 1)`;
rank starts at zero. Default order weights are 1, 0.5, and 0.25 for pairs,
triplets, and quadruplets respectively.

## Controls and limits

| Option | Default | Meaning |
| --- | --- | --- |
| `supercell` | `(2, 2, 2)` | Positive repetitions along the parent cell vectors |
| `active_element` | `None` | All sites, or the chosen element's sublattice |
| `pair_cutoff` | `4.0` | Pair range in Å |
| `triplet_cutoff` / `quadruplet_cutoff` | `0.0` | Higher-order ranges in Å; zero disables them |
| `distance_tolerance` | `0.05` | Rounded distance-bin width in Å |
| `n_steps` | `10000` | Swap-attempt budget per restart |
| `restarts` | `3` | Independent starting decorations |
| `seed` | `42` | Reproducible RNG seed; `None` requests fresh randomness |
| `start_temperature` / `end_temperature` | `0.1` / `0.0001` | Dimensionless annealing scales, not kelvin |
| `tolerance` | `1e-8` | Weighted objective threshold for early termination |
| `pair_weight` / `triplet_weight` / `quadruplet_weight` | `1` / `0.5` / `0.25` | Positive order weights |
| `max_clusters` | `200000` | Hard per-order cluster limit; exceeding it raises an error |

The supercell must have at least one periodic direction with independent,
nonzero cell vectors. Partial periodicity and skew cells are supported.
Cutoffs that reach a site's own periodic image are rejected; increase the cell
size for that interaction range. Higher-order enumeration and large neighbor
ranges can become expensive. Cluster lists are never silently truncated.

The optimizer preserves integer species counts through swaps, returns the best
state across all restarts, and retallies its correlations independently before
reporting them. `iterations` and `accepted_moves` record actual swap attempts
and accepted moves; early convergence can leave part of the budget unused.
`history` contains strictly improving best objective values.

## Choosing a structure workflow

| Workflow | Purpose |
| --- | --- |
| `DisorderedAlloyGenerator` | Random exact-composition decoration, without correlation optimization |
| `generate_sqs` | Optimize one fixed-composition supercell toward random-alloy correlations |
| `generate_cluster_expansion` | Generate correlation-distinct representatives to evaluate and fit an energy model |
| `MetropolisMonteCarlo` | Sample calculator energies or search relaxed structures with composition-preserving moves |

An SQS can be evaluated by a calculator or used as another training configuration
for Cluster Expansion. To obtain its CE features, construct a
`ClusterExpansionGenerator` on the same parent with the same `supercell` and
active element, then call `ce_builder.correlation(sqs.atoms)`. Do not substitute
SQS probability vectors for CE's raw histogram features; their periodic cluster
conventions and normalization differ. Subregular thermodynamic models still
require their usual energy matrices; an SQS structure alone does not supply
mixing energies, entropy, or a phase diagram.

Use an SQS as the initial state for the [Monte Carlo workflow](monte_carlo.md)
to explore chemical occupations and atomic displacements with ASE energies
and optional geometry relaxation.
