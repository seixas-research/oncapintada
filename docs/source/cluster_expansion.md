# Cluster Expansion

The builder generates substitutional alloy configurations on an ASE parent
structure. Its approach follows Calango Studio's native
[`ClusterExpansion.cpp`](https://github.com/seixas-research/calango-studio/blob/main/src/core/ClusterExpansion.cpp):
repeat the parent cell, select every site of an active element, group geometric
clusters by distance, and retain representatives with distinct occupational
correlation fingerprints. It supports binary, ternary, and higher-component
alloys without ICET or a separate crystallography dependency.

## Generate a binary ensemble

```python
from ase.build import bulk
from oncapintada import ClusterExpansionOptions, generate_cluster_expansion

parent = bulk("Cu", "fcc", a=3.6, cubic=True)
options = ClusterExpansionOptions(
    active_element="Cu",
    species=("Cu", "Au"),
    supercell=(2, 1, 1),
    pair_cutoff=2.7,        # Å; nearest-neighbor clusters
    max_configs=200,
    seed=42,
)
result = generate_cluster_expansion(parent, options)
result.write("CuAu.extxyz")
print(len(result.configs), result.enumerated, result.exhaustive)
```

The parent is copied. Only the selected active element is substituted; all
spectator species, positions, cell, and periodic boundary conditions are
preserved after repetition. You may pass a structure filename instead of an ASE
`Atoms` object. The exported multi-frame extended XYZ includes `ce_species`,
`ce_species_counts`, `ce_correlation`, `ce_degeneracy`, and `ce_exhaustive` metadata.

Unrestricted generation always includes one pure **active-sublattice** decoration
per substitution species. Spectators remain present at these endpoints: in an
oxide, these are pure metal-sublattice end members, not isolated elemental metals.
Both `max_configs` and `max_enumeration` must accommodate every endpoint.

## Fix a multicomponent composition

```python
from oncapintada import ClusterExpansionGenerator

options = ClusterExpansionOptions(
    active_element="Cu",
    species=("Cu", "Au", "Ag"),
    supercell=(2, 1, 1),
    pair_cutoff=2.7,
    triplet_cutoff=2.7,
    composition={"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
    max_configs=200,
    max_enumeration=10_000,
    seed=42,
)
builder = ClusterExpansionGenerator(parent, options)
result = builder.generate()
result.write("CuAuAg.extxyz")
for config in result.configs:
    print(config.species_counts, config.composition, config.degeneracy)
```

Composition values are fractions of **active sites**, must be finite and
non-negative, and must sum to one. A sequence follows the declared species
order; a mapping must contain every species. Counts use largest-remainder
rounding and always sum to the active-site count. Equal remainders are resolved
in species order. For example, four sites shared equally among three species
produce counts `(2, 1, 1)`. Fixed-composition generation does not add endpoints
that would violate those counts. If elemental or end-member energy references
are needed, compute them separately.

## Clusters, fingerprints, and coverage

| Option | Default | Meaning |
| --- | --- | --- |
| `supercell` | `(2, 2, 2)` | Positive integer repetitions along each cell vector |
| `pair_cutoff` | `4.0` | Pair radius in Å; zero disables this order |
| `triplet_cutoff` | `0.0` | Triplet radius in Å |
| `quadruplet_cutoff` | `0.0` | Quadruplet radius in Å |
| `distance_tolerance` | `0.05` | Width in Å of rounded distance bins |
| `max_configs` | `200` | Maximum retained representatives |
| `max_enumeration` | `200000` | Maximum candidate exploration budget, including reserved endpoints |
| `max_clusters` | `200000` | Hard limit on clusters per order; exceeding it raises an error |
| `seed` | `42` | Seed for random sampling; `None` requests fresh randomness |

At least one cluster order must be enabled. Clusters span distinct active sites
and every mutual minimum-image distance must fit the order's cutoff. Periodic
self-images and multiple images of the same site pair are excluded from this
finite-supercell basis. ASE handles minimum-image distances in orthogonal and
skew cells, including partial periodicity. Use a sufficiently large supercell
for the interaction range you wish to represent; this is not an infinite-lattice
orbit construction.

Clusters are grouped by their rounded, sorted pair-distance signatures. The
integer fingerprint contains species counts followed by one species-tuple
histogram per orbit. Pair, triplet, and quadruplet species tuples are sorted.
`result.correlation_labels` names every column of `result.correlations`.
Each orbit's histogram sums to `orbit.multiplicity`. Higher-order histograms
store only canonical tuple bins rather than Calango's larger mixed-radix arrays
with unused zero bins; fingerprint equality is unchanged.

**Matching fingerprints mean correlation equivalence at the selected cutoffs.**
They do not establish space-group symmetry equivalence. Distinct structures
can share truncated correlations, and distance-based grouping may merge
geometries that have the same sorted distances. Changing cutoffs or tolerance
can change the representatives. A cutoff that contains no clusters reduces
deduplication to the point terms (composition).

Unrestricted spaces contain `K**M` decorations for `K` species and `M` active
sites. Fixed-composition spaces contain the multinomial number of distinct
permutations. The builder enumerates a space exhaustively when it fits the
exploration budget; otherwise it samples occupations with a seeded generator.
Fixed-composition sampling shuffles the exact species multiset.

Inspect coverage before using populations in thermodynamics:

- `total_decorations`: size of the complete occupational space.
- `enumerated`: number of candidate fingerprints examined, including duplicate
  sample hits and any final candidate rejected by the configuration cap.
- `sampled`: random sampling was used.
- `truncated`: exploration ended with incomplete coverage, through sampling or
  the representative cap.
- `exhaustive`: neither sampling nor truncation occurred.
- `config.degeneracy`: observed candidate hits for that fingerprint. These are
  exact class populations only for an exhaustive result. In sampled results,
  endpoint populations are seeded once and mixed-class hits are sample counts.

The draw budget is also bounded, so skipped pure draws can make `enumerated`
smaller than `max_enumeration`. Cluster limits raise an error rather than
silently retaining incomplete geometric orbits. Reduce the supercell or cutoffs
when a limit is exceeded. Distance storage grows quadratically with active-site
count; higher-order cluster counts can grow much faster.

## Fit an energy model

Generation supplies configurations for DFT or an ASE calculator. It does not
compute their energies or relax them. Once you have one consistent energy in
**eV/atom** per representative, fit a ridge model:

```python
from oncapintada import fit_effective_cluster_interactions

# energies_eV_per_atom must align with result.configs.
model = fit_effective_cluster_interactions(
    result.correlations,
    energies_eV_per_atom,
    labels=result.correlation_labels,
    cv_folds=5,   # omit for leave-one-out; at least 5 configurations needed here
)
print(model.rmse, model.cv_score, model.alpha)
predicted = model.predict(result.correlations)
```

The model standardizes features separately inside each training fold, leaves
the intercept unpenalized, solves ridge regression by SVD, and selects the
regularization strength with minimum held-out RMSE. You can supply `alphas`
to control the search. `rmse` measures the final training error; `cv_score`
measures held-out error. Duplicate or related configurations across folds can
make validation optimistic. Calango additionally offers LASSO and ARD; this
Python implementation currently provides ridge fitting only.

The raw histogram basis is redundant. Coefficients are expressed per input
column and are not unique conventional orthogonal-basis ECIs. Predictions
apply to this basis and supercell, with consistent energies and geometry;
do not transfer raw count coefficients to another cell size. Use
`builder.correlation(atoms)` to fingerprint another decoration on the identical
cell and atom ordering. It rejects changed spectator species and changed
geometry. Relaxed training structures must be mapped to their original ideal
occupations before correlating them.

See [the runnable example](https://github.com/seixas-research/oncapintada/blob/main/examples/cluster_expansion.py)
for binary and multicomponent generation. Automated tests include independent
brute-force fingerprints and exact populations on a small square lattice.
