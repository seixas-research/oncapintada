# Review and development notes

The refactoring preserves the binary and multicomponent subregular formulas
while strengthening validation and adding a native occupational Cluster
Expansion workflow. The original tests remain in place and are complemented
by numerical and atomistic regression tests.

## Corrected behavior

| Area | Finding | Correction |
| --- | --- | --- |
| Imports | Every import printed a system banner, suppressed all warnings, and required pytest | Silent imports, explicit `banner()`, pytest moved to the development extra |
| Energy matrices | List inputs failed, non-finite matrices were accepted, and binary models accepted more components | Validated copied matrices, read-only matrix storage, strict 2×2 binary shape |
| Mixing entropy | Clipping hid invalid compositions and gave nonzero pure-component entropy | Shared composition validation and exact zero-safe `x log x` |
| Mixing enthalpy | Adding epsilon to every pair denominator biased all normalized values | Masked division only at zero-occupation pairs |
| Composition rounding | Different builders used different rounding methods | Shared largest-remainder apportionment with deterministic tie handling |
| Substitution masks | Duplicate or multidimensional indices corrupted counts and site tables | Validated distinct 1D indices and correctly sized boolean masks |
| Bond analysis | Periodic images collapsed onto atom-index pairs; subset updates left stale pair lists | Image-aware undirected bonds and updated subset species |
| Pair SRO | Global bond proportions failed for unequal species coordination | Species-conditional directed-neighbor probabilities |
| Square-lattice model | Self-image bonds counted twice; open boundary settings were ignored | Half the directed neighbor energy and boundary-aware neighbor lists |
| Pair QCA | Zero interaction caused division by zero; endpoints and strong interactions produced invalid probabilities; direct calls missed prerequisites | Stable analytic limits, positive-temperature validation, zero-safe entropy, lazy prerequisites |
| Phase boundaries | Sparse probes could miss narrow gaps; convex data could be reported as coexistence; empty critical curves crashed | Full discrete lower convex envelope and empty-result handling |
| Vibrations | Empty construction failed; temperatures did not broadcast; cached values were reused at other temperatures; DOS refresh left stale conversions; deprecated integration | Validated DOS updates, vector temperatures, zero-temperature limits, stable occupation functions, refreshed results, SciPy trapezoidal integration |
| Units | Rounded eV-to-kJ/mol conversion disagreed with the gas and Boltzmann constants | Consistent conversion factor |
| Packaging and docs | pytest was a runtime dependency; quick start was empty; docs imported unrelated mocks and a second theme | Development and docs extras, runnable workflows, source-based Sphinx configuration and API pages |

These corrections intentionally change results that depended on the defects:
periodic small-cell bond counts, a 1×1 periodic square's energy, unequal-degree
SRO, binodal detection, endpoint entropy, and invalid-input acceptance. Update
energy matrices by assignment rather than mutating the stored read-only array.
Random-builder tie rounding now follows input species order.

`BondsCounter.warren_cowley_parameters()` keeps one row per lexically sorted
pair; its first species is the central species. On unequal-coordination
structures the two directional SRO parameters need not coincide. QCA pair
parameters for an absent species are reported as NaN because they are undefined.
QCA's composition axis denotes fraction of species A, whereas `BinaryAlloy`'s
scalar axis denotes the second component's fraction; symmetric mixing limits
agree, but names and orientation should be set consistently by the caller.

## Scientific limits and future work

The [Cluster Expansion page](cluster_expansion.md) describes the finite-cell
fingerprint approximation, exploration budgets, population semantics, and
ridge model's non-unique histogram coefficients. Treat the generated ensemble
as structures to evaluate with a calculator, not as an automatically converged
energy model. The CE builder does not automate calculator batches,
formation-energy hull plotting, LASSO, or ARD. Composition-preserving simulated
annealing is available for [SQS generation](sqs.md), and ASE calculator-backed
Metropolis chains support energy evaluation and relaxed structure searches.

`PhaseDiagram` supports binary composition curves; multicomponent lower hulls
need a separate multidimensional implementation. Its derivatives and phase
boundaries are estimates on the supplied grid. `critical_point()` averages the
highest-temperature boundary compositions and reports that sampled temperature;
it does not extrapolate an exact critical point, particularly for multiple gaps.

Harmonic vibrational properties exclude imaginary and zero frequencies from
thermal integrals. Negative modes signal that harmonic thermodynamics may not
be appropriate for that structure. DOS normalization controls whether outputs
are per atom or per cell; the library does not invent that normalization.

The reverse Monte Carlo interface, polymorph containers, vibrational subregular
container, and empty view/YAML/pair-distribution modules remain planned work.
The [Monte Carlo module](monte_carlo.md) now provides energy-based Metropolis
chains with species swaps, whole-structure displacements, and optional
ASE relaxation. Molecular-dynamics-assisted `MDMC` remains unavailable.
Use random, Cluster Expansion, SQS, or Monte Carlo generation as appropriate.
The previous `generator.py` SQS placeholders now resolve to `sqs.SQSGenerator`;
the supported API takes an ASE parent (or filename) and `SQSOptions`, rather
than the old unimplemented lattice-name constructor.

## Validation

The suite checks original thermodynamic formulas and conversion behavior,
independent brute-force binary and ternary fingerprints, known class populations,
higher-order feature normalization, finite budgets, endpoint preservation,
exact compositions, skew and partially periodic cells, spectator preservation,
ASE export round trips, held-out ridge fitting, limiting QCA probabilities,
convex-envelope phase boundaries, and harmonic thermal limits. SQS tests check
periodic cluster multiplicities, random-alloy targets, and independently
enumerated optimum errors. Monte Carlo tests check a known two-state Boltzmann
population, rejection residence times, both noise distributions, calculator
failures, reproducible continuation, and ASE relaxation with analytic forces.
Temperature-schedule checks cover known linear, exponential, and logarithmic
temperatures, endpoint holding, zero-temperature rejection, resumed annealing,
per-step acceptance probabilities, and exported temperature traces.
Tests also verify module importability and that user warnings remain visible.

Run the checks listed in [Installation](installation.md) before changing the
scientific kernels. Current ASE can emit NumPy 2.5 deprecation warnings during
supercell repetition; those dependency warnings remain visible rather than
being globally suppressed by Onça-pintada.
