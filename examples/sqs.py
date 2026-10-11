"""Generate binary/ternary FCC SQS cells: python examples/sqs.py --output-dir results."""

from pathlib import Path

from ase.build import bulk

from oncapintada import BondsCounter, generate_sqs


def main(output_dir=".", n_steps=10_000, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent = bulk("Cu", "fcc", a=3.6, cubic=True)
    for name, composition in (
        ("CuAu", {"Cu": 0.5, "Au": 0.5}),
        ("CuAuAg", {"Cu": 0.5, "Au": 0.25, "Ag": 0.25}),
    ):
        result = generate_sqs(
            parent,
            composition,
            supercell=(2, 2, 2),
            pair_cutoff=3.7,
            n_steps=n_steps,
            seed=seed,
        )
        result.write(output_dir / f"{name}_sqs.extxyz")
        result.correlation_table().to_csv(output_dir / f"{name}_sqs_correlations.csv", index=False)
        order = BondsCounter(
            atoms=result.atoms, cutoff=2.7, subset_indices=result.active_indices
        ).warren_cowley_parameters()
        print(
            f"{name}: {len(result.atoms)} atoms, counts={result.species_counts}, "
            f"objective={result.initial_objective:.6g} -> {result.objective:.6g}, "
            f"converged={result.converged}, swaps={result.iterations}"
        )
        print(order.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.output_dir, args.steps, args.seed)
