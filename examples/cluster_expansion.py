"""Generate binary and ternary alloy ensembles: python examples/cluster_expansion.py."""

from pathlib import Path

from ase.build import bulk

from oncapintada import ClusterExpansionOptions, generate_cluster_expansion


def main(output_dir="."):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent = bulk("Cu", "fcc", a=3.6, cubic=True)
    for name, species, composition in (
        ("CuAu", ("Cu", "Au"), None),
        ("CuAuAg", ("Cu", "Au", "Ag"), (0.5, 0.25, 0.25)),
    ):
        options = ClusterExpansionOptions(
            active_element="Cu",
            species=species,
            supercell=(2, 1, 1),
            pair_cutoff=2.7,
            triplet_cutoff=2.7,
            composition=composition,
            max_configs=200,
            max_enumeration=10_000,
            seed=42,
        )
        result = generate_cluster_expansion(parent, options)
        result.write(output_dir / f"{name}.extxyz")
        print(
            f"{name}: {len(result.configs)} representatives; "
            f"{result.enumerated}/{result.total_decorations} examined; "
            f"exhaustive={result.exhaustive}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    main(parser.parse_args().output_dir)
