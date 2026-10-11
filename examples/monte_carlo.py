"""Generate a ternary SQS, then run ASE-backed sampling or simulated annealing."""

from pathlib import Path

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from oncapintada import MetropolisMonteCarlo, TemperatureSchedule, generate_sqs


def main(
    output_dir=".",
    n_steps=20,
    relax=False,
    seed=42,
    temperature=600,
    schedule="constant",
    final_temperature=100,
):
    temperature_control = (
        temperature
        if schedule == "constant"
        else TemperatureSchedule(
            temperature,
            final_temperature,
            n_steps=n_steps,
            scheme=schedule,
        )
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # This illustrative cell uses a chosen fixed lattice parameter.
    parent = bulk("Cu", "fcc", a=3.9, cubic=True)
    sqs = generate_sqs(
        parent,
        {"Cu": 0.5, "Au": 0.25, "Ag": 0.25},
        supercell=(2, 2, 2),
        pair_cutoff=4.0,
        n_steps=1000,
        seed=seed,
    )
    sqs.write(output_dir / "initial_sqs.extxyz")
    sampler = MetropolisMonteCarlo(
        sqs.atoms,
        calculator=EMT,
        temperature=temperature_control,
        move="mixed",
        distribution="gaussian",
        displacement=0.03,
        swap_indices=sqs.active_indices,
        relax=relax,
        optimizer="BFGS",
        fmax=0.05,
        relax_steps=200,
        seed=seed,
    )
    result = sampler.run(n_steps)
    result.write(output_dir / "mc_trajectory.extxyz")
    result.to_dataframe().to_csv(output_dir / "mc_steps.csv", index=False)
    write(output_dir / "best_structure.extxyz", result.best_atoms)
    print(
        f"SQS objective={sqs.objective:.6g}; "
        f"MC energy={result.initial_energy:.6f} -> {result.energy:.6f} eV; "
        f"best={result.best_energy:.6f} eV; acceptance={result.acceptance_rate:.3f}; "
        f"temperature={result.initial_temperature:g} -> {result.temperature:g} K; "
        f"schedule={schedule}; relaxed_proposals={result.relaxed_proposals}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperature", type=float, default=600, help="Initial temperature in kelvin"
    )
    parser.add_argument(
        "--schedule",
        choices=("constant", "linear", "exponential", "logarithmic"),
        default="constant",
        help="Temperature schedule over all attempted steps",
    )
    parser.add_argument(
        "--final-temperature",
        type=float,
        default=100,
        help="Final cooling temperature in kelvin; unused for constant temperature",
    )
    parser.add_argument(
        "--relax", action="store_true", help="Relax every proposal before Metropolis comparison"
    )
    args = parser.parse_args()
    main(
        args.output_dir,
        args.steps,
        args.relax,
        args.seed,
        args.temperature,
        args.schedule,
        args.final_temperature,
    )
