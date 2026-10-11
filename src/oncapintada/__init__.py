"""Thermodynamics and atomistic structure generation for alloys.

Importing the package is silent and does not change warning filters.
Call :func:`banner` explicitly for environment diagnostics.
"""

from ._version import __version__
from .bonds_counter import BondCountResult, BondsCounter
from .cluster_expansion import (
    ClusterExpansionConfig,
    ClusterExpansionGenerator,
    ClusterExpansionOptions,
    ClusterExpansionResult,
    ClusterOrbit,
    generate_cluster_expansion,
)
from .cluster_expansion_fit import EciFitResult, fit_effective_cluster_interactions
from .disordered_alloy import DisorderedAlloyConfig, DisorderedAlloyGenerator
from .monte_carlo import (
    BasinHopping,
    ChemicalSwapMC,
    MCStep,
    MetropolisMonteCarlo,
    MonteCarloResult,
    RelaxationError,
    TemperatureSchedule,
    metropolis_probability,
)
from .phase_diagram import PhaseDiagram
from .qca import QCABinary
from .sqs import SQSGenerator, SQSOptions, SQSOrbit, SQSResult, generate_sqs
from .subregular_model import BinaryAlloy, MultiComponentAlloy

__all__ = [
    "__version__",
    "banner",
    "BinaryAlloy",
    "MultiComponentAlloy",
    "BondCountResult",
    "BondsCounter",
    "DisorderedAlloyConfig",
    "DisorderedAlloyGenerator",
    "PhaseDiagram",
    "QCABinary",
    "ClusterExpansionConfig",
    "ClusterExpansionGenerator",
    "ClusterExpansionOptions",
    "ClusterExpansionResult",
    "ClusterOrbit",
    "generate_cluster_expansion",
    "EciFitResult",
    "fit_effective_cluster_interactions",
    "SQSGenerator",
    "SQSOptions",
    "SQSOrbit",
    "SQSResult",
    "generate_sqs",
    "MetropolisMonteCarlo",
    "MonteCarloResult",
    "MCStep",
    "ChemicalSwapMC",
    "BasinHopping",
    "RelaxationError",
    "TemperatureSchedule",
    "metropolis_probability",
]


def banner():
    """Print package and dependency versions on request (also under MPI)."""
    import platform
    from importlib.metadata import version

    from ase.parallel import parprint

    parprint(f"Onça-pintada {__version__}")
    parprint(f"Python {platform.python_version()} on {platform.system()}")
    for dependency in ("ase", "numpy", "pandas", "scipy", "matplotlib", "pyyaml"):
        parprint(f"{dependency}: {version(dependency)}")
