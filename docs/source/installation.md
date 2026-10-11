# Installation

Onça-pintada requires Python 3.10 or later.

## From PyPI

```sh
python -m pip install oncapintada
```

The Cluster Expansion, SQS, and Monte Carlo modules use the package's normal
runtime dependencies, including NumPy and ASE. They need no ICET installation.

## From source

```sh
git clone https://github.com/seixas-research/oncapintada.git
cd oncapintada
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,docs]'
```

On Windows, activate with `.venv\Scripts\activate`.

## Development checks

```sh
python -m pytest
ruff check src test examples/cluster_expansion.py examples/sqs.py examples/monte_carlo.py
ruff format --check src test examples/cluster_expansion.py examples/sqs.py examples/monte_carlo.py
sphinx-build -W --keep-going -b html docs/source docs/build/html
python -m build
```

Tests use the source layout configured in `pyproject.toml`. Importing the
package is silent; call `oncapintada.banner()` explicitly to print environment
and dependency versions. Pytest, lint, build, and documentation tools are
optional development dependencies and are not required to use the library.
