"""SVD ridge fitting and held-out validation for cluster correlations.

This provides the linear fitting step of Calango's alloy workflow, using
ridge regularization. Coefficients refer to the supplied histogram columns;
redundant histogram features do not have uniquely identifiable physical ECIs.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import positive_integer


def _matrix(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Correlations must be a non-empty, finite 2D matrix.")
    return values


def _solve(x, y, alpha):
    # Recompute preprocessing for every training fold to avoid CV leakage.
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    centered = (x - mean) / scale
    target = y - y.mean()
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if alpha == 0:
        threshold = np.finfo(float).eps * max(centered.shape) * singular.max(initial=0)
        factors = np.divide(1, singular, out=np.zeros_like(singular), where=singular > threshold)
    else:
        factors = singular / (singular**2 + alpha)
    coefficients = (vt.T @ (factors * (u.T @ target))) / scale
    return coefficients, float(y.mean() - mean @ coefficients)


@dataclass(frozen=True)
class EciFitResult:
    """Linear histogram model in the target energy units (normally eV/atom)."""

    coefficients: np.ndarray
    intercept: float
    alpha: float
    rmse: float
    cv_score: float
    alpha_path: np.ndarray
    cv_path: np.ndarray
    predictions: np.ndarray
    residuals: np.ndarray
    labels: tuple[str, ...]

    def predict(self, correlations):
        """Predict energies from fingerprints with the same feature layout."""
        correlations = _matrix(correlations)
        if correlations.shape[1] != len(self.coefficients):
            raise ValueError("Prediction columns must match the fitted correlation basis.")
        return correlations @ self.coefficients + self.intercept


def fit_effective_cluster_interactions(
    correlations,
    energies,
    *,
    alphas=None,
    cv_folds=None,
    seed=42,
    labels=None,
):
    """Select ridge strength by held-out RMSE, then fit the complete dataset.

    Energies must have one value per configuration, on a consistent per-atom
    scale. Standardization and an unpenalized intercept are learned separately
    inside every fold. ``cv_folds=None`` performs leave-one-out; an integer
    performs seeded k-fold CV. No inverse or normal equations are formed.
    Pass ``result.correlations`` and ``result.correlation_labels`` from the
    builder. Use independent configurations for meaningful CV; duplicate
    fingerprints across folds inflate apparent predictive quality.
    """
    x = _matrix(correlations)
    y = np.asarray(energies, dtype=float)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("Energies must be finite with one value per configuration.")
    if len(x) < 2:
        raise ValueError("At least two configurations are required for cross-validation.")
    if alphas is None:
        alphas = np.logspace(-8, 2, 25)
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or len(alphas) == 0 or not np.isfinite(alphas).all() or np.any(alphas < 0):
        raise ValueError("alphas must be a non-empty sequence of finite non-negative values.")
    if labels is None:
        labels = tuple(f"column {i}" for i in range(x.shape[1]))
    else:
        labels = tuple(labels)
        if len(labels) != x.shape[1]:
            raise ValueError("There must be exactly one label per correlation column.")
    folds = len(x) if cv_folds is None else positive_integer(cv_folds, "cv_folds")
    if not 2 <= folds <= len(x):
        raise ValueError("cv_folds must be between 2 and the number of configurations.")
    indices = np.random.default_rng(seed).permutation(len(x))
    splits = np.array_split(indices, folds)
    scores = []
    for alpha in alphas:
        held_out = np.empty(len(y))
        for test in splits:
            train = np.ones(len(x), dtype=bool)
            train[test] = False
            coefficients, intercept = _solve(x[train], y[train], alpha)
            held_out[test] = x[test] @ coefficients + intercept
        scores.append(float(np.sqrt(np.mean((held_out - y) ** 2))))
    selected = int(np.argmin(scores))
    coefficients, intercept = _solve(x, y, alphas[selected])
    predictions = x @ coefficients + intercept
    residuals = predictions - y
    return EciFitResult(
        coefficients,
        intercept,
        float(alphas[selected]),
        float(np.sqrt(np.mean(residuals**2))),
        scores[selected],
        alphas.copy(),
        np.asarray(scores),
        predictions,
        residuals,
        labels,
    )
