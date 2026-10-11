import numpy as np
import pytest

from oncapintada import fit_effective_cluster_interactions


def test_known_linear_energy_with_redundant_features():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(30, 3))
    x = np.column_stack((x, x[:, 0], np.ones(30)))
    y = 2.3 + x[:, 0] * 0.5 - x[:, 1] * 0.2
    model = fit_effective_cluster_interactions(x, y, alphas=[0, 1, 100])
    assert model.alpha == 0
    assert model.cv_score < 1e-12
    assert model.rmse < 1e-12
    assert model.intercept == pytest.approx(2.3)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-12)
    test = rng.normal(size=(5, 3))
    test = np.column_stack((test, test[:, 0], np.ones(5)))
    np.testing.assert_allclose(
        model.predict(test), 2.3 + test[:, 0] * 0.5 - test[:, 1] * 0.2, atol=1e-12
    )


def test_held_out_error_reveals_interpolation_overfit():
    x = np.eye(6)
    y = np.arange(6, dtype=float)
    model = fit_effective_cluster_interactions(x, y, alphas=[0])
    assert model.rmse < 1e-12
    assert model.cv_score > 1


def test_kfold_reproducibility_and_feature_scaling():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(24, 4))
    y = x[:, 0] + rng.normal(scale=0.1, size=24)
    first = fit_effective_cluster_interactions(x, y, cv_folds=4, seed=42)
    second = fit_effective_cluster_interactions(x * [1e-3, 100, 5, 0.1], y, cv_folds=4, seed=42)
    np.testing.assert_allclose(first.cv_path, second.cv_path, atol=1e-12)
    np.testing.assert_allclose(first.predictions, second.predictions, atol=1e-12)
    assert np.isfinite(first.coefficients).all()


@pytest.mark.parametrize(
    "x,y,kwargs",
    [
        ([[1]], [1], {}),
        ([[1], [2]], [1], {}),
        ([[np.nan], [2]], [1, 2], {}),
        ([[1], [2]], [1, np.inf], {}),
        ([[1], [2]], [1, 2], {"alphas": []}),
        ([[1], [2]], [1, 2], {"alphas": [-1]}),
        ([[1], [2]], [1, 2], {"cv_folds": 1}),
        ([[1], [2]], [1, 2], {"cv_folds": 3}),
        ([[1], [2]], [1, 2], {"labels": []}),
    ],
)
def test_invalid_fitting_inputs(x, y, kwargs):
    with pytest.raises(ValueError):
        fit_effective_cluster_interactions(x, y, **kwargs)
