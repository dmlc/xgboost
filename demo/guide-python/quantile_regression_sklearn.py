"""
Quantile Regression with the Scikit-Learn Interface
====================================================

    .. versionadded:: 3.4.0

This is a companion to :ref:`sphx_glr_python_examples_prediction_intervals.py`,
showing the same ``reg:quantileerror`` objective through the scikit-learn
estimator interface (:py:class:`~xgboost.XGBRegressor`) instead of the low-level
:py:class:`~xgboost.Booster`/:py:func:`~xgboost.train` API. See the other example
for more background on quantile regression itself, including a comparison against
expectile regression and a squared-error baseline.

.. note::

    The feature is only supported using the Python, R, and C packages. In addition,
    quantile crossing can happen due to limitation in the algorithm.

"""

import numpy as np
from sklearn.model_selection import train_test_split

import xgboost as xgb


def f(x: np.ndarray) -> np.ndarray:
    """The function to predict."""
    return x * np.sin(x)


def main() -> None:
    # Same data-generating process as the Booster-based example, so the two can
    # be compared directly.
    rng = np.random.RandomState(1994)
    X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    expected_y = f(X).ravel()
    sigma = 0.5 + X.ravel() / 10.0
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2.0 / 2.0)
    y = expected_y + noise

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

    # `quantile_alpha` accepts either a single float or an array of quantiles to
    # fit jointly as a multi-output model, exactly as with the low-level API.
    alpha = np.array([0.05, 0.5, 0.95])
    reg = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        tree_method="hist",
        max_depth=5,
        n_estimators=64,
    )
    reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # One column per quantile in `alpha`, in the same order.
    scores = reg.predict(X_test)
    assert scores.shape == (X_test.shape[0], alpha.shape[0])

    lower, median, upper = scores[:, 0], scores[:, 1], scores[:, 2]
    # The lower and upper quantile predictions form a (nominal) 90% interval;
    # with only 1000 samples and imperfect training, don't expect this to land
    # exactly on 0.90 for any particular held-out set.
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    print(f"Empirical coverage of the [0.05, 0.95] interval: {coverage:.3f}")
    print(
        f"Median absolute error vs. true median: "
        f"{np.mean(np.abs(median - f(X_test).ravel())):.3f}"
    )


if __name__ == "__main__":
    main()
