"""
A demo for multi-output regression
==================================

The demo is adopted from scikit-learn:

https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_regression_multioutput.html#sphx-glr-auto-examples-ensemble-plot-random-forest-regression-multioutput-py

See :doc:`/tutorials/multioutput` for more information.
"""
import numpy as np
import xgboost as xgb
import argparse
from matplotlib import pyplot as plt


def plot_predt(y, y_predt, name):
    s = 25
    plt.scatter(y[:, 0], y[:, 1], c="navy", s=s,
                edgecolor="black", label="data")
    plt.scatter(y_predt[:, 0], y_predt[:, 1], c="cornflowerblue", s=s,
                edgecolor="black")
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.show()


def main(plot_result: bool):
    """Draw a circle with 2-dim coordinate as target variables."""
    rng = np.random.RandomState(1994)
    X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20, 2))
    y = y - y.min()
    y = y / y.max()

    # Train a regressor on it
    reg = xgb.XGBRegressor(tree_method="hist", n_estimators=64)
    reg.fit(X, y, eval_set=[(X, y)])

    y_predt = reg.predict(X)
    if plot_result:
        plot_predt(y, y_predt, 'multi')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", choices=[0, 1], type=int, default=1)
    args = parser.parse_args()
    main(args.plot == 1)
