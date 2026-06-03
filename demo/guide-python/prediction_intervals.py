"""
Prediction Intervals with Quantile and Expectile Regression
===========================================================

    .. versionadded:: 2.0.0

The script is inspired by this awesome example in sklearn:
https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html

.. note::

    The feature is only supported using the Python, R, and C packages. In addition, quantile
    crossing can happen due to limitation in the algorithm.

This example also trains ``reg:expectileerror``. Expectiles are asymmetric means,
not percentiles, but they can be used to construct tail-sensitive bands around the
conditional mean.

    .. versionadded:: 3.3.0

"""

import argparse
from typing import Dict

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def f(x: np.ndarray) -> np.ndarray:
    """The function to predict."""
    return x * np.sin(x)


def make_dataset(rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Generate heteroscedastic data with asymmetric noise."""
    # The data generating process is adapted from the sklearn example.
    X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    expected_y = f(X).ravel()

    sigma = 0.5 + X.ravel() / 10.0
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2.0 / 2.0)
    y = expected_y + noise
    return X, y


def train_interval_model(
    objective: str,
    alpha_name: str,
    alpha: np.ndarray,
    dtrain: xgb.QuantileDMatrix,
    dtest: xgb.QuantileDMatrix,
    args: argparse.Namespace,
) -> tuple[xgb.Booster, Dict[str, Dict]]:
    """Train a multi-output interval model."""
    evals_result: Dict[str, Dict] = {}
    booster = xgb.train(
        {
            "objective": objective,
            "tree_method": "hist",
            alpha_name: alpha,
            "learning_rate": 0.04,
            "max_depth": 5,
            "multi_strategy": args.multi_strategy,
            "device": args.device,
        },
        dtrain,
        num_boost_round=64,
        early_stopping_rounds=4,
        evals=[(dtrain, "Train"), (dtest, "Test")],
        evals_result=evals_result,
    )
    return booster, evals_result


def squared_error_model(
    dtrain: xgb.QuantileDMatrix, dtest: xgb.QuantileDMatrix, args: argparse.Namespace
) -> xgb.Booster:
    """Train a squared-error model for comparison."""
    return xgb.train(
        {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "learning_rate": 0.04,
            "max_depth": 5,
            "device": args.device,
        },
        dtrain,
        num_boost_round=64,
        early_stopping_rounds=4,
        evals=[(dtrain, "Train"), (dtest, "Test")],
    )


def prediction_intervals(args: argparse.Namespace) -> None:
    """Train quantile and expectile interval models."""
    rng = np.random.RandomState(1994)
    X, y = make_dataset(rng)
    alpha = np.array([0.05, 0.5, 0.95])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
    # We will be using the `hist` tree method, quantile DMatrix can be used to preserve
    # memory (which has nothing to do with quantile regression itself, see its document
    # for details).
    # Do not use the `exact` tree method for quantile regression, otherwise the
    # performance might drop.
    Xy = xgb.QuantileDMatrix(X_train, y_train)
    # use Xy as a reference
    Xy_test = xgb.QuantileDMatrix(X_test, y_test, ref=Xy)

    quantile_model, quantile_evals = train_interval_model(
        "reg:quantileerror", "quantile_alpha", alpha, Xy, Xy_test, args
    )
    expectile_model, expectile_evals = train_interval_model(
        "reg:expectileerror", "expectile_alpha", alpha, Xy, Xy_test, args
    )
    mean_model = squared_error_model(Xy, Xy_test, args)

    xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
    quantile_pred = quantile_model.inplace_predict(xx)
    expectile_pred = expectile_model.inplace_predict(xx)
    mean_pred = mean_model.inplace_predict(xx)

    assert quantile_pred.shape == (xx.shape[0], alpha.shape[0])
    assert expectile_pred.shape == (xx.shape[0], alpha.shape[0])

    print("Quantile test metric:", quantile_evals["Test"]["quantile"][-1])
    print("Expectile test metric:", expectile_evals["Test"]["expectile"][-1])

    if args.plot:
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)

        def plot_band(
            ax: "plt.Axes", pred: np.ndarray, title: str, center_label: str
        ) -> None:
            ax.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(x)$")
            ax.plot(
                X_test,
                y_test,
                "b.",
                markersize=5,
                alpha=0.35,
                label="Test observations",
            )
            ax.plot(xx, pred[:, 1], "r-", label=center_label)
            ax.plot(xx, mean_pred, "m--", label="Squared-error mean")
            ax.plot(xx, pred[:, 0], "k-")
            ax.plot(xx, pred[:, 2], "k-")
            ax.fill_between(
                xx.ravel(), pred[:, 0], pred[:, 2], alpha=0.3, label="90% band"
            )
            ax.set_title(title)
            ax.set_ylabel("$y$")
            ax.set_ylim(-10, 25)
            ax.legend(loc="upper left")

        plot_band(
            axes[0], quantile_pred, "Quantile regression interval", "Predicted median"
        )
        plot_band(
            axes[1],
            expectile_pred,
            "Expectile regression band",
            "Predicted 0.5 expectile",
        )
        axes[1].set_xlabel("$x$")
        fig.tight_layout()

        if args.output:
            fig.savefig(args.output, dpi=150)
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Specify it to enable plotting the outputs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path for saving the plot instead of displaying it.",
    )
    parser.add_argument(
        "--multi_strategy",
        choices=["multi_output_tree", "one_output_per_tree"],
        default="one_output_per_tree",
        help="See the parameter `multi_strategy` for more info. (Experimental)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    prediction_intervals(args)
