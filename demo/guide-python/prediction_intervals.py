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
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

Data = Tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix]
Predictions = Dict[str, np.ndarray]


def f(x: np.ndarray) -> np.ndarray:
    """The function to predict."""
    return x * np.sin(x)


def make_dataset(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Generate heteroscedastic data with asymmetric noise."""
    features = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    expected_y = f(features).ravel()

    sigma = 0.5 + features.ravel() / 10.0
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2.0 / 2.0)
    target = expected_y + noise
    return features, target


def train_interval_model(
    params: Dict[str, object],
    alpha_name: str,
    alpha: np.ndarray,
    data: Data,
) -> Tuple[xgb.Booster, Dict[str, Dict]]:
    """Train a multi-output interval model."""
    dtrain, dtest = data
    evals_result: Dict[str, Dict] = {}
    params = params.copy()
    params[alpha_name] = alpha
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=64,
        early_stopping_rounds=4,
        evals=[(dtrain, "Train"), (dtest, "Test")],
        evals_result=evals_result,
    )
    return booster, evals_result


def squared_error_model(params: Dict[str, object], data: Data) -> xgb.Booster:
    """Train a squared-error model for comparison."""
    dtrain, dtest = data
    params = params.copy()
    params["objective"] = "reg:squarederror"
    return xgb.train(
        params,
        dtrain,
        num_boost_round=64,
        early_stopping_rounds=4,
        evals=[(dtrain, "Train"), (dtest, "Test")],
    )


def base_params(cli_args: argparse.Namespace) -> Dict[str, object]:
    """Parameters shared by the three models in the demo."""
    return {
        "tree_method": "hist",
        "learning_rate": 0.04,
        "max_depth": 5,
        "multi_strategy": cli_args.multi_strategy,
        "device": cli_args.device,
    }


def train_models(
    data: Data, alpha: np.ndarray, params: Dict[str, object]
) -> Tuple[Dict[str, xgb.Booster], Dict[str, Dict[str, Dict]]]:
    """Train quantile, expectile, and squared-error models."""
    quantile_model, quantile_evals = train_interval_model(
        {**params, "objective": "reg:quantileerror"}, "quantile_alpha", alpha, data
    )
    expectile_model, expectile_evals = train_interval_model(
        {**params, "objective": "reg:expectileerror"}, "expectile_alpha", alpha, data
    )
    models = {
        "quantile": quantile_model,
        "expectile": expectile_model,
        "mean": squared_error_model(params, data),
    }
    return models, {"quantile": quantile_evals, "expectile": expectile_evals}


def make_predictions(models: Dict[str, xgb.Booster]) -> Tuple[np.ndarray, Predictions]:
    """Predict on a dense grid for plotting."""
    grid = np.atleast_2d(np.linspace(0, 10, 1000)).T
    predictions = {
        "quantile": models["quantile"].inplace_predict(grid),
        "expectile": models["expectile"].inplace_predict(grid),
        "mean": models["mean"].inplace_predict(grid),
    }
    return grid, predictions


def make_matrices(
    rng: np.random.RandomState,
) -> Tuple[Data, Tuple[np.ndarray, np.ndarray]]:
    """Create train/test DMatrices and return held-out data for plotting."""
    features, target = make_dataset(rng)
    split = train_test_split(features, target, random_state=rng)
    train_features, test_features, train_target, test_target = split
    train = xgb.QuantileDMatrix(train_features, train_target)
    test = xgb.QuantileDMatrix(test_features, test_target, ref=train)
    return (train, test), (test_features, test_target)


def plot_prediction_intervals(
    grid: np.ndarray,
    test_data: Tuple[np.ndarray, np.ndarray],
    predictions: Predictions,
    output: str,
) -> None:
    """Plot quantile interval and expectile band."""
    from matplotlib import pyplot as plt

    features, target = test_data
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True, sharey=True)

    def plot_band(
        ax: "plt.Axes", pred: np.ndarray, title: str, center_label: str
    ) -> None:
        ax.plot(grid, f(grid), "g:", linewidth=3, label=r"$f(x) = x\,\sin(x)$")
        ax.plot(
            features,
            target,
            "b.",
            markersize=5,
            alpha=0.35,
            label="Test observations",
        )
        ax.plot(grid, pred[:, 1], "r-", label=center_label)
        ax.plot(grid, predictions["mean"], "m--", label="Squared-error mean")
        ax.plot(grid, pred[:, 0], "k-")
        ax.plot(grid, pred[:, 2], "k-")
        ax.fill_between(
            grid.ravel(), pred[:, 0], pred[:, 2], alpha=0.3, label="90% band"
        )
        ax.set_title(title)
        ax.set_ylabel("$y$")
        ax.set_ylim(-10, 25)
        ax.legend(loc="upper left")

    plot_band(
        axes[0],
        predictions["quantile"],
        "Quantile regression interval",
        "Predicted median",
    )
    plot_band(
        axes[1],
        predictions["expectile"],
        "Expectile regression band",
        "Predicted 0.5 expectile",
    )
    axes[1].set_xlabel("$x$")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()


def prediction_intervals(cli_args: argparse.Namespace) -> None:
    """Train quantile and expectile interval models."""
    rng = np.random.RandomState(1994)
    alpha = np.array([0.05, 0.5, 0.95])

    data, test_data = make_matrices(rng)
    models, evals = train_models(data, alpha, base_params(cli_args))
    grid, predictions = make_predictions(models)

    assert predictions["quantile"].shape == (grid.shape[0], alpha.shape[0])
    assert predictions["expectile"].shape == (grid.shape[0], alpha.shape[0])

    print("Quantile test metric:", evals["quantile"]["Test"]["quantile"][-1])
    print("Expectile test metric:", evals["expectile"]["Test"]["expectile"][-1])

    if cli_args.plot:
        plot_prediction_intervals(grid, test_data, predictions, cli_args.output)


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
    prediction_intervals(parser.parse_args())
