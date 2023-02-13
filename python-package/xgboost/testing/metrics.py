"""Tests for evaluation metrics."""
from typing import Dict

import numpy as np

import xgboost as xgb


def check_quantile_error(tree_method: str) -> None:
    """Test for the `quantile` loss."""
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_pinball_loss

    rng = np.random.RandomState(19)
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(128, 3, random_state=rng)
    Xy = xgb.QuantileDMatrix(X, y)
    evals_result: Dict[str, Dict] = {}
    booster = xgb.train(
        {"tree_method": tree_method, "eval_metric": "quantile", "quantile_alpha": 0.3},
        Xy,
        evals=[(Xy, "Train")],
        evals_result=evals_result,
    )
    predt = booster.inplace_predict(X)
    loss = mean_pinball_loss(y, predt, alpha=0.3)
    np.testing.assert_allclose(evals_result["Train"]["quantile"][-1], loss)
