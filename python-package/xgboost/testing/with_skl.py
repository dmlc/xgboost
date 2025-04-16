# pylint: disable=invalid-name, too-many-arguments, too-many-positional-arguments
"""Tests for compatiblity with sklearn."""

from typing import Callable, Optional, Type

import numpy as np
import pytest

from ..core import DMatrix
from ..sklearn import XGBClassifier, XGBRFRegressor
from .utils import Device


def run_boost_from_prediction_binary(
    tree_method: str,
    device: Device,
    X: np.ndarray,
    y: np.ndarray,
    as_frame: Optional[Callable],
) -> None:
    """
    Parameters
    ----------

    as_frame: A callable function to convert margin into DataFrame, useful for different
    df implementations.
    """

    model_0 = XGBClassifier(
        learning_rate=0.3,
        random_state=0,
        n_estimators=4,
        tree_method=tree_method,
        device=device,
    )
    model_0.fit(X=X, y=y)
    margin = model_0.predict(X, output_margin=True)
    if as_frame is not None:
        margin = as_frame(margin)

    model_1 = XGBClassifier(
        learning_rate=0.3,
        random_state=0,
        n_estimators=4,
        tree_method=tree_method,
        device=device,
    )
    model_1.fit(X=X, y=y, base_margin=margin)
    predictions_1 = model_1.predict(X, base_margin=margin)

    cls_2 = XGBClassifier(
        learning_rate=0.3,
        random_state=0,
        n_estimators=8,
        tree_method=tree_method,
        device=device,
    )
    cls_2.fit(X=X, y=y)
    predictions_2 = cls_2.predict(X)
    np.testing.assert_allclose(predictions_1, predictions_2)


def run_boost_from_prediction_multi_clasas(
    estimator: Type,
    tree_method: str,
    device: Device,
    X: np.ndarray,
    y: np.ndarray,
    as_frame: Optional[Callable],
) -> None:
    """Boosting from prediction with multi-class clf."""
    # Multi-class
    model_0 = estimator(
        learning_rate=0.3,
        random_state=0,
        n_estimators=4,
        tree_method=tree_method,
        device=device,
    )
    model_0.fit(X=X, y=y)
    margin = model_0.get_booster().inplace_predict(X, predict_type="margin")
    if as_frame is not None:
        margin = as_frame(margin)

    model_1 = estimator(
        learning_rate=0.3,
        random_state=0,
        n_estimators=4,
        tree_method=tree_method,
        device=device,
    )
    model_1.fit(X=X, y=y, base_margin=margin)
    predictions_1 = model_1.get_booster().predict(
        DMatrix(X, base_margin=margin), output_margin=True
    )

    model_2 = estimator(
        learning_rate=0.3,
        random_state=0,
        n_estimators=8,
        tree_method=tree_method,
        device=device,
    )
    model_2.fit(X=X, y=y)
    predictions_2 = model_2.get_booster().inplace_predict(X, predict_type="margin")

    if hasattr(predictions_1, "get"):
        predictions_1 = predictions_1.get()
    if hasattr(predictions_2, "get"):
        predictions_2 = predictions_2.get()
    np.testing.assert_allclose(predictions_1, predictions_2, atol=1e-6)


def run_housing_rf_regression(tree_method: str, device: Device) -> None:
    """Testwith the cali housing dataset."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    X, y = fetch_california_housing(return_X_y=True)
    rng = np.random.RandomState(1994)
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = XGBRFRegressor(
            random_state=42, tree_method=tree_method, device=device
        ).fit(X[train_index], y[train_index])
        preds = xgb_model.predict(X[test_index])
        labels = y[test_index]
        assert mean_squared_error(preds, labels) < 35

    rfreg = XGBRFRegressor(device=device)
    with pytest.raises(NotImplementedError):
        rfreg.set_params(early_stopping_rounds=10)
        rfreg.fit(X, y)
