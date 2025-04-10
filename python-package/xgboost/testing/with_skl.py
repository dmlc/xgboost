"""Tests for compatiblity with sklearn."""

from typing import Callable, Optional

import numpy as np

from ..sklearn import XGBClassifier
from .utils import Device


def run_boost_from_prediction_binary(
    tree_method: str,
    device: Device,
    X: np.ndarray,  # pylint: disable=invalid-name
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
