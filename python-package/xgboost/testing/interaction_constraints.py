"""Tests for interaction constraints."""

from typing import Optional, Sequence, Union

import numpy as np

from .._typing import FeatureNames
from ..core import DMatrix
from ..training import train
from .utils import Device


def run_interaction_constraints(  # pylint: disable=too-many-locals
    tree_method: str,
    device: Device,
    feature_names: Optional[FeatureNames] = None,
    interaction_constraints: Union[str, Sequence] = "[[0, 1]]",
) -> None:
    """Tests interaction constraints on a synthetic dataset."""
    x1 = np.random.normal(loc=1.0, scale=1.0, size=1000)
    x2 = np.random.normal(loc=1.0, scale=1.0, size=1000)
    x3 = np.random.choice([1, 2, 3], size=1000, replace=True)
    y = (
        x1
        + x2
        + x3
        + x1 * x2 * x3
        + np.random.normal(loc=0.001, scale=1.0, size=1000)
        + 3 * np.sin(x1)
    )
    X = np.column_stack((x1, x2, x3))
    dtrain = DMatrix(X, label=y, feature_names=feature_names)

    params = {
        "max_depth": 3,
        "eta": 0.1,
        "nthread": 2,
        "interaction_constraints": interaction_constraints,
        "tree_method": tree_method,
        "device": device,
    }
    num_boost_round = 12
    # Fit a model that only allows interaction between x1 and x2
    bst = train(params, dtrain, num_boost_round, evals=[(dtrain, "train")])

    # Set all observations to have the same x3 values then increment by the same amount
    def f(x: int) -> np.ndarray:
        tmat = DMatrix(
            np.column_stack((x1, x2, np.repeat(x, 1000))), feature_names=feature_names
        )
        return bst.predict(tmat)

    preds = [f(x) for x in [1, 2, 3]]

    # Check incrementing x3 has the same effect on all observations
    #   since x3 is constrained to be independent of x1 and x2
    #   and all observations start off from the same x3 value
    diff1 = preds[1] - preds[0]
    assert np.all(np.abs(diff1 - diff1[0]) < 1e-4)
    diff2 = preds[2] - preds[1]
    assert np.all(np.abs(diff2 - diff2[0]) < 1e-4)


def training_accuracy(tree_method: str, dpath: str, device: Device) -> None:
    """Test accuracy, reused by GPU tests."""
    from sklearn.metrics import accuracy_score

    dtrain = DMatrix(dpath + "agaricus.txt.train?indexing_mode=1&format=libsvm")
    dtest = DMatrix(dpath + "agaricus.txt.test?indexing_mode=1&format=libsvm")
    params = {
        "eta": 1,
        "max_depth": 6,
        "objective": "binary:logistic",
        "tree_method": tree_method,
        "device": device,
        "interaction_constraints": "[[1,2], [2,3,4]]",
    }
    num_boost_round = 5

    params["grow_policy"] = "lossguide"
    bst = train(params, dtrain, num_boost_round)
    pred_dtest = bst.predict(dtest) < 0.5
    assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1

    params["grow_policy"] = "depthwise"
    bst = train(params, dtrain, num_boost_round)
    pred_dtest = bst.predict(dtest) < 0.5
    assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1
