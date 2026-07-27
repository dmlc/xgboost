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
    n_targets: int = 1,
) -> None:
    """Tests interaction constraints on a synthetic dataset. Only x1 and x2 are allowed
    to interact; x3 must stay additive.

    """
    rng = np.random.default_rng(2026)
    n_samples = 1000
    x1 = rng.normal(loc=1.0, scale=1.0, size=n_samples)
    x2 = rng.normal(loc=1.0, scale=1.0, size=n_samples)
    x3 = rng.choice([1, 2, 3], size=n_samples, replace=True)
    X = np.column_stack((x1, x2, x3))

    # The constraint must force x3's contribution to remain additive regardless.
    shared = (
        x1
        + x2
        + x1 * x2 * x3
        + 3.0 * np.sin(x1)
        + rng.normal(loc=0.0, scale=1.0, size=n_samples)
    )
    # Each target gets a distinct, non-zero additive coef on x3.
    slopes = np.arange(1, n_targets + 1, dtype=np.float64)
    slopes[1::2] *= -1.0
    targets = [shared + slope * x3 for slope in slopes]
    y = targets[0] if n_targets == 1 else np.column_stack(targets)

    dtrain = DMatrix(X, label=y, feature_names=feature_names)
    params = {
        "max_depth": 3,
        "eta": 0.1,
        "nthread": 2,
        "interaction_constraints": interaction_constraints,
        "tree_method": tree_method,
        "device": device,
    }
    if n_targets > 1:
        params["multi_strategy"] = "multi_output_tree"

    num_boost_round = 12
    # Fit a model that only allows interaction between x1 and x2.
    bst = train(params, dtrain, num_boost_round, evals=[(dtrain, "train")])

    # Set all observations to have the same x3 value then increment by the same amount.
    def f(x: int) -> np.ndarray:
        tmat = DMatrix(
            np.column_stack((x1, x2, np.repeat(x, n_samples))),
            feature_names=feature_names,
        )
        return bst.predict(tmat)

    preds = [f(x) for x in [1, 2, 3]]
    expected_shape = (n_samples,) if n_targets == 1 else (n_samples, n_targets)
    assert preds[0].shape == expected_shape

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
