"""Helpers for testing monotone constraints."""

import json
from typing import Optional

import numpy as np
import pytest

from .._typing import FeatureNames
from ..core import Booster, DMatrix
from ..training import train
from .utils import Device


def is_increasing(v: np.ndarray) -> bool:
    """Whether is v increasing."""
    return np.count_nonzero(np.diff(v) < 0.0) == 0


def is_decreasing(v: np.ndarray) -> bool:
    """Whether is v decreasing."""
    return np.count_nonzero(np.diff(v) > 0.0) == 0


def is_correctly_constrained(
    learner: Booster, feature_names: Optional[FeatureNames] = None
) -> bool:
    """Whether the monotone constraint is correctly applied."""
    n = 100
    variable_x = np.linspace(0, 1, n).reshape((n, 1))
    fixed_xs_values = np.linspace(0, 1, n)

    for i in range(n):
        fixed_x = fixed_xs_values[i] * np.ones((n, 1))
        monotonically_increasing_x = np.column_stack((variable_x, fixed_x))
        monotonically_increasing_dset = DMatrix(
            monotonically_increasing_x, feature_names=feature_names
        )
        monotonically_increasing_y = learner.predict(monotonically_increasing_dset)

        monotonically_decreasing_x = np.column_stack((fixed_x, variable_x))
        monotonically_decreasing_dset = DMatrix(
            monotonically_decreasing_x, feature_names=feature_names
        )
        monotonically_decreasing_y = learner.predict(monotonically_decreasing_dset)

        if not (
            is_increasing(monotonically_increasing_y)
            and is_decreasing(monotonically_decreasing_y)
        ):
            return False

    return True


NUMBER_OF_DPOINTS = 1000
x1_positively_correlated_with_y = np.random.random(size=NUMBER_OF_DPOINTS)
x2_negatively_correlated_with_y = np.random.random(size=NUMBER_OF_DPOINTS)

x = np.column_stack((x1_positively_correlated_with_y, x2_negatively_correlated_with_y))
zs = np.random.normal(loc=0.0, scale=0.01, size=NUMBER_OF_DPOINTS)
y = (
    5 * x1_positively_correlated_with_y
    + np.sin(10 * np.pi * x1_positively_correlated_with_y)
    - 5 * x2_negatively_correlated_with_y
    - np.cos(10 * np.pi * x2_negatively_correlated_with_y)
    + zs
)
training_dset = DMatrix(x, label=y)


def run_parent_gain(
    device: Device, multi_strategy: str = "one_output_per_tree"
) -> None:
    """Test that parent gain uses the node's inherited monotonic bounds."""
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    grad = np.array([3.0, -2.75, -0.125, 0.875], dtype=np.float32)
    hess = np.array([1.0, 0.5, 0.25, 0.25], dtype=np.float32)
    expected_gain = 0.25
    if multi_strategy == "multi_output_tree":
        gradients = np.stack([grad, 2.0 * grad], axis=1)
        hessians = np.stack([hess, hess], axis=1)
        expected_gain += 1.0
    else:
        gradients, hessians = grad, hess
    dtrain = DMatrix(X, label=np.zeros_like(gradients))

    def objective(
        _predt: np.ndarray, _dtrain: DMatrix
    ) -> tuple[np.ndarray, np.ndarray]:
        return gradients, hessians

    params = {
        "tree_method": "hist",
        "monotone_constraints": "(1, 0, 0)",
        "reg_alpha": 0,
        "reg_lambda": 0,
        "max_delta_step": 0,
        "min_child_weight": 0,
        "max_depth": 3,
        "eta": 1,
        "base_score": 0,
        "device": device,
        "multi_strategy": multi_strategy,
    }
    model = train(params, dtrain, num_boost_round=1, obj=objective)
    trees = json.loads(model.get_dump(dump_format="json", with_stats=True)[0])

    constrained_parent = trees["children"][1]["children"][1]
    assert constrained_parent["split"] == "f2"
    assert constrained_parent["gain"] == pytest.approx(expected_gain)
