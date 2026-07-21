"""Helpers for testing monotone constraints."""

import json
from typing import Optional, Sequence

import numpy as np
import pytest

from .._typing import FeatureNames
from ..core import Booster, DMatrix
from ..training import train
from .utils import Device


def is_increasing(v: np.ndarray) -> bool:
    """Whether ``v`` is nondecreasing along the sweep axis.

    ``v`` can be a 1-D vector (scalar leaf) or a 2-D ``(grid, targets)`` matrix (vector
    leaf).  For a matrix every output column must be nondecreasing.
    """
    return np.count_nonzero(np.diff(v, axis=0) < 0.0) == 0


def is_decreasing(v: np.ndarray) -> bool:
    """Whether ``v`` is nonincreasing along the sweep axis.

    See :py:func:`is_increasing` for the handling of vector-leaf prediction matrices.
    """
    return np.count_nonzero(np.diff(v, axis=0) > 0.0) == 0


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

# Multi-output labels sharing the same ``(f0: increasing, f1: decreasing)`` structure as
# ``y``. Each column is a positive-scale affine transform of ``y``, so every target must
# be increasing in ``f0`` and decreasing in ``f1`` while still differing enough to
# exercise shared vector-leaf splits.
_mt_noise = np.random.default_rng(2026).normal(
    loc=0.0, scale=0.01, size=(NUMBER_OF_DPOINTS, 2)
)
y_mt = np.column_stack(
    [y, 2.0 * y + _mt_noise[:, 0], 0.5 * y + _mt_noise[:, 1]]
).astype(np.float32)


def run_parent_gain(device: Device, multi_strategy: str) -> None:
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


def run_monotone_constraints(
    device: Device,
    tree_method: str,
    grow_policy: str,
    multi_strategy: str = "one_output_per_tree",
) -> None:
    """Check for a positive (``f0``) and negative (``f1``) constraint."""
    label = y_mt if multi_strategy == "multi_output_tree" else y
    dtrain = DMatrix(x, label=label)
    params = {
        "tree_method": tree_method,
        "grow_policy": grow_policy,
        "device": device,
        "multi_strategy": multi_strategy,
        "monotone_constraints": "(1, -1)",
    }
    model = train(params, dtrain, num_boost_round=16)
    assert is_correctly_constrained(model)


def _assert_monotone(
    booster: Booster,
    constraints: Sequence[int],
    *,
    n_grid: int = 64,
    n_ref: int = 32,
    seed: int = 0,
) -> None:
    """Grid-check monotonicity per output column for every constrained feature.

    For each feature with a nonzero constraint, sweep it over ``[0, 1]`` while holding
    the remaining features fixed at random reference rows.

    """
    rng = np.random.RandomState(seed)
    n_features = len(constraints)
    grid = np.linspace(0.0, 1.0, n_grid, dtype=np.float32)
    for fidx, direction in enumerate(constraints):
        if direction == 0:
            continue
        for _ in range(n_ref):
            base = rng.rand(n_features).astype(np.float32)
            features = np.tile(base, (n_grid, 1))
            features[:, fidx] = grid
            pred = booster.predict(DMatrix(features))
            if direction > 0:
                assert is_increasing(pred), (fidx, "increasing")
            else:
                assert is_decreasing(pred), (fidx, "decreasing")


def run_multi_output_monotone(
    device: Device, grow_policy: str, multi_strategy: str
) -> None:
    """Monotonicity check for deep trees with mixed feature constraints.

    Uses more features than constraints so that constrained splits are followed by
    splits on unconstrained features.

    """
    constraints = (1, -1, 0, 0, 0)
    rng = np.random.RandomState(2026)
    n_samples, n_features, n_targets = 2048, len(constraints), 3
    features = rng.rand(n_samples, n_features).astype(np.float32)
    labels = np.empty((n_samples, n_targets), dtype=np.float32)
    for t in range(n_targets):
        labels[:, t] = (
            (t + 1) * features[:, 0]
            - (t + 1) * features[:, 1]
            + np.sin(6.0 * features[:, 2])
            + 0.5 * features[:, 3] * features[:, 4]
            + rng.normal(scale=0.05, size=n_samples)
        )
    params = {
        "tree_method": "hist",
        "device": device,
        "grow_policy": grow_policy,
        "multi_strategy": multi_strategy,
        "monotone_constraints": constraints,
        "max_depth": 6,
        "eta": 0.3,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "min_child_weight": 0.0,
    }
    booster = train(params, DMatrix(features, label=labels), num_boost_round=40)
    _assert_monotone(booster, constraints, seed=2026 + 1)
