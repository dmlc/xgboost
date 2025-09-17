"""Tests for inference."""

from typing import Type

import numpy as np
from scipy.special import logit  # pylint: disable=no-name-in-module

from ..core import DMatrix
from ..training import train
from .shared import validate_leaf_output
from .updater import get_basescore
from .utils import Device


# pylint: disable=invalid-name,too-many-locals
def run_predict_leaf(device: Device, DMatrixT: Type[DMatrix]) -> np.ndarray:
    """Run tests for leaf index prediction."""
    rows = 100
    cols = 4
    classes = 5
    num_parallel_tree = 4
    num_boost_round = 10
    rng = np.random.RandomState(1994)
    X = rng.randn(rows, cols)
    y = rng.randint(low=0, high=classes, size=rows)

    m = DMatrixT(X, y)
    booster = train(
        {
            "num_parallel_tree": num_parallel_tree,
            "num_class": classes,
            "tree_method": "hist",
        },
        m,
        num_boost_round=num_boost_round,
    )

    booster.set_param({"device": device})
    empty = DMatrixT(np.ones(shape=(0, cols)))
    empty_leaf = booster.predict(empty, pred_leaf=True)
    assert empty_leaf.shape[0] == 0

    leaf = booster.predict(m, pred_leaf=True, strict_shape=True)
    assert leaf.shape[0] == rows
    assert leaf.shape[1] == num_boost_round
    assert leaf.shape[2] == classes
    assert leaf.shape[3] == num_parallel_tree

    validate_leaf_output(leaf, num_parallel_tree)

    n_iters = np.int32(2)
    sliced = booster.predict(
        m,
        pred_leaf=True,
        iteration_range=(0, n_iters),
        strict_shape=True,
    )
    first = sliced[0, ...]

    assert np.prod(first.shape) == classes * num_parallel_tree * n_iters

    # When there's only 1 tree, the output is a 1 dim vector
    booster = train({"tree_method": "hist"}, num_boost_round=1, dtrain=m)
    booster.set_param({"device": device})
    assert booster.predict(m, pred_leaf=True).shape == (rows,)

    return leaf


def run_base_margin_vs_base_score(device: Device) -> None:
    """Test for the relation between score and margin."""
    from sklearn.datasets import make_classification

    intercept = 0.5

    X, y = make_classification(random_state=2025)
    booster = train(
        {"base_score": intercept, "objective": "binary:logistic", "device": device},
        dtrain=DMatrix(X, y),
        num_boost_round=1,
    )
    np.testing.assert_allclose(get_basescore(booster), intercept)
    predt_0 = booster.predict(DMatrix(X, y))

    margin = np.full(y.shape, fill_value=logit(intercept), dtype=np.float32)
    Xy = DMatrix(X, y, base_margin=margin)
    # 0.2 is a dummy value
    booster = train(
        {"base_score": 0.2, "objective": "binary:logistic", "device": device},
        dtrain=Xy,
        num_boost_round=1,
    )
    np.testing.assert_allclose(get_basescore(booster), 0.2)
    predt_1 = booster.predict(Xy)

    np.testing.assert_allclose(predt_0, predt_1)
