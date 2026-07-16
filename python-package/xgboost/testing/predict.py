"""Tests for inference."""

import json
from typing import Any, Optional, Type

import numpy as np
import pytest
from scipy.special import logit  # pylint: disable=no-name-in-module

from ..compat import import_cupy
from ..core import Booster, DMatrix, ExtMemQuantileDMatrix, QuantileDMatrix
from ..training import train
from .data import IteratorForTest
from .shared import validate_leaf_output
from .updater import ResetStrategy, get_basescore
from .utils import Device


def _make_leaf_dmatrix(
    device: Device, DMatrixT: Type[DMatrix], X: np.ndarray, y: np.ndarray
) -> tuple[DMatrix, Optional[QuantileDMatrix]]:
    """Make a prediction matrix and an in-memory reference for external memory."""
    if DMatrixT is not ExtMemQuantileDMatrix:
        return DMatrixT(X, y), None

    xp = import_cupy() if device.startswith("cuda") else np
    X, y = xp.asarray(X), xp.asarray(y)
    it = IteratorForTest(
        xp.array_split(X, 4),
        xp.array_split(y, 4),
        None,
        cache="cache",
        on_host=True,
        min_cache_page_bytes=0,
    )
    m = ExtMemQuantileDMatrix(
        it, cache_host_ratio=1.0 if device.startswith("cuda") else None
    )
    return m, QuantileDMatrix(X, y, ref=m)


def _predict_leaf(
    booster: Booster,
    m: DMatrix,
    reference: Optional[DMatrix],
    **kwargs: Any,
) -> np.ndarray:
    """Predict leaves and compare external memory against in-memory data."""
    predt = booster.predict(m, pred_leaf=True, **kwargs)
    if reference is not None:
        ref_predt = booster.predict(reference, pred_leaf=True, **kwargs)
        np.testing.assert_array_equal(predt, ref_predt)
    return predt


def _validate_leaf_indices(leaf: np.ndarray, trees: list[dict]) -> None:
    """Validate that each predicted node is a leaf in the corresponding tree."""
    assert leaf.ndim == 2
    assert leaf.shape[1] == len(trees)
    for tree_idx, tree in enumerate(trees):
        prediction = leaf[:, tree_idx]
        assert np.isfinite(prediction).all()
        node_idx = prediction.astype(np.int64)
        np.testing.assert_array_equal(prediction, node_idx)

        left = np.asarray(tree["left_children"])
        assert ((0 <= node_idx) & (node_idx < left.size)).all()
        assert (left[node_idx] == -1).all()


# pylint: disable=too-many-locals
def run_predict_leaf(device: Device, DMatrixT: Type[DMatrix]) -> np.ndarray:
    """Run tests for leaf index prediction."""
    rows, cols, classes = 100, 4, 5
    n_parallel, n_rounds = 4, 10
    rng = np.random.RandomState(1994)
    X = rng.randn(rows, cols)
    y = rng.randint(low=0, high=classes, size=rows)

    m, reference = _make_leaf_dmatrix(device, DMatrixT, X, y)
    dtrain = DMatrix(X, y) if DMatrixT is ExtMemQuantileDMatrix else m
    booster = train(
        {
            "num_parallel_tree": n_parallel,
            "num_class": classes,
            "tree_method": "hist",
        },
        dtrain,
        num_boost_round=n_rounds,
    )
    booster.set_param({"device": device})
    assert booster.predict(DMatrix(np.empty((0, cols))), pred_leaf=True).shape[0] == 0

    leaf = _predict_leaf(booster, m, reference, strict_shape=True)
    assert leaf.shape == (rows, n_rounds, classes, n_parallel)
    validate_leaf_output(leaf, n_parallel)

    n_iters = 2
    sliced = _predict_leaf(
        booster,
        m,
        reference,
        iteration_range=(0, n_iters),
        strict_shape=True,
    )
    assert sliced.shape == (rows, n_iters, classes, n_parallel)

    # When there's only 1 tree, the output is a 1 dim vector
    booster = train({"tree_method": "hist"}, num_boost_round=1, dtrain=dtrain)
    booster.set_param({"device": device})
    assert _predict_leaf(booster, m, reference).shape == (rows,)

    # The first two rounds have only vector-leaf trees. The full model contains
    # both vector- and scalar-leaf trees.
    mixed_rounds = 4
    booster = train(
        {
            "num_parallel_tree": classes,
            "num_class": classes,
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree",
        },
        dtrain,
        num_boost_round=mixed_rounds,
        callbacks=[ResetStrategy()],
    )
    booster.set_param({"device": device})

    model = json.loads(booster.save_raw(raw_format="json"))
    gbtree_model = model["learner"]["gradient_booster"]["model"]
    trees = gbtree_model["trees"]
    iteration_indptr = gbtree_model["iteration_indptr"]
    leaf_sizes = [int(tree["tree_param"]["size_leaf_vector"]) for tree in trees]
    vector_end = 2
    assert set(leaf_sizes) == {1, classes}
    assert set(leaf_sizes[: iteration_indptr[vector_end]]) == {classes}

    for end, n_trees in [(vector_end, iteration_indptr[vector_end]), (0, len(trees))]:
        kwargs = {"iteration_range": (0, end)}
        predt = _predict_leaf(booster, m, reference, **kwargs)
        selected_trees = trees[:n_trees]
        assert predt.shape == (rows, len(selected_trees))
        _validate_leaf_indices(predt, selected_trees)
        with pytest.raises(ValueError, match="vector leaf trees"):
            booster.predict(m, pred_leaf=True, strict_shape=True, **kwargs)

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
