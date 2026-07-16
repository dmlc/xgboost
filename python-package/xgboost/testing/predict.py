"""Tests for inference."""

import json
from typing import Optional, Type

import numpy as np
import pytest
from scipy.special import logit  # pylint: disable=no-name-in-module

from ..compat import import_cupy
from ..core import DMatrix, ExtMemQuantileDMatrix, QuantileDMatrix
from ..training import train
from .data import IteratorForTest
from .shared import validate_leaf_output
from .updater import ResetStrategy, get_basescore
from .utils import Device


def _make_leaf_dmatrix(
    device: Device, DMatrixT: Type[DMatrix], X: np.ndarray, y: np.ndarray
) -> tuple[DMatrix, Optional[QuantileDMatrix]]:
    """Make the prediction matrix and an in-memory external-memory reference."""
    if DMatrixT is not ExtMemQuantileDMatrix:
        return DMatrixT(X, y), None

    X_batches = np.array_split(X, 4)
    y_batches = np.array_split(y, 4)
    cache_host_ratio = None
    if device.startswith("cuda"):
        cp = import_cupy()
        X_batches = [cp.asarray(batch) for batch in X_batches]
        y_batches = [cp.asarray(batch) for batch in y_batches]
        X = cp.asarray(X)  # type: ignore[assignment]
        y = cp.asarray(y)  # type: ignore[assignment]
        cache_host_ratio = 1.0

    it = IteratorForTest(
        X_batches,
        y_batches,
        None,
        cache="cache",
        on_host=True,
        min_cache_page_bytes=0,
    )
    m = ExtMemQuantileDMatrix(it, cache_host_ratio=cache_host_ratio)
    reference = QuantileDMatrix(X, y, ref=m)
    return m, reference


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


# pylint: disable=too-many-locals,too-many-statements
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

    m, reference = _make_leaf_dmatrix(device, DMatrixT, X, y)
    dtrain = DMatrix(X, y) if DMatrixT is ExtMemQuantileDMatrix else m
    booster = train(
        {
            "num_parallel_tree": num_parallel_tree,
            "num_class": classes,
            "tree_method": "hist",
        },
        dtrain,
        num_boost_round=num_boost_round,
    )

    booster.set_param({"device": device})
    empty = DMatrix(np.ones(shape=(0, cols)))
    empty_leaf = booster.predict(empty, pred_leaf=True)
    assert empty_leaf.shape[0] == 0

    leaf = booster.predict(m, pred_leaf=True, strict_shape=True)
    if reference is not None:
        ref_leaf = booster.predict(reference, pred_leaf=True, strict_shape=True)
        np.testing.assert_array_equal(leaf, ref_leaf)
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
    if reference is not None:
        ref_sliced = booster.predict(
            reference,
            pred_leaf=True,
            iteration_range=(0, n_iters),
            strict_shape=True,
        )
        np.testing.assert_array_equal(sliced, ref_sliced)
    first = sliced[0, ...]

    assert np.prod(first.shape) == classes * num_parallel_tree * n_iters

    # When there's only 1 tree, the output is a 1 dim vector
    booster = train({"tree_method": "hist"}, num_boost_round=1, dtrain=dtrain)
    booster.set_param({"device": device})
    assert booster.predict(m, pred_leaf=True).shape == (rows,)

    # The first two rounds have only vector-leaf trees. The full model contains
    # both vector- and scalar-leaf trees.
    mixed_rounds = 4
    mixed_parallel_trees = classes
    booster = train(
        {
            "num_parallel_tree": mixed_parallel_trees,
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
    assert all(size == classes for size in leaf_sizes[: iteration_indptr[vector_end]])
    # The total size can be reshaped into the legacy strict dimensions even
    # though the per-iteration tree counts are ragged.
    assert len(trees) % (mixed_rounds * classes) == 0

    vector_leaf = booster.predict(m, pred_leaf=True, iteration_range=(0, vector_end))
    vector_trees = trees[: iteration_indptr[vector_end]]
    assert vector_leaf.shape == (rows, len(vector_trees))
    _validate_leaf_indices(vector_leaf, vector_trees)
    with pytest.raises(ValueError, match="uniform scalar-tree layout"):
        booster.predict(
            m,
            pred_leaf=True,
            iteration_range=(0, vector_end),
            strict_shape=True,
        )

    mixed_leaf = booster.predict(m, pred_leaf=True)
    assert mixed_leaf.shape == (rows, iteration_indptr[-1])
    _validate_leaf_indices(mixed_leaf, trees)
    with pytest.raises(ValueError, match="mixed scalar/vector-leaf"):
        booster.predict(m, pred_leaf=True, strict_shape=True)

    if reference is not None:
        ref_vector_leaf = booster.predict(
            reference, pred_leaf=True, iteration_range=(0, vector_end)
        )
        ref_mixed_leaf = booster.predict(reference, pred_leaf=True)
        np.testing.assert_array_equal(vector_leaf, ref_vector_leaf)
        np.testing.assert_array_equal(mixed_leaf, ref_mixed_leaf)

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
