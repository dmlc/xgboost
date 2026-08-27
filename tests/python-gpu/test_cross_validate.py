# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import json
from functools import wraps
from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest
import xgboost as xgb
from pytest import fixture
from xgboost import _cross_validation as xcv
from xgboost import testing as tm

if TYPE_CHECKING:
    import cupy as cp


type XywExtQdm = tuple[
    list[cp.ndarray], list[cp.ndarray], list[cp.ndarray], xgb.ExtMemQuantileDMatrix
]

N_SAMPLES_PER_BATCH, N_FEATURES, N_BATCHES = 16, 4, 2


def use_cuda_async_pool[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def impl(*args: P.args, **kwargs: P.kwargs) -> R:
        with xgb.config_context(use_cuda_async_pool=True):
            return fn(*args, **kwargs)

    return impl


@fixture(scope="module")
@use_cuda_async_pool
def xyw_extqdm() -> XywExtQdm:
    X, y, w = tm.make_batches(N_SAMPLES_PER_BATCH, N_FEATURES, N_BATCHES, use_cupy=True)
    it = tm.IteratorForTest(X, y, w, cache=None, min_cache_page_bytes=0, on_host=True)
    Xy = xgb.ExtMemQuantileDMatrix(it)
    return X, y, w, Xy


def get_fold_tree(cv_folds: xcv.FoldModels, k: int, iteration: int = -1) -> dict:
    """The `iteration`-th tree of the k-th fold model, as a JSON object."""
    model = json.loads(cv_folds.save_raw("json"))
    trees = model["cv_folds"][k]["gradient_booster"]["model"]["trees"]
    return trees[iteration]


def get_leaf_weight(tree: dict, nidx: int, n_targets: int = 1) -> list[float]:
    """The leaf weight of a node of a vector-leaf tree.

    `leaf_weights` is indexed by leaf index rather than node index, and `SetLeaves`
    repurposes `right_children` as the node-to-leaf mapping.

    """
    assert is_leaf(tree, nidx)
    leaf_idx = tree["right_children"][nidx]
    weights = tree["leaf_weights"]
    return weights[leaf_idx * n_targets : (leaf_idx + 1) * n_targets]


def is_leaf(tree: dict, nidx: int) -> bool:
    """`right_children` is the leaf mapping for a vector-leaf tree, only `left_children`
    marks a leaf."""
    # Guard against -1 reaching here from a child array, which would silently wrap around.
    assert nidx >= 0
    return tree["left_children"][nidx] == -1


def fold_rows(k_folds: int, k: int) -> tuple[cp.ndarray, cp.ndarray]:
    """Global row indices of the training and the held-out rows of the k^th fold.

    The folds are split within each batch, hence the per-batch offset.

    """
    import cupy as cp
    from sklearn.model_selection import KFold

    # Every batch has the same number of rows here, hence the same within-batch split.
    train_idx, valid_idx = list(
        KFold(n_splits=k_folds).split(np.arange(N_SAMPLES_PER_BATCH))
    )[k]
    train = [cp.asarray(train_idx) + i * N_SAMPLES_PER_BATCH for i in range(N_BATCHES)]
    valid = [cp.asarray(valid_idx) + i * N_SAMPLES_PER_BATCH for i in range(N_BATCHES)]
    return cp.concatenate(train), cp.concatenate(valid)


@pytest.mark.skipif(**tm.no_cupy())
@use_cuda_async_pool
def test_cv_tree_method(xyw_extqdm: XywExtQdm) -> None:
    X, y, w, Xy = xyw_extqdm
    k_folds = 3

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    assert cv_folds.num_boosted_rounds() == 0

    predts = xcv.FoldPredictions()
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    assert cv_folds.init_prediction(Xy, folds, out=predts) is predts
    gpairs = xcv.FoldGpairs()
    assert cv_folds.get_gradient(Xy, 0, folds, predts, out=gpairs) is gpairs
    tree_method = xcv.FoldTreeMethod(cv_folds, Xy, params={"max_depth": 1})
    tree_method.update(cv_folds, Xy, folds, gpairs, predts)
    assert cv_folds.num_boosted_rounds() == 1

    # The data is continuous and random, so every fold has a positive-gain root split.
    for k in range(k_folds):
        tree = get_fold_tree(cv_folds, k)
        assert tree["left_children"] == [1, -1, -1]
        # 0 and 1 are leaf indices here, not "no child".
        assert tree["right_children"] == [2, 0, 1]
        left, right = get_leaf_weight(tree, 1), get_leaf_weight(tree, 2)
        assert left != right


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_sklearn())
@use_cuda_async_pool
def test_cv_prediction_cache(xyw_extqdm: XywExtQdm) -> None:
    import cupy as cp

    X, _, _, Xy = xyw_extqdm
    k_folds = 3
    # Fixed by FoldModels, which pins `boost_from_average` to false and the base score to
    # the objective default. It cannot be configured from here.
    base_score = 0.5
    # The device splits on the binned value, whose cut is the smallest value strictly
    # greater than the raw value, so the raw-feature equivalent of the device comparison is
    # a strict `<`. The cast matches the float32 split condition.
    features = cp.concatenate(X).astype(cp.float32)

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    cv_folds.init_prediction(Xy, folds, out=predts)
    gpairs = xcv.FoldGpairs()
    # `debug_synchronize` gates the check that every training row of a fold, and only those,
    # received a leaf position. The fused-page-pass check runs either way.
    tree_method = xcv.FoldTreeMethod(
        cv_folds, Xy, params={"max_depth": 1, "debug_synchronize": True}
    )

    expected = [
        cp.full((Xy.num_row(), 1), base_score, dtype=cp.float32) for _ in range(k_folds)
    ]

    for it in range(2):
        cv_folds.get_gradient(Xy, it, folds, predts, out=gpairs)
        tree_method.update(cv_folds, Xy, folds, gpairs, predts)
        assert cv_folds.num_boosted_rounds() == it + 1

        for k in range(k_folds):
            train_rows, valid_rows = fold_rows(k_folds, k)
            tree = get_fold_tree(cv_folds, k)
            go_left = (
                features[:, tree["split_indices"][0]] < tree["split_conditions"][0]
            )
            left = get_leaf_weight(tree, tree["left_children"][0])[0]
            right = get_leaf_weight(tree, tree["right_children"][0])[0]
            leaf_value = cp.where(go_left, left, right).astype(cp.float32)

            # Neither leaf may be empty, otherwise the check below is vacuous on one side.
            n_left = int(go_left[train_rows].sum())
            assert 0 < n_left < train_rows.size

            expected[k][train_rows] += leaf_value[train_rows].reshape(-1, 1)
            predt = predts.get(k)
            assert predt.shape == (Xy.num_row(), 1)
            cp.testing.assert_allclose(predt[train_rows], expected[k][train_rows])
            # The rows held out by the fold are padding, nothing may write to them.
            cp.testing.assert_array_equal(
                predt[valid_rows], cp.full((valid_rows.size, 1), base_score)
            )


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_sklearn())
@use_cuda_async_pool
def test_cv_oof_prediction_cache(xyw_extqdm: XywExtQdm) -> None:
    """Each row's OOF prediction comes from the fold that held it out."""
    import cupy as cp

    X, _, _, Xy = xyw_extqdm
    k_folds = 3
    features = cp.concatenate(X).astype(cp.float32)

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    cv_folds.init_prediction(Xy, folds, out=predts)
    gpairs = xcv.FoldGpairs()
    cv_folds.get_gradient(Xy, 0, folds, predts, out=gpairs)
    tree_method = xcv.FoldTreeMethod(cv_folds, Xy, params={"max_depth": 1})
    tree_method.update(cv_folds, Xy, folds, gpairs, predts)

    expected = cp.full((Xy.num_row(), 1), 0.5, dtype=cp.float32)
    for k in range(k_folds):
        _, valid_rows = fold_rows(k_folds, k)
        tree = get_fold_tree(cv_folds, k)
        go_left = features[:, tree["split_indices"][0]] < tree["split_conditions"][0]
        left = get_leaf_weight(tree, tree["left_children"][0])[0]
        right = get_leaf_weight(tree, tree["right_children"][0])[0]
        leaf_value = cp.where(go_left, left, right).astype(cp.float32)
        expected[valid_rows] += leaf_value[valid_rows].reshape(-1, 1)

    cp.testing.assert_allclose(predts.get_valid(), expected)


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_sklearn())
@use_cuda_async_pool
def test_cv_vs_reference(xyw_extqdm: XywExtQdm) -> None:
    """Each fold must train exactly like a booster fitted on that fold's rows alone."""
    import cupy as cp

    X, y, w, Xy = xyw_extqdm
    k_folds, n_rounds = 3, 3
    params = {"max_depth": 1, "debug_synchronize": True}

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    cv_folds.init_prediction(Xy, folds, out=predts)
    gpairs = xcv.FoldGpairs()
    tree_method = xcv.FoldTreeMethod(cv_folds, Xy, params=params)
    for it in range(n_rounds):
        cv_folds.get_gradient(Xy, it, folds, predts, out=gpairs)
        tree_method.update(cv_folds, Xy, folds, gpairs, predts)

    features, labels, weights = (cp.concatenate(v) for v in (X, y, w))
    for k in range(k_folds):
        train_rows, _ = fold_rows(k_folds, k)
        # `ref` shares the CV cuts, so the reference sees the same bins.
        Xyk = xgb.QuantileDMatrix(
            features[train_rows],
            label=labels[train_rows],
            weight=weights[train_rows],
            ref=Xy,
        )
        booster = xgb.train(
            {
                **params,
                "base_score": 0.5,
                "device": "cuda",
                "multi_strategy": "multi_output_tree",
            },
            Xyk,
            num_boost_round=n_rounds,
        )
        margin = cp.asarray(booster.predict(Xyk, output_margin=True)).reshape(-1, 1)
        cp.testing.assert_allclose(
            predts.get(k)[train_rows], margin, rtol=1e-6, atol=1e-6
        )


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_sklearn())
@use_cuda_async_pool
def test_cv_fold_info_batches(xyw_extqdm: XywExtQdm) -> None:
    import cupy as cp
    from sklearn.model_selection import KFold

    X, y, w, Xy = xyw_extqdm
    k_folds = 3

    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)

    assert isinstance(folds.handle, ctypes.c_void_p)
    assert folds.handle.value is not None
    assert folds.k_folds == k_folds

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    assert cv_folds.init_prediction(Xy, folds, out=predts) is predts
    gpairs = xcv.FoldGpairs()
    assert cv_folds.get_gradient(Xy, 0, folds, predts, out=gpairs) is gpairs

    assert isinstance(gpairs.handle, ctypes.c_void_p)
    assert gpairs.handle.value is not None
    for k in range(k_folds):
        grad, hess = gpairs.get(k, copy=False)
        # The gradient is indexed by the global row index, the validation rows of the fold
        # are zeroed out.
        assert grad.shape == (Xy.num_row(), 1)
        assert grad.shape == hess.shape
        assert grad.dtype == hess.dtype
        assert grad.data.ptr + ctypes.sizeof(ctypes.c_float) == hess.data.ptr
        assert grad.strides == hess.strides
        assert grad.strides == (
            2 * ctypes.sizeof(ctypes.c_float),
            2 * ctypes.sizeof(ctypes.c_float),
        )

        expected_labels = []
        expected_weights = []
        for batch_y, batch_w in zip(y, w):
            train_idx, _ = list(KFold(n_splits=k_folds).split(batch_y))[k]
            idx = cp.asarray(train_idx)
            masked_w = cp.zeros_like(batch_w)
            masked_w[idx] = batch_w[idx]
            expected_labels.append(batch_y)
            expected_weights.append(masked_w)

        expected_labels = (
            cp.concatenate(expected_labels).astype(cp.float32).reshape(grad.shape)
        )
        expected_weights = (
            cp.concatenate(expected_weights).astype(cp.float32).reshape(hess.shape)
        )
        cp.testing.assert_allclose(grad, (0.5 - expected_labels) * expected_weights)
        cp.testing.assert_allclose(hess, expected_weights)

    assert cv_folds.get_gradient(Xy, 1, folds, predts, out=gpairs) is gpairs


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_sklearn())
@use_cuda_async_pool
def test_cv_base_margin() -> None:
    import cupy as cp
    from sklearn.model_selection import KFold

    k_folds = 3
    X, y, w = tm.make_batches(16, 4, 2, use_cupy=True)
    it = tm.IteratorForTest(X, y, w, cache=None, min_cache_page_bytes=0, on_host=True)
    Xy = xgb.ExtMemQuantileDMatrix(it)
    # A distinct margin for every row, the gradient of a row must be calculated from the
    # margin of that same row.
    margin = cp.arange(Xy.num_row(), dtype=cp.float32) / Xy.num_row()
    Xy.set_info(base_margin=margin)

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    cv_folds.init_prediction(Xy, folds, out=predts)
    gpairs = xcv.FoldGpairs()
    cv_folds.get_gradient(Xy, 0, folds, predts, out=gpairs)

    for k in range(k_folds):
        grad, hess = gpairs.get(k, copy=False)

        expected_weights = []
        for batch_w in w:
            train_idx, _ = list(KFold(n_splits=k_folds).split(batch_w))[k]
            idx = cp.asarray(train_idx)
            masked_w = cp.zeros_like(batch_w)
            masked_w[idx] = batch_w[idx]
            expected_weights.append(masked_w)

        labels = cp.concatenate(y).astype(cp.float32).reshape(grad.shape)
        weights = (
            cp.concatenate(expected_weights).astype(cp.float32).reshape(hess.shape)
        )
        cp.testing.assert_allclose(
            grad, (margin.reshape(grad.shape) - labels) * weights
        )
        cp.testing.assert_allclose(hess, weights)
