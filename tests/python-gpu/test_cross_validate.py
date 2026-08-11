# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
from functools import wraps
from typing import TYPE_CHECKING, Callable

import pytest
import xgboost as xgb
from pytest import fixture
from xgboost import _cross_validation as xcv
from xgboost import testing as tm

if TYPE_CHECKING:
    import cupy as cp


type XywExtQdm = tuple[cp.ndarray, cp.ndarray, cp.ndarray, xgb.ExtMemQuantileDMatrix]

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
    tree_method.update(cv_folds, Xy, folds, gpairs)
    assert cv_folds.num_boosted_rounds() == 1


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
