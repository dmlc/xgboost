# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0
import ctypes

import pytest
import xgboost as xgb
from xgboost import _cross_validation as xcv
from xgboost import testing as tm


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_sklearn())
def test_cv_fold_info_batches() -> None:
    import cupy as cp
    from sklearn.model_selection import KFold

    k_folds = 3
    X, y, w = tm.make_batches(16, 4, 2, use_cupy=True)
    it = tm.IteratorForTest(X, y, w, cache=None, min_cache_page_bytes=0, on_host=True)
    Xy = xgb.ExtMemQuantileDMatrix(it)

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
        grad, hess = gpairs.get(k)
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
            expected_labels.append(batch_y[idx])
            expected_weights.append(batch_w[idx])

        expected_labels = (
            cp.concatenate(expected_labels).astype(cp.float32).reshape(grad.shape)
        )
        expected_weights = (
            cp.concatenate(expected_weights).astype(cp.float32).reshape(hess.shape)
        )
        cp.testing.assert_allclose(grad, (0.5 - expected_labels) * expected_weights)
        cp.testing.assert_allclose(hess, expected_weights)

    assert cv_folds.get_gradient(Xy, 1, folds, predts, out=gpairs) is gpairs
