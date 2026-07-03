# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0
import ctypes

import pytest
import xgboost as xgb
from xgboost import _cross_validation as xcv
from xgboost import testing as tm


@pytest.mark.skipif(**tm.no_cupy())
def test_cv_fold_info_batches() -> None:
    it = tm.IteratorForTest(
        *tm.make_batches(16, 4, 2, use_cupy=True),
        cache=None,
        on_host=True,
    )
    Xy = xgb.ExtMemQuantileDMatrix(it)

    folds = xcv.cross_validate(Xy, k_folds=3)

    assert isinstance(folds.handle, ctypes.c_void_p)
    assert folds.handle.value is not None
    assert folds.k_folds == 3

    cv_folds = xcv.CvFolds(k_folds=3)
    gpairs = xcv.CvFoldGpairs()
    assert xcv.get_gradient(Xy, cv_folds, folds, iteration=0, out=gpairs) is gpairs

    assert isinstance(gpairs.handle, ctypes.c_void_p)
    assert gpairs.handle.value is not None
    grad, hess = gpairs.get(0)
    assert grad.shape == hess.shape
    assert grad.dtype == hess.dtype
    assert grad.data.ptr + ctypes.sizeof(ctypes.c_float) == hess.data.ptr
    assert grad.strides == hess.strides
    assert grad.strides == (
        2 * ctypes.sizeof(ctypes.c_float),
        2 * ctypes.sizeof(ctypes.c_float),
    )
    assert xcv.get_gradient(Xy, cv_folds, folds, iteration=1, out=gpairs) is gpairs
