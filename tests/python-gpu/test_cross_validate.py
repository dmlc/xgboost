# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0
import ctypes

import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost import _cross_validation as xcv


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
