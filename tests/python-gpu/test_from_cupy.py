import numpy as np
import xgboost as xgb
import sys
import pytest

sys.path.append("tests/python")
import testing as tm


def dmatrix_from_cupy(input_type, missing=np.NAN):
    '''Test constructing DMatrix from cupy'''
    import cupy as cp

    kRows = 80
    kCols = 3

    np_X = np.random.randn(kRows, kCols).astype(dtype=input_type)
    X = cp.array(np_X)
    X[5, 0] = missing
    X[3, 1] = missing
    y = cp.random.randn(kRows).astype(dtype=input_type)
    dtrain = xgb.DMatrix(X, missing=missing, label=y)
    assert dtrain.num_col() == kCols
    assert dtrain.num_row() == kRows
    return dtrain


class TestFromArrayInterface:
    '''Tests for constructing DMatrix from data structure conforming Apache
Arrow specification.'''

    @pytest.mark.skipif(**tm.no_cupy())
    def test_from_cupy(self):
        '''Test constructing DMatrix from cudf'''
        import cupy as cp
        dmatrix_from_cupy(np.float32, np.NAN)
        dmatrix_from_cupy(np.float64, np.NAN)

        dmatrix_from_cupy(np.uint8, 2)
        dmatrix_from_cupy(np.uint32, 3)
        dmatrix_from_cupy(np.uint64, 4)

        dmatrix_from_cupy(np.int8, 2)
        dmatrix_from_cupy(np.int32, -2)
        dmatrix_from_cupy(np.int64, -3)

        with pytest.raises(Exception):
            X = cp.random.randn(2, 2, dtype="float32")
            dtrain = xgb.DMatrix(X, label=X)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_cupy_training(self):
        import cupy as cp
        X = cp.random.randn(50, 10, dtype="float32")
        y = cp.random.randn(50, dtype="float32")

        evals_result_cupy = {}
        dtrain_cp = xgb.DMatrix(X, y)
        xgb.train({'gpu_id': 0}, dtrain_cp, evals=[(dtrain_cp, "train")],
                  evals_result=evals_result_cupy)
        evals_result_np = {}
        dtrain_np = xgb.DMatrix(cp.asnumpy(X), cp.asnumpy(y))
        xgb.train({'gpu_id': 0}, dtrain_np, evals=[(dtrain_np, "train")],
                  evals_result=evals_result_np)
        assert np.array_equal(evals_result_cupy["train"]["rmse"], evals_result_np["train"]["rmse"])
