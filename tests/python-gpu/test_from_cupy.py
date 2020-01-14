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


class TestFromArrayInterface:
    '''Tests for constructing DMatrix from data structure conforming Apache
Arrow specification.'''

    @pytest.mark.skipif(**tm.no_cupy())
    def test_from_cupy(self):
        '''Test constructing DMatrix from cudf'''
        import cupy
        dmatrix_from_cupy(np.float32, np.NAN)
        dmatrix_from_cupy(np.float64, np.NAN)

        dmatrix_from_cupy(np.uint8, 2)
        dmatrix_from_cupy(np.uint32, 3)
        dmatrix_from_cupy(np.uint64, 4)

        dmatrix_from_cupy(np.int8, 2)
        dmatrix_from_cupy(np.int32, -2)
        dmatrix_from_cupy(np.int64, -3)

        with pytest.raises(Exception):
            np_X = cp.random.randn(2, 2, dtype="float32")
            dtrain = xgb.DMatrix(X, label=X)
