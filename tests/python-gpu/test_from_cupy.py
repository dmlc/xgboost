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
        '''Test constructing DMatrix from cupy'''
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
        weights = np.random.random(50)
        cupy_weights = cp.array(weights)
        base_margin = np.random.random(50)
        cupy_base_margin = cp.array(base_margin)

        evals_result_cupy = {}
        dtrain_cp = xgb.DMatrix(X, y, weight=cupy_weights, base_margin=cupy_base_margin)
        xgb.train({'gpu_id': 0}, dtrain_cp, evals=[(dtrain_cp, "train")],
                  evals_result=evals_result_cupy)
        evals_result_np = {}
        dtrain_np = xgb.DMatrix(cp.asnumpy(X), cp.asnumpy(y), weight=weights,
                                base_margin=base_margin)
        xgb.train({'gpu_id': 0}, dtrain_np, evals=[(dtrain_np, "train")],
                  evals_result=evals_result_np)
        assert np.array_equal(evals_result_cupy["train"]["rmse"], evals_result_np["train"]["rmse"])

    @pytest.mark.skipif(**tm.no_cupy())
    def test_cupy_metainfo(self):
        import cupy as cp
        n = 100
        X = np.random.random((n, 2))
        dmat_cupy = xgb.DMatrix(X)
        dmat = xgb.DMatrix(X)
        floats = np.random.random(n)
        uints = np.array([4, 2, 8]).astype("uint32")
        cupy_floats = cp.array(floats)
        cupy_uints = cp.array(uints)
        dmat.set_float_info('weight', floats)
        dmat.set_float_info('label', floats)
        dmat.set_float_info('base_margin', floats)
        dmat.set_uint_info('group', uints)
        dmat_cupy.set_interface_info('weight', cupy_floats)
        dmat_cupy.set_interface_info('label', cupy_floats)
        dmat_cupy.set_interface_info('base_margin', cupy_floats)
        dmat_cupy.set_interface_info('group', cupy_uints)

        # Test setting info with cupy 
        assert np.array_equal(dmat.get_float_info('weight'), dmat_cupy.get_float_info('weight'))
        assert np.array_equal(dmat.get_float_info('label'), dmat_cupy.get_float_info('label'))
        assert np.array_equal(dmat.get_float_info('base_margin'),
                              dmat_cupy.get_float_info('base_margin'))
        assert np.array_equal(dmat.get_uint_info('group_ptr'), dmat_cupy.get_uint_info('group_ptr'))
