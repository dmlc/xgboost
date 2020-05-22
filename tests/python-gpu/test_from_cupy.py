import numpy as np
import xgboost as xgb
import sys
import pytest

sys.path.append("tests/python")
import testing as tm


def dmatrix_from_cupy(input_type, DMatrixT, missing=np.NAN):
    '''Test constructing DMatrix from cupy'''
    import cupy as cp

    kRows = 80
    kCols = 3

    np_X = np.random.randn(kRows, kCols).astype(dtype=input_type)
    X = cp.array(np_X)
    X[5, 0] = missing
    X[3, 1] = missing
    y = cp.random.randn(kRows).astype(dtype=input_type)
    dtrain = DMatrixT(X, missing=missing, label=y)
    assert dtrain.num_col() == kCols
    assert dtrain.num_row() == kRows

    if DMatrixT is xgb.DeviceQuantileDMatrix:
        # Slice is not supported by DeviceQuantileDMatrix
        with pytest.raises(xgb.core.XGBoostError):
            dtrain.slice(rindex=[0, 1, 2])
            dtrain.slice(rindex=[0, 1, 2])
    else:
        dtrain.slice(rindex=[0, 1, 2])
        dtrain.slice(rindex=[0, 1, 2])

    return dtrain


def _test_from_cupy(DMatrixT):
    '''Test constructing DMatrix from cupy'''
    import cupy as cp
    dmatrix_from_cupy(np.float32, DMatrixT, np.NAN)
    dmatrix_from_cupy(np.float64, DMatrixT, np.NAN)

    dmatrix_from_cupy(np.uint8, DMatrixT, 2)
    dmatrix_from_cupy(np.uint32, DMatrixT, 3)
    dmatrix_from_cupy(np.uint64, DMatrixT, 4)

    dmatrix_from_cupy(np.int8, DMatrixT, 2)
    dmatrix_from_cupy(np.int32, DMatrixT, -2)
    dmatrix_from_cupy(np.int64, DMatrixT, -3)

    with pytest.raises(Exception):
        X = cp.random.randn(2, 2, dtype="float32")
        DMatrixT(X, label=X)


def _test_cupy_training(DMatrixT):
    import cupy as cp
    np.random.seed(1)
    cp.random.seed(1)
    X = cp.random.randn(50, 10, dtype="float32")
    y = cp.random.randn(50, dtype="float32")
    weights = np.random.random(50) + 1
    cupy_weights = cp.array(weights)
    base_margin = np.random.random(50)
    cupy_base_margin = cp.array(base_margin)

    evals_result_cupy = {}
    dtrain_cp = DMatrixT(X, y, weight=cupy_weights, base_margin=cupy_base_margin)
    params = {'gpu_id': 0, 'nthread': 1, 'tree_method': 'gpu_hist'}
    xgb.train(params, dtrain_cp, evals=[(dtrain_cp, "train")],
              evals_result=evals_result_cupy)
    evals_result_np = {}
    dtrain_np = xgb.DMatrix(cp.asnumpy(X), cp.asnumpy(y), weight=weights,
                            base_margin=base_margin)
    xgb.train(params, dtrain_np, evals=[(dtrain_np, "train")],
              evals_result=evals_result_np)
    assert np.array_equal(evals_result_cupy["train"]["rmse"], evals_result_np["train"]["rmse"])


def _test_cupy_metainfo(DMatrixT):
    import cupy as cp
    n = 100
    X = np.random.random((n, 2))
    dmat_cupy = DMatrixT(cp.array(X))
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
    assert np.array_equal(dmat.get_float_info('weight'),
                          dmat_cupy.get_float_info('weight'))
    assert np.array_equal(dmat.get_float_info('label'),
                          dmat_cupy.get_float_info('label'))
    assert np.array_equal(dmat.get_float_info('base_margin'),
                          dmat_cupy.get_float_info('base_margin'))
    assert np.array_equal(dmat.get_uint_info('group_ptr'),
                          dmat_cupy.get_uint_info('group_ptr'))


class TestFromCupy:
    '''Tests for constructing DMatrix from data structure conforming Apache
Arrow specification.'''

    @pytest.mark.skipif(**tm.no_cupy())
    def test_simple_dmat_from_cupy(self):
        _test_from_cupy(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_device_dmat_from_cupy(self):
        _test_from_cupy(xgb.DeviceQuantileDMatrix)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_cupy_training_device_dmat(self):
        _test_cupy_training(xgb.DeviceQuantileDMatrix)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_cupy_training_simple_dmat(self):
        _test_cupy_training(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_cupy_metainfo_simple_dmat(self):
        _test_cupy_metainfo(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_cupy_metainfo_device_dmat(self):
        _test_cupy_metainfo(xgb.DeviceQuantileDMatrix)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dlpack_simple_dmat(self):
        import cupy as cp
        n = 100
        X = cp.random.random((n, 2))
        xgb.DMatrix(X.toDlpack())

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dlpack_device_dmat(self):
        import cupy as cp
        n = 100
        X = cp.random.random((n, 2))
        m = xgb.DeviceQuantileDMatrix(X.toDlpack())
        with pytest.raises(xgb.core.XGBoostError):
            m.slice(rindex=[0, 1, 2])

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.mgpu
    def test_specified_device(self):
        import cupy as cp
        cp.cuda.runtime.setDevice(0)
        dtrain = dmatrix_from_cupy(
            np.float32, xgb.DeviceQuantileDMatrix, np.nan)
        with pytest.raises(xgb.core.XGBoostError):
            xgb.train({'tree_method': 'gpu_hist', 'gpu_id': 1},
                      dtrain, num_boost_round=10)
