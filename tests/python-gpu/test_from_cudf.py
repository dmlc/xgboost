import numpy as np
import xgboost as xgb
import sys
import pytest

sys.path.append("tests/python")
import testing as tm


def dmatrix_from_cudf(input_type, DMatrixT, missing=np.NAN):
    '''Test constructing DMatrix from cudf'''
    import cudf
    import pandas as pd

    kRows = 80
    kCols = 3

    na = np.random.randn(kRows, kCols)
    na[:, 0:2] = na[:, 0:2].astype(input_type)

    na[5, 0] = missing
    na[3, 1] = missing

    pa = pd.DataFrame({'0': na[:, 0],
                       '1': na[:, 1],
                       '2': na[:, 2].astype(np.int32)})

    np_label = np.random.randn(kRows).astype(input_type)
    pa_label = pd.DataFrame(np_label)

    cd = cudf.from_pandas(pa)
    cd_label = cudf.from_pandas(pa_label).iloc[:, 0]

    dtrain = DMatrixT(cd, missing=missing, label=cd_label)
    assert dtrain.num_col() == kCols
    assert dtrain.num_row() == kRows


def _test_from_cudf(DMatrixT):
    '''Test constructing DMatrix from cudf'''
    import cudf
    dmatrix_from_cudf(np.float32, DMatrixT, np.NAN)
    dmatrix_from_cudf(np.float64, DMatrixT, np.NAN)

    dmatrix_from_cudf(np.int8, DMatrixT, 2)
    dmatrix_from_cudf(np.int32, DMatrixT, -2)
    dmatrix_from_cudf(np.int64, DMatrixT, -3)

    cd = cudf.DataFrame({'x': [1, 2, 3], 'y': [0.1, 0.2, 0.3]})
    dtrain = DMatrixT(cd)

    assert dtrain.feature_names == ['x', 'y']
    assert dtrain.feature_types == ['int', 'float']

    series = cudf.DataFrame({'x': [1, 2, 3]}).iloc[:, 0]
    assert isinstance(series, cudf.Series)
    dtrain = DMatrixT(series)

    assert dtrain.feature_names == ['x']
    assert dtrain.feature_types == ['int']

    with pytest.raises(Exception):
        dtrain = DMatrixT(cd, label=cd)

    # Test when number of elements is less than 8
    X = cudf.DataFrame({'x': cudf.Series([0, 1, 2, np.NAN, 4],
                                         dtype=np.int32)})
    dtrain = DMatrixT(X)
    assert dtrain.num_col() == 1
    assert dtrain.num_row() == 5

    # Boolean is not supported.
    X_boolean = cudf.DataFrame({'x': cudf.Series([True, False])})
    with pytest.raises(Exception):
        dtrain = DMatrixT(X_boolean)

    y_boolean = cudf.DataFrame({
        'x': cudf.Series([True, False, True, True, True])})
    with pytest.raises(Exception):
        dtrain = DMatrixT(X_boolean, label=y_boolean)


def _test_cudf_training(DMatrixT):
    from cudf import DataFrame as df
    import pandas as pd
    np.random.seed(1)
    X = pd.DataFrame(np.random.randn(50, 10))
    y = pd.DataFrame(np.random.randn(50))
    weights = np.random.random(50) + 1.0
    cudf_weights = df.from_pandas(pd.DataFrame(weights))
    base_margin = np.random.random(50)
    cudf_base_margin = df.from_pandas(pd.DataFrame(base_margin))

    evals_result_cudf = {}
    dtrain_cudf = DMatrixT(df.from_pandas(X), df.from_pandas(y), weight=cudf_weights,
                           base_margin=cudf_base_margin)
    params = {'gpu_id': 0, 'tree_method': 'gpu_hist'}
    xgb.train(params, dtrain_cudf, evals=[(dtrain_cudf, "train")],
              evals_result=evals_result_cudf)
    evals_result_np = {}
    dtrain_np = xgb.DMatrix(X, y, weight=weights, base_margin=base_margin)
    xgb.train(params, dtrain_np, evals=[(dtrain_np, "train")],
              evals_result=evals_result_np)
    assert np.array_equal(evals_result_cudf["train"]["rmse"], evals_result_np["train"]["rmse"])


def _test_cudf_metainfo(DMatrixT):
    from cudf import DataFrame as df
    import pandas as pd
    n = 100
    X = np.random.random((n, 2))
    dmat_cudf = DMatrixT(df.from_pandas(pd.DataFrame(X)))
    dmat = xgb.DMatrix(X)
    floats = np.random.random(n)
    uints = np.array([4, 2, 8]).astype("uint32")
    cudf_floats = df.from_pandas(pd.DataFrame(floats))
    cudf_uints = df.from_pandas(pd.DataFrame(uints))
    dmat.set_float_info('weight', floats)
    dmat.set_float_info('label', floats)
    dmat.set_float_info('base_margin', floats)
    dmat.set_uint_info('group', uints)
    dmat_cudf.set_interface_info('weight', cudf_floats)
    dmat_cudf.set_interface_info('label', cudf_floats)
    dmat_cudf.set_interface_info('base_margin', cudf_floats)
    dmat_cudf.set_interface_info('group', cudf_uints)

    # Test setting info with cudf DataFrame
    assert np.array_equal(dmat.get_float_info('weight'), dmat_cudf.get_float_info('weight'))
    assert np.array_equal(dmat.get_float_info('label'), dmat_cudf.get_float_info('label'))
    assert np.array_equal(dmat.get_float_info('base_margin'),
                          dmat_cudf.get_float_info('base_margin'))
    assert np.array_equal(dmat.get_uint_info('group_ptr'), dmat_cudf.get_uint_info('group_ptr'))

    # Test setting info with cudf Series
    dmat_cudf.set_interface_info('weight', cudf_floats[cudf_floats.columns[0]])
    dmat_cudf.set_interface_info('label', cudf_floats[cudf_floats.columns[0]])
    dmat_cudf.set_interface_info('base_margin', cudf_floats[cudf_floats.columns[0]])
    dmat_cudf.set_interface_info('group', cudf_uints[cudf_uints.columns[0]])
    assert np.array_equal(dmat.get_float_info('weight'), dmat_cudf.get_float_info('weight'))
    assert np.array_equal(dmat.get_float_info('label'), dmat_cudf.get_float_info('label'))
    assert np.array_equal(dmat.get_float_info('base_margin'),
                          dmat_cudf.get_float_info('base_margin'))
    assert np.array_equal(dmat.get_uint_info('group_ptr'), dmat_cudf.get_uint_info('group_ptr'))


class TestFromColumnar:
    '''Tests for constructing DMatrix from data structure conforming Apache
Arrow specification.'''

    @pytest.mark.skipif(**tm.no_cudf())
    def test_simple_dmatrix_from_cudf(self):
        _test_from_cudf(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_device_dmatrix_from_cudf(self):
        _test_from_cudf(xgb.DeviceQuantileDMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_training_simple_dmatrix(self):
        _test_cudf_training(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_training_device_dmatrix(self):
        _test_cudf_training(xgb.DeviceQuantileDMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_metainfo_simple_dmatrix(self):
        _test_cudf_metainfo(xgb.DMatrix)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_cudf_metainfo_device_dmatrix(self):
        _test_cudf_metainfo(xgb.DeviceQuantileDMatrix)
