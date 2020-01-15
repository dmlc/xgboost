import numpy as np
import xgboost as xgb
import sys
import pytest
sys.path.append("tests/python")
import testing as tm


def dmatrix_from_cudf(input_type, missing=np.NAN):
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

    dtrain = xgb.DMatrix(cd, missing=missing, label=cd_label)
    assert dtrain.num_col() == kCols
    assert dtrain.num_row() == kRows


class TestFromColumnar:
    '''Tests for constructing DMatrix from data structure conforming Apache
Arrow specification.'''

    @pytest.mark.skipif(**tm.no_cudf())
    def test_from_cudf(self):
        '''Test constructing DMatrix from cudf'''
        import cudf
        dmatrix_from_cudf(np.float32, np.NAN)
        dmatrix_from_cudf(np.float64, np.NAN)

        dmatrix_from_cudf(np.uint8, 2)
        dmatrix_from_cudf(np.uint32, 3)
        dmatrix_from_cudf(np.uint64, 4)

        dmatrix_from_cudf(np.int8, 2)
        dmatrix_from_cudf(np.int32, -2)
        dmatrix_from_cudf(np.int64, -3)

        cd = cudf.DataFrame({'x': [1, 2, 3], 'y': [0.1, 0.2, 0.3]})
        dtrain = xgb.DMatrix(cd)

        assert dtrain.feature_names == ['x', 'y']
        assert dtrain.feature_types == ['int', 'float']

        series = cudf.DataFrame({'x': [1, 2, 3]}).iloc[:, 0]
        assert isinstance(series, cudf.Series)
        dtrain = xgb.DMatrix(series)

        assert dtrain.feature_names == ['x']
        assert dtrain.feature_types == ['int']

        with pytest.raises(Exception):
            dtrain = xgb.DMatrix(cd, label=cd)

        # Test when number of elements is less than 8
        X = cudf.DataFrame({'x': cudf.Series([0, 1, 2, np.NAN, 4],
                                             dtype=np.int32)})
        dtrain = xgb.DMatrix(X)
        assert dtrain.num_col() == 1
        assert dtrain.num_row() == 5

        # Boolean is not supported.
        X_boolean = cudf.DataFrame({'x': cudf.Series([True, False])})
        with pytest.raises(Exception):
            dtrain = xgb.DMatrix(X_boolean)

        y_boolean = cudf.DataFrame({
            'x': cudf.Series([True, False, True, True, True])})
        with pytest.raises(Exception):
            dtrain = xgb.DMatrix(X_boolean, label=y_boolean)
