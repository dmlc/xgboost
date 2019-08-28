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
    kCols = 2

    na = np.random.randn(kRows, kCols).astype(input_type)
    na[3, 1] = missing
    na[5, 0] = missing

    pa = pd.DataFrame(na)

    np_label = np.random.randn(kRows).astype(input_type)
    pa_label = pd.DataFrame(np_label)

    names = []

    for i in range(0, kCols):
        names.append(str(i))
    pa.columns = names

    cd: cudf.DataFrame = cudf.from_pandas(pa)
    cd_label: cudf.DataFrame = cudf.from_pandas(pa_label)

    dtrain = xgb.DMatrix(cd, label=cd_label, missing=missing)
    assert dtrain.num_col() == kCols
    assert dtrain.num_row() == kRows


class TestFromColumnar:
    '''Tests for constructing DMatrix from data structure conforming Apache
Arrow specification.'''

    @pytest.mark.skipif(**tm.no_cudf())
    def test_from_cudf(self):
        '''Test constructing DMatrix from cudf'''
        dmatrix_from_cudf(np.float32, np.NAN)
        dmatrix_from_cudf(np.float64, np.NAN)

        dmatrix_from_cudf(np.uint8, 2)
        dmatrix_from_cudf(np.uint32, 3)
        dmatrix_from_cudf(np.uint64, 4)

        dmatrix_from_cudf(np.int8, 2)
        dmatrix_from_cudf(np.int32, -2)
        dmatrix_from_cudf(np.int64, -3)
