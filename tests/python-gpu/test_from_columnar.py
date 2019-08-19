import numpy as np
import xgboost as xgb
import sys
import pytest
sys.path.append("tests/python")
import testing as tm

pytestmark = pytest.mark.skipif(**tm.no_cudf())


class TestFromColumnar:
    '''Tests for constructing DMatrix from data structure conforming Apache
Arrow specification.'''

    @pytest.mark.skipif(**tm.no_cudf())
    def test_from_cudf():
        '''Test constructing DMatrix from cudf'''
        import cudf
        import pandas as pd

        kRows = 80
        kCols = 2

        na = np.random.randn(kRows, kCols).astype(np.float32)
        na[3, 1] = np.NAN
        na[5, 0] = np.NAN

        pa = pd.DataFrame(na)

        np_label = np.random.randn(kRows).astype(np.float32)
        pa_label = pd.DataFrame(np_label)

        names = []

        for i in range(0, kCols):
            names.append(str(i))
        pa.columns = names

        cd: cudf.DataFrame = cudf.from_pandas(pa)
        cd_label: cudf.DataFrame = cudf.from_pandas(pa_label)

        dtrain = xgb.DMatrix(cd, label=cd_label)
        assert dtrain.num_col() == kCols
        assert dtrain.num_row() == kRows
