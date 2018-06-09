# -*- coding: utf-8 -*-
import unittest

import testing as tm
import xgboost as xgb

try:
    import datatable as dt
    import pandas as pd
except ImportError:
    pass

tm._skip_if_no_dt()
tm._skip_if_no_pandas()


class TestDataTable(unittest.TestCase):

    def test_dt(self):
        df = pd.DataFrame([[1, 2., True], [2, 3., False]], columns=['a', 'b', 'c'])
        dtable = dt.Frame(df)
        labels = dt.Frame([1, 2])
        dm = xgb.DMatrix(dtable, label=labels)
        assert dm.feature_names == ['a', 'b', 'c']
        assert dm.feature_types == ['int', 'float', 'i']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

        # overwrite feature_names
        dm = xgb.DMatrix(dtable, label=pd.Series([1, 2]),
                         feature_names=['x', 'y', 'z'])
        assert dm.feature_names == ['x', 'y', 'z']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

        # incorrect dtypes
        df = pd.DataFrame([[1, 2., 'x'], [2, 3., 'y']], columns=['a', 'b', 'c'])
        dtable = dt.Frame(df)
        self.assertRaises(ValueError, xgb.DMatrix, dtable)

        df = pd.DataFrame({'A=1': [1, 2, 3], 'A=2': [4, 5, 6]})
        dtable = dt.Frame(df)
        dm = xgb.DMatrix(dtable)
        assert dm.feature_names == ['A=1', 'A=2']
        assert dm.feature_types == ['int', 'int']
        assert dm.num_row() == 3
        assert dm.num_col() == 2
