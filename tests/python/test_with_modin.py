import numpy as np
import pytest
from test_dmatrix import set_base_margin_info

import xgboost as xgb
from xgboost import testing as tm

try:
    import modin.pandas as md
except ImportError:
    pass


pytestmark = pytest.mark.skipif(**tm.no_modin())


class TestModin:
    @pytest.mark.xfail
    def test_modin(self):
        df = md.DataFrame([[1, 2., True], [2, 3., False]],
                          columns=['a', 'b', 'c'])
        dm = xgb.DMatrix(df, label=md.Series([1, 2]))
        assert dm.feature_names == ['a', 'b', 'c']
        assert dm.feature_types == ['int', 'float', 'i']
        assert dm.num_row() == 2
        assert dm.num_col() == 3
        np.testing.assert_array_equal(dm.get_label(), np.array([1, 2]))

        # overwrite feature_names and feature_types
        dm = xgb.DMatrix(df, label=md.Series([1, 2]),
                         feature_names=['x', 'y', 'z'],
                         feature_types=['q', 'q', 'q'])
        assert dm.feature_names == ['x', 'y', 'z']
        assert dm.feature_types == ['q', 'q', 'q']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

        # incorrect dtypes
        df = md.DataFrame([[1, 2., 'x'], [2, 3., 'y']],
                          columns=['a', 'b', 'c'])
        with pytest.raises(ValueError):
            xgb.DMatrix(df)

        # numeric columns
        df = md.DataFrame([[1, 2., True], [2, 3., False]])
        dm = xgb.DMatrix(df, label=md.Series([1, 2]))
        assert dm.feature_names == ['0', '1', '2']
        assert dm.feature_types == ['int', 'float', 'i']
        assert dm.num_row() == 2
        assert dm.num_col() == 3
        np.testing.assert_array_equal(dm.get_label(), np.array([1, 2]))

        df = md.DataFrame([[1, 2., 1], [2, 3., 1]], columns=[4, 5, 6])
        dm = xgb.DMatrix(df, label=md.Series([1, 2]))
        assert dm.feature_names == ['4', '5', '6']
        assert dm.feature_types == ['int', 'float', 'int']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

        df = md.DataFrame({'A': ['X', 'Y', 'Z'], 'B': [1, 2, 3]})
        dummies = md.get_dummies(df)
        #    B  A_X  A_Y  A_Z
        # 0  1    1    0    0
        # 1  2    0    1    0
        # 2  3    0    0    1
        result, _, _ = xgb.data._transform_pandas_df(dummies,
                                                     enable_categorical=False)
        exp = np.array([[1., 1., 0., 0.],
                        [2., 0., 1., 0.],
                        [3., 0., 0., 1.]])
        np.testing.assert_array_equal(result, exp)
        dm = xgb.DMatrix(dummies)
        assert dm.feature_names == ['B', 'A_X', 'A_Y', 'A_Z']
        assert dm.feature_types == ['int', 'int', 'int', 'int']
        assert dm.num_row() == 3
        assert dm.num_col() == 4

        df = md.DataFrame({'A=1': [1, 2, 3], 'A=2': [4, 5, 6]})
        dm = xgb.DMatrix(df)
        assert dm.feature_names == ['A=1', 'A=2']
        assert dm.feature_types == ['int', 'int']
        assert dm.num_row() == 3
        assert dm.num_col() == 2

        df_int = md.DataFrame([[1, 1.1], [2, 2.2]], columns=[9, 10])
        dm_int = xgb.DMatrix(df_int)
        df_range = md.DataFrame([[1, 1.1], [2, 2.2]], columns=range(9, 11, 1))
        dm_range = xgb.DMatrix(df_range)
        assert dm_int.feature_names == ['9', '10']  # assert not "9 "
        assert dm_int.feature_names == dm_range.feature_names

        # test MultiIndex as columns
        df = md.DataFrame(
            [
                (1, 2, 3, 4, 5, 6),
                (6, 5, 4, 3, 2, 1)
            ],
            columns=md.MultiIndex.from_tuples((
                ('a', 1), ('a', 2), ('a', 3),
                ('b', 1), ('b', 2), ('b', 3),
            ))
        )
        dm = xgb.DMatrix(df)
        assert dm.feature_names == ['a 1', 'a 2', 'a 3', 'b 1', 'b 2', 'b 3']
        assert dm.feature_types == ['int', 'int', 'int', 'int', 'int', 'int']
        assert dm.num_row() == 2
        assert dm.num_col() == 6

    def test_modin_label(self):
        # label must be a single column
        df = md.DataFrame({'A': ['X', 'Y', 'Z'], 'B': [1, 2, 3]})
        with pytest.raises(ValueError):
            xgb.data._transform_pandas_df(df, False, None, None, 'label', 'float')

        # label must be supported dtype
        df = md.DataFrame({'A': np.array(['a', 'b', 'c'], dtype=object)})
        with pytest.raises(ValueError):
            xgb.data._transform_pandas_df(df, False, None, None, 'label', 'float')

        df = md.DataFrame({'A': np.array([1, 2, 3], dtype=int)})
        result, _, _ = xgb.data._transform_pandas_df(df, False, None, None,
                                                     'label', 'float')
        np.testing.assert_array_equal(result, np.array([[1.], [2.], [3.]],
                                                       dtype=float))
        dm = xgb.DMatrix(np.random.randn(3, 2), label=df)
        assert dm.num_row() == 3
        assert dm.num_col() == 2

    def test_modin_weight(self):
        kRows = 32
        kCols = 8

        X = np.random.randn(kRows, kCols)
        y = np.random.randn(kRows)
        w = np.random.uniform(size=kRows).astype(np.float32)
        w_pd = md.DataFrame(w)
        data = xgb.DMatrix(X, y, w_pd)

        assert data.num_row() == kRows
        assert data.num_col() == kCols

        np.testing.assert_array_equal(data.get_weight(), w)

    def test_base_margin(self):
        set_base_margin_info(md.DataFrame, xgb.DMatrix, "hist")
