# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import testing as tm
import unittest
import pytest

try:
    import pandas as pd
except ImportError:
    pass


pytestmark = pytest.mark.skipif(**tm.no_pandas())


dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestPandas(unittest.TestCase):

    def test_pandas(self):

        df = pd.DataFrame([[1, 2., True], [2, 3., False]],
                          columns=['a', 'b', 'c'])
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]))
        assert dm.feature_names == ['a', 'b', 'c']
        assert dm.feature_types == ['int', 'float', 'i']
        assert dm.num_row() == 2
        assert dm.num_col() == 3
        np.testing.assert_array_equal(dm.get_label(), np.array([1, 2]))

        # overwrite feature_names and feature_types
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]),
                         feature_names=['x', 'y', 'z'],
                         feature_types=['q', 'q', 'q'])
        assert dm.feature_names == ['x', 'y', 'z']
        assert dm.feature_types == ['q', 'q', 'q']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

        # incorrect dtypes
        df = pd.DataFrame([[1, 2., 'x'], [2, 3., 'y']],
                          columns=['a', 'b', 'c'])
        self.assertRaises(ValueError, xgb.DMatrix, df)

        # numeric columns
        df = pd.DataFrame([[1, 2., True], [2, 3., False]])
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]))
        assert dm.feature_names == ['0', '1', '2']
        assert dm.feature_types == ['int', 'float', 'i']
        assert dm.num_row() == 2
        assert dm.num_col() == 3
        np.testing.assert_array_equal(dm.get_label(), np.array([1, 2]))

        df = pd.DataFrame([[1, 2., 1], [2, 3., 1]], columns=[4, 5, 6])
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]))
        assert dm.feature_names == ['4', '5', '6']
        assert dm.feature_types == ['int', 'float', 'int']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

        df = pd.DataFrame({'A': ['X', 'Y', 'Z'], 'B': [1, 2, 3]})
        dummies = pd.get_dummies(df)
        #    B  A_X  A_Y  A_Z
        # 0  1    1    0    0
        # 1  2    0    1    0
        # 2  3    0    0    1
        pandas_handler = xgb.data.PandasHandler(np.nan, 0, False)
        result, _, _ = pandas_handler._maybe_pandas_data(dummies, None, None)
        exp = np.array([[1., 1., 0., 0.],
                        [2., 0., 1., 0.],
                        [3., 0., 0., 1.]])
        np.testing.assert_array_equal(result, exp)
        dm = xgb.DMatrix(dummies)
        assert dm.feature_names == ['B', 'A_X', 'A_Y', 'A_Z']
        assert dm.feature_types == ['int', 'int', 'int', 'int']
        assert dm.num_row() == 3
        assert dm.num_col() == 4

        df = pd.DataFrame({'A=1': [1, 2, 3], 'A=2': [4, 5, 6]})
        dm = xgb.DMatrix(df)
        assert dm.feature_names == ['A=1', 'A=2']
        assert dm.feature_types == ['int', 'int']
        assert dm.num_row() == 3
        assert dm.num_col() == 2

        df_int = pd.DataFrame([[1, 1.1], [2, 2.2]], columns=[9, 10])
        dm_int = xgb.DMatrix(df_int)
        df_range = pd.DataFrame([[1, 1.1], [2, 2.2]], columns=range(9, 11, 1))
        dm_range = xgb.DMatrix(df_range)
        assert dm_int.feature_names == ['9', '10']  # assert not "9 "
        assert dm_int.feature_names == dm_range.feature_names

        # test MultiIndex as columns
        df = pd.DataFrame(
            [
                (1, 2, 3, 4, 5, 6),
                (6, 5, 4, 3, 2, 1)
            ],
            columns=pd.MultiIndex.from_tuples((
                ('a', 1), ('a', 2), ('a', 3),
                ('b', 1), ('b', 2), ('b', 3),
            ))
        )
        dm = xgb.DMatrix(df)
        assert dm.feature_names == ['a 1', 'a 2', 'a 3', 'b 1', 'b 2', 'b 3']
        assert dm.feature_types == ['int', 'int', 'int', 'int', 'int', 'int']
        assert dm.num_row() == 2
        assert dm.num_col() == 6

    def test_pandas_sparse(self):
        import pandas as pd
        rows = 100
        X = pd.DataFrame(
            {"A": pd.arrays.SparseArray(np.random.randint(0, 10, size=rows)),
             "B": pd.arrays.SparseArray(np.random.randn(rows)),
             "C": pd.arrays.SparseArray(np.random.permutation(
                 [True, False] * (rows // 2)))}
        )
        y = pd.Series(pd.arrays.SparseArray(np.random.randn(rows)))
        dtrain = xgb.DMatrix(X, y)
        booster = xgb.train({}, dtrain, num_boost_round=4)
        predt_sparse = booster.predict(xgb.DMatrix(X))
        predt_dense = booster.predict(xgb.DMatrix(X.sparse.to_dense()))
        np.testing.assert_allclose(predt_sparse, predt_dense)

    def test_pandas_label(self):
        # label must be a single column
        df = pd.DataFrame({'A': ['X', 'Y', 'Z'], 'B': [1, 2, 3]})
        pandas_handler = xgb.data.PandasHandler(np.nan, 0, False)
        self.assertRaises(ValueError, pandas_handler._maybe_pandas_data, df,
                          None, None, 'label', 'float')

        # label must be supported dtype
        df = pd.DataFrame({'A': np.array(['a', 'b', 'c'], dtype=object)})
        self.assertRaises(ValueError, pandas_handler._maybe_pandas_data, df,
                          None, None, 'label', 'float')

        df = pd.DataFrame({'A': np.array([1, 2, 3], dtype=int)})
        result, _, _ = pandas_handler._maybe_pandas_data(df, None, None,
                                                         'label', 'float')
        np.testing.assert_array_equal(result, np.array([[1.], [2.], [3.]],
                                                       dtype=float))
        dm = xgb.DMatrix(np.random.randn(3, 2), label=df)
        assert dm.num_row() == 3
        assert dm.num_col() == 2

    def test_pandas_weight(self):
        kRows = 32
        kCols = 8

        X = np.random.randn(kRows, kCols)
        y = np.random.randn(kRows)
        w = np.random.randn(kRows).astype(np.float32)
        w_pd = pd.DataFrame(w)
        data = xgb.DMatrix(X, y, w_pd)

        assert data.num_row() == kRows
        assert data.num_col() == kCols

        np.testing.assert_array_equal(data.get_weight(), w)

    def test_cv_as_pandas(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic'}

        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert len(cv.columns.intersection(exp)) == 4

        # show progress log (result is the same as above)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    verbose_eval=True)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert len(cv.columns.intersection(exp)) == 4
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    verbose_eval=True, show_stdv=False)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert len(cv.columns.intersection(exp)) == 4

        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic', 'eval_metric': 'auc'}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=True)
        assert 'eval_metric' in params
        assert 'auc' in cv.columns[0]

        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic', 'eval_metric': ['auc']}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=True)
        assert 'eval_metric' in params
        assert 'auc' in cv.columns[0]

        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic', 'eval_metric': ['auc']}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, early_stopping_rounds=1)
        assert 'eval_metric' in params
        assert 'auc' in cv.columns[0]
        assert cv.shape[0] < 10

        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic'}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, metrics='auc')
        assert 'auc' in cv.columns[0]

        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic'}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, metrics=['auc'])
        assert 'auc' in cv.columns[0]

        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic', 'eval_metric': ['auc']}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, metrics='error')
        assert 'eval_metric' in params
        assert 'auc' not in cv.columns[0]
        assert 'error' in cv.columns[0]

        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, metrics=['error'])
        assert 'eval_metric' in params
        assert 'auc' not in cv.columns[0]
        assert 'error' in cv.columns[0]

        params = list(params.items())
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, metrics=['error'])
        assert isinstance(params, list)
        assert 'auc' not in cv.columns[0]
        assert 'error' in cv.columns[0]
