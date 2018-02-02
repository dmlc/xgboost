# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import testing as tm
import unittest

try:
    import pandas as pd
except ImportError:
    pass


tm._skip_if_no_pandas()


dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestPandas(unittest.TestCase):

    def test_pandas(self):

        df = pd.DataFrame([[1, 2., True], [2, 3., False]], columns=['a', 'b', 'c'])
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]))
        assert dm.feature_names == ['a', 'b', 'c']
        assert dm.feature_types == ['int', 'float', 'i']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

        # overwrite feature_names and feature_types
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]),
                         feature_names=['x', 'y', 'z'], feature_types=['q', 'q', 'q'])
        assert dm.feature_names == ['x', 'y', 'z']
        assert dm.feature_types == ['q', 'q', 'q']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

        # incorrect dtypes
        df = pd.DataFrame([[1, 2., 'x'], [2, 3., 'y']], columns=['a', 'b', 'c'])
        self.assertRaises(ValueError, xgb.DMatrix, df)

        # numeric columns
        df = pd.DataFrame([[1, 2., True], [2, 3., False]])
        dm = xgb.DMatrix(df, label=pd.Series([1, 2]))
        assert dm.feature_names == ['0', '1', '2']
        assert dm.feature_types == ['int', 'float', 'i']
        assert dm.num_row() == 2
        assert dm.num_col() == 3

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
        result, _, _ = xgb.core._maybe_pandas_data(dummies, None, None)
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

    def test_pandas_label(self):
        # label must be a single column
        df = pd.DataFrame({'A': ['X', 'Y', 'Z'], 'B': [1, 2, 3]})
        self.assertRaises(ValueError, xgb.core._maybe_pandas_label, df)

        # label must be supported dtype
        df = pd.DataFrame({'A': np.array(['a', 'b', 'c'], dtype=object)})
        self.assertRaises(ValueError, xgb.core._maybe_pandas_label, df)

        df = pd.DataFrame({'A': np.array([1, 2, 3], dtype=int)})
        result = xgb.core._maybe_pandas_label(df)
        np.testing.assert_array_equal(result, np.array([[1.], [2.], [3.]], dtype=float))

        dm = xgb.DMatrix(np.random.randn(3, 2), label=df)
        assert dm.num_row() == 3
        assert dm.num_col() == 2

    def test_cv_as_pandas(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

        import pandas as pd
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert cv.columns.equals(exp)

        # show progress log (result is the same as above)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    verbose_eval=True)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert cv.columns.equals(exp)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    verbose_eval=True, show_stdv=False)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert cv.columns.equals(exp)

        params = {'max_depth': 2, 'eta': 1, 'silent': 1,
                  'objective': 'binary:logistic', 'eval_metric': 'auc'}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=True)
        assert 'eval_metric' in params
        assert 'auc' in cv.columns[0]

        params = {'max_depth': 2, 'eta': 1, 'silent': 1,
                  'objective': 'binary:logistic', 'eval_metric': ['auc']}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=True)
        assert 'eval_metric' in params
        assert 'auc' in cv.columns[0]

        params = {'max_depth': 2, 'eta': 1, 'silent': 1,
                  'objective': 'binary:logistic', 'eval_metric': ['auc']}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, early_stopping_rounds=1)
        assert 'eval_metric' in params
        assert 'auc' in cv.columns[0]
        assert cv.shape[0] < 10

        params = {'max_depth': 2, 'eta': 1, 'silent': 1,
                  'objective': 'binary:logistic'}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, metrics='auc')
        assert 'auc' in cv.columns[0]

        params = {'max_depth': 2, 'eta': 1, 'silent': 1,
                  'objective': 'binary:logistic'}
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    as_pandas=True, metrics=['auc'])
        assert 'auc' in cv.columns[0]

        params = {'max_depth': 2, 'eta': 1, 'silent': 1,
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
