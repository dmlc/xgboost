# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import unittest

import matplotlib
matplotlib.use('Agg')

dpath = 'demo/data/'
rng = np.random.RandomState(1994)

class TestBasic(unittest.TestCase):

    def test_basic(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
        # specify validations set to watch performance
        watchlist  = [(dtest,'eval'), (dtrain,'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        # save dmatrix into binary buffer
        dtest.save_binary('dtest.buffer')
        # save model
        bst.save_model('xgb.model')
        # load model and data in
        bst2 = xgb.Booster(model_file='xgb.model')
        dtest2 = xgb.DMatrix('dtest.buffer')
        preds2 = bst2.predict(dtest2)
        # assert they are the same
        assert np.sum(np.abs(preds2-preds)) == 0

    def test_dmatrix_init(self):
        data = np.random.randn(5, 5)

        # different length
        self.assertRaises(ValueError, xgb.DMatrix, data,
                          feature_names=list('abcdef'))
        # contains duplicates
        self.assertRaises(ValueError, xgb.DMatrix, data,
                          feature_names=['a', 'b', 'c', 'd', 'd'])
        # contains symbol
        self.assertRaises(ValueError, xgb.DMatrix, data,
                          feature_names=['a', 'b', 'c', 'd', 'e<1'])

        dm = xgb.DMatrix(data)
        dm.feature_names = list('abcde')
        assert dm.feature_names == list('abcde')

        dm.feature_types = 'q'
        assert dm.feature_types == list('qqqqq')

        dm.feature_types = list('qiqiq')
        assert dm.feature_types == list('qiqiq')

        def incorrect_type_set():
            dm.feature_types = list('abcde')
        self.assertRaises(ValueError, incorrect_type_set)

        # reset
        dm.feature_names = None
        assert dm.feature_names is None
        assert dm.feature_types is None

    def test_feature_names(self):
        data = np.random.randn(100, 5)
        target = np.array([0, 1] * 50)

        cases = [['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
                 [u'要因1', u'要因2', u'要因3', u'要因4', u'要因5']]

        for features in cases:
            dm = xgb.DMatrix(data, label=target,
                             feature_names=features)
            assert dm.feature_names == features
            assert dm.num_row() == 100
            assert dm.num_col() == 5

            params={'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'eta': 0.3,
                    'num_class': 3}

            bst = xgb.train(params, dm, num_boost_round=10)
            scores = bst.get_fscore()
            assert list(sorted(k for k in scores)) == features

            dummy = np.random.randn(5, 5)
            dm = xgb.DMatrix(dummy, feature_names=features)
            bst.predict(dm)

            # different feature name must raises error
            dm = xgb.DMatrix(dummy, feature_names=list('abcde'))
            self.assertRaises(ValueError, bst.predict, dm)

    def test_pandas(self):
        import pandas as pd
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
        exp = np.array([[ 1.,  1.,  0.,  0.],
                        [ 2.,  0.,  1.,  0.],
                        [ 3.,  0.,  0.,  1.]])
        np.testing.assert_array_equal(result, exp)

        dm = xgb.DMatrix(dummies)
        assert dm.feature_names == ['B', 'A_X', 'A_Y', 'A_Z']
        assert dm.feature_types == ['int', 'float', 'float', 'float']
        assert dm.num_row() == 3
        assert dm.num_col() == 4

        df = pd.DataFrame({'A=1': [1, 2, 3], 'A=2': [4, 5, 6]})
        dm = xgb.DMatrix(df)
        assert dm.feature_names == ['A=1', 'A=2']
        assert dm.feature_types == ['int', 'int']
        assert dm.num_row() == 3
        assert dm.num_col() == 2

    def test_pandas_label(self):
        import pandas as pd

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


    def test_load_file_invalid(self):

        self.assertRaises(ValueError, xgb.Booster,
                          model_file='incorrect_path')

        self.assertRaises(ValueError, xgb.Booster,
                          model_file=u'不正なパス')

    def test_dmatrix_numpy_init(self):
        data = np.random.randn(5, 5)
        dm = xgb.DMatrix(data)
        assert dm.num_row() == 5
        assert dm.num_col() == 5

        data = np.matrix([[1, 2], [3, 4]])
        dm = xgb.DMatrix(data)
        assert dm.num_row() == 2
        assert dm.num_col() == 2

        # 0d array
        self.assertRaises(ValueError, xgb.DMatrix, np.array(1))
        # 1d array
        self.assertRaises(ValueError, xgb.DMatrix, np.array([1, 2, 3]))
        # 3d array
        data = np.random.randn(5, 5, 5)
        self.assertRaises(ValueError, xgb.DMatrix, data)
        # object dtype
        data = np.array([['a', 'b'], ['c', 'd']])
        self.assertRaises(ValueError, xgb.DMatrix, data)

    def test_cv(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }

        import pandas as pd
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert cv.columns.equals(exp)

        # show progress log (result is the same as above)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    show_progress=True)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert cv.columns.equals(exp)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    show_progress=True, show_stdv=False)
        assert isinstance(cv, pd.DataFrame)
        exp = pd.Index([u'test-error-mean', u'test-error-std',
                        u'train-error-mean', u'train-error-std'])
        assert cv.columns.equals(exp)

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=False)
        assert isinstance(cv, np.ndarray)
        assert cv.shape == (10, 4)

    def test_plotting(self):
        bst2 = xgb.Booster(model_file='xgb.model')
        # plotting

        from matplotlib.axes import Axes
        from graphviz import Digraph

        ax = xgb.plot_importance(bst2)
        assert isinstance(ax, Axes)
        assert ax.get_title() == 'Feature importance'
        assert ax.get_xlabel() == 'F score'
        assert ax.get_ylabel() == 'Features'
        assert len(ax.patches) == 4

        ax = xgb.plot_importance(bst2, color='r',
                                 title='t', xlabel='x', ylabel='y')
        assert isinstance(ax, Axes)
        assert ax.get_title() == 't'
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        assert len(ax.patches) == 4
        for p in ax.patches:
            assert p.get_facecolor() == (1.0, 0, 0, 1.0) # red

        ax = xgb.plot_importance(bst2, color=['r', 'r', 'b', 'b'],
                                 title=None, xlabel=None, ylabel=None)
        assert isinstance(ax, Axes)
        assert ax.get_title() == ''
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == ''
        assert len(ax.patches) == 4
        assert ax.patches[0].get_facecolor() == (1.0, 0, 0, 1.0) # red
        assert ax.patches[1].get_facecolor() == (1.0, 0, 0, 1.0) # red
        assert ax.patches[2].get_facecolor() == (0, 0, 1.0, 1.0) # blue
        assert ax.patches[3].get_facecolor() == (0, 0, 1.0, 1.0) # blue

        g = xgb.to_graphviz(bst2, num_trees=0)
        assert isinstance(g, Digraph)

        ax = xgb.plot_tree(bst2, num_trees=0)
        assert isinstance(ax, Axes)

    def test_importance_plot_lim(self):
        np.random.seed(1)
        dm = xgb.DMatrix(np.random.randn(100, 100), label=[0, 1]*50)
        bst = xgb.train({}, dm)
        assert len(bst.get_fscore()) == 71
        ax = xgb.plot_importance(bst)
        assert ax.get_xlim() == (0., 11.)
        assert ax.get_ylim() == (-1., 71.)

        ax = xgb.plot_importance(bst, xlim=(0, 5), ylim=(10, 71))
        assert ax.get_xlim() == (0., 5.)
        assert ax.get_ylim() == (10., 71.)

    def test_sklearn_api(self):
        from sklearn import datasets
        from sklearn.cross_validation import train_test_split

        np.random.seed(1)

        iris = datasets.load_iris()
        tr_d, te_d, tr_l, te_l = train_test_split(iris.data, iris.target, train_size=120)

        classifier = xgb.XGBClassifier()
        classifier.fit(tr_d, tr_l)

        preds = classifier.predict(te_d)
        labels = te_l
        err = sum([1 for p, l in zip(preds, labels) if p != l]) / len(te_l)
        # error must be smaller than 10%
        assert err < 0.1

    def test_sklearn_plotting(self):
        from sklearn import datasets
        iris = datasets.load_iris()

        classifier = xgb.XGBClassifier()
        classifier.fit(iris.data, iris.target)

        import matplotlib
        matplotlib.use('Agg')

        from matplotlib.axes import Axes
        from graphviz import Digraph

        ax = xgb.plot_importance(classifier)
        assert isinstance(ax, Axes)
        assert ax.get_title() == 'Feature importance'
        assert ax.get_xlabel() == 'F score'
        assert ax.get_ylabel() == 'Features'
        assert len(ax.patches) == 4

        g = xgb.to_graphviz(classifier, num_trees=0)
        assert isinstance(g, Digraph)

        ax = xgb.plot_tree(classifier, num_trees=0)
        assert isinstance(ax, Axes)
