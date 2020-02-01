# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import unittest
import scipy.sparse
from scipy.sparse import rand

rng = np.random.RandomState(1)

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestDMatrix(unittest.TestCase):
    def test_dmatrix_numpy_init(self):
        data = np.random.randn(5, 5)
        dm = xgb.DMatrix(data)
        assert dm.num_row() == 5
        assert dm.num_col() == 5

        data = np.array([[1, 2], [3, 4]])
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

    def test_csr(self):
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        X = scipy.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        dtrain = xgb.DMatrix(X)
        assert dtrain.num_row() == 3
        assert dtrain.num_col() == 3

    def test_csc(self):
        row = np.array([0, 2, 2, 0, 1, 2])
        col = np.array([0, 0, 1, 2, 2, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        X = scipy.sparse.csc_matrix((data, (row, col)), shape=(3, 3))
        dtrain = xgb.DMatrix(X)
        assert dtrain.num_row() == 3
        assert dtrain.num_col() == 3

    def test_np_view(self):
        # Sliced Float32 array
        y = np.array([12, 34, 56], np.float32)[::2]
        from_view = xgb.DMatrix(np.array([[]]), label=y).get_label()
        from_array = xgb.DMatrix(np.array([[]]), label=y + 0).get_label()
        assert (from_view.shape == from_array.shape)
        assert (from_view == from_array).all()

        # Sliced UInt array
        z = np.array([12, 34, 56], np.uint32)[::2]
        dmat = xgb.DMatrix(np.array([[]]))
        dmat.set_uint_info('group', z)
        from_view = dmat.get_uint_info('group_ptr')
        dmat = xgb.DMatrix(np.array([[]]))
        dmat.set_uint_info('group', z + 0)
        from_array = dmat.get_uint_info('group_ptr')
        assert (from_view.shape == from_array.shape)
        assert (from_view == from_array).all()

    def test_feature_names(self):
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

        assert dm.slice([0, 1]).feature_names == dm.feature_names

        dm.feature_types = 'q'
        assert dm.feature_types == list('qqqqq')

        dm.feature_types = list('qiqiq')
        assert dm.feature_types == list('qiqiq')

        def incorrect_type_set():
            dm.feature_types = list('abcde')

        self.assertRaises(ValueError, incorrect_type_set)

        # reset
        dm.feature_names = None
        self.assertEqual(dm.feature_names, ['f0', 'f1', 'f2', 'f3', 'f4'])
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

            params = {'objective': 'multi:softprob',
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

    def test_get_info(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtrain.get_float_info('label')
        dtrain.get_float_info('weight')
        dtrain.get_float_info('base_margin')
        dtrain.get_uint_info('group_ptr')

    def test_sparse_dmatrix_csr(self):
        nrow = 100
        ncol = 1000
        x = rand(nrow, ncol, density=0.0005, format='csr', random_state=rng)
        assert x.indices.max() < ncol - 1
        x.data[:] = 1
        dtrain = xgb.DMatrix(x, label=np.random.binomial(1, 0.3, nrow))
        assert (dtrain.num_row(), dtrain.num_col()) == (nrow, ncol)
        watchlist = [(dtrain, 'train')]
        param = {'max_depth': 3, 'objective': 'binary:logistic', 'verbosity': 0}
        bst = xgb.train(param, dtrain, 5, watchlist)
        bst.predict(dtrain)

    def test_sparse_dmatrix_csc(self):
        nrow = 1000
        ncol = 100
        x = rand(nrow, ncol, density=0.0005, format='csc', random_state=rng)
        assert x.indices.max() < nrow - 1
        x.data[:] = 1
        dtrain = xgb.DMatrix(x, label=np.random.binomial(1, 0.3, nrow))
        assert (dtrain.num_row(), dtrain.num_col()) == (nrow, ncol)
        watchlist = [(dtrain, 'train')]
        param = {'max_depth': 3, 'objective': 'binary:logistic', 'verbosity': 0}
        bst = xgb.train(param, dtrain, 5, watchlist)
        bst.predict(dtrain)
