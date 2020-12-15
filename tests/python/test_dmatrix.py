# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import scipy.sparse
import pytest
from scipy.sparse import rand, csr_matrix

rng = np.random.RandomState(1)

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestDMatrix:
    def test_warn_missing(self):
        from xgboost import data
        with pytest.warns(UserWarning):
            data._warn_unused_missing('uri', 4)

        with pytest.warns(None) as record:
            data._warn_unused_missing('uri', None)
            data._warn_unused_missing('uri', np.nan)

            assert len(record) == 0

        with pytest.warns(None) as record:
            x = rng.randn(10, 10)
            y = rng.randn(10)

            xgb.DMatrix(x, y, missing=4)

            assert len(record) == 0

        with pytest.warns(UserWarning):
            csr = csr_matrix(x)
            xgb.DMatrix(csr, y, missing=4)

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
        with pytest.raises(ValueError):
            xgb.DMatrix(np.array(1))
        # 1d array
        with pytest.raises(ValueError):
            xgb.DMatrix(np.array([1, 2, 3]))
        # 3d array
        data = np.random.randn(5, 5, 5)
        with pytest.raises(ValueError):
            xgb.DMatrix(data)
        # object dtype
        data = np.array([['a', 'b'], ['c', 'd']])
        with pytest.raises(ValueError):
            xgb.DMatrix(data)

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

    def test_coo(self):
        row = np.array([0, 2, 2, 0, 1, 2])
        col = np.array([0, 0, 1, 2, 2, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        X = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))
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

    def test_slice(self):
        X = rng.randn(100, 100)
        y = rng.randint(low=0, high=3, size=100).astype(np.float32)
        d = xgb.DMatrix(X, y)
        np.testing.assert_equal(d.get_label(), y)

        fw = rng.uniform(size=100).astype(np.float32)
        d.set_info(feature_weights=fw)

        # base margin is per-class in multi-class classifier
        base_margin = rng.randn(100, 3).astype(np.float32)
        d.set_base_margin(base_margin.flatten())

        ridxs = [1, 2, 3, 4, 5, 6]
        sliced = d.slice(ridxs)

        # Slicing works with label and other meta info fields
        np.testing.assert_equal(sliced.get_label(), y[1:7])
        np.testing.assert_equal(sliced.get_float_info('feature_weights'), fw)
        np.testing.assert_equal(sliced.get_base_margin(), base_margin[1:7, :].flatten())
        np.testing.assert_equal(sliced.get_base_margin(), sliced.get_float_info('base_margin'))

        # Slicing a DMatrix results into a DMatrix that's equivalent to a DMatrix that's
        # constructed from the corresponding NumPy slice
        d2 = xgb.DMatrix(X[1:7, :], y[1:7])
        d2.set_base_margin(base_margin[1:7, :].flatten())
        eval_res = {}
        _ = xgb.train(
            {'num_class': 3, 'objective': 'multi:softprob',
             'eval_metric': 'mlogloss'},
            d,
            num_boost_round=2, evals=[(d2, 'd2'), (sliced, 'sliced')], evals_result=eval_res)
        np.testing.assert_equal(eval_res['d2']['mlogloss'], eval_res['sliced']['mlogloss'])

        ridxs_arr = np.array(ridxs)[1:]  # handles numpy slice correctly
        sliced = d.slice(ridxs_arr)
        np.testing.assert_equal(sliced.get_label(), y[2:7])

    def test_feature_names_slice(self):
        data = np.random.randn(5, 5)

        # different length
        with pytest.raises(ValueError):
            xgb.DMatrix(data, feature_names=list('abcdef'))
        # contains duplicates
        with pytest.raises(ValueError):
            xgb.DMatrix(data, feature_names=['a', 'b', 'c', 'd', 'd'])
        # contains symbol
        with pytest.raises(ValueError):
            xgb.DMatrix(data, feature_names=['a', 'b', 'c', 'd', 'e<1'])

        dm = xgb.DMatrix(data)
        dm.feature_names = list('abcde')
        assert dm.feature_names == list('abcde')

        assert dm.slice([0, 1]).num_col() == dm.num_col()
        assert dm.slice([0, 1]).feature_names == dm.feature_names

        dm.feature_types = 'q'
        assert dm.feature_types == list('qqqqq')

        dm.feature_types = list('qiqiq')
        assert dm.feature_types == list('qiqiq')

        with pytest.raises(ValueError):
            dm.feature_types = list('abcde')

        # reset
        dm.feature_names = None
        assert dm.feature_names == ['f0', 'f1', 'f2', 'f3', 'f4']
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
            with pytest.raises(ValueError):
                bst.predict(dm)

    def test_get_info(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtrain.get_float_info('label')
        dtrain.get_float_info('weight')
        dtrain.get_float_info('base_margin')
        dtrain.get_uint_info('group_ptr')

    def test_feature_weights(self):
        kRows = 10
        kCols = 50
        rng = np.random.RandomState(1994)
        fw = rng.uniform(size=kCols)
        X = rng.randn(kRows, kCols)
        m = xgb.DMatrix(X)
        m.set_info(feature_weights=fw)
        np.testing.assert_allclose(fw, m.get_float_info('feature_weights'))
        # Handle empty
        m.set_info(feature_weights=np.empty((0, 0)))

        assert m.get_float_info('feature_weights').shape[0] == 0

        fw -= 1

        with pytest.raises(ValueError):
            m.set_info(feature_weights=fw)

    def test_sparse_dmatrix_csr(self):
        nrow = 100
        ncol = 1000
        x = rand(nrow, ncol, density=0.0005, format='csr', random_state=rng)
        assert x.indices.max() < ncol - 1
        x.data[:] = 1
        dtrain = xgb.DMatrix(x, label=rng.binomial(1, 0.3, nrow))
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
        dtrain = xgb.DMatrix(x, label=rng.binomial(1, 0.3, nrow))
        assert (dtrain.num_row(), dtrain.num_col()) == (nrow, ncol)
        watchlist = [(dtrain, 'train')]
        param = {'max_depth': 3, 'objective': 'binary:logistic', 'verbosity': 0}
        bst = xgb.train(param, dtrain, 5, watchlist)
        bst.predict(dtrain)
