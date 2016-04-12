# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import unittest

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestBasic(unittest.TestCase):
    def test_basic(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
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
        assert np.sum(np.abs(preds2 - preds)) == 0

    def test_multiclass(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'num_class' : 2}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds)) if preds[i] != labels[i]) / float(len(preds))
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
        assert np.sum(np.abs(preds2 - preds)) == 0


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

    def test_load_file_invalid(self):
        self.assertRaises(xgb.core.XGBoostError, xgb.Booster,
                          model_file='incorrect_path')

        self.assertRaises(xgb.core.XGBoostError, xgb.Booster,
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
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=False)
        assert isinstance(cv, np.ndarray)
        assert cv.shape == (10, 4)
