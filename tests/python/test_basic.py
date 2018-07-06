# -*- coding: utf-8 -*-
import sys
from contextlib import contextmanager
try:
    # python 2
    from StringIO import StringIO
except ImportError:
    # python 3
    from io import StringIO
import numpy as np
import xgboost as xgb
import unittest
import json

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


@contextmanager
def captured_output():
    """
    Reassign stdout temporarily in order to test printed statements
    Taken from: https://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


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

    def test_record_results(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        result = {}
        res2 = {}
        xgb.train(param, dtrain, num_round, watchlist,
                  callbacks=[xgb.callback.record_evaluation(result)])
        xgb.train(param, dtrain, num_round, watchlist,
                  evals_result=res2)
        assert result['train']['error'][0] < 0.1
        assert res2 == result

    def test_multiclass(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'num_class': 2}
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

    def test_dump(self):
        data = np.random.randn(100, 2)
        target = np.array([0, 1] * 50)
        features = ['Feature1', 'Feature2']

        dm = xgb.DMatrix(data, label=target, feature_names=features)
        params = {'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'eta': 0.3,
                  'max_depth': 1}

        bst = xgb.train(params, dm, num_boost_round=1)

        # number of feature importances should == number of features
        dump1 = bst.get_dump()
        self.assertEqual(len(dump1), 1, "Expected only 1 tree to be dumped.")
        self.assertEqual(len(dump1[0].splitlines()), 3,
                         "Expected 1 root and 2 leaves - 3 lines in dump.")

        dump2 = bst.get_dump(with_stats=True)
        self.assertEqual(dump2[0].count('\n'), 3,
                         "Expected 1 root and 2 leaves - 3 lines in dump.")
        self.assertGreater(dump2[0].find('\n'), dump1[0].find('\n'),
                           "Expected more info when with_stats=True is given.")

        dump3 = bst.get_dump(dump_format="json")
        dump3j = json.loads(dump3[0])
        self.assertEqual(dump3j["nodeid"], 0, "Expected the root node on top.")

        dump4 = bst.get_dump(dump_format="json", with_stats=True)
        dump4j = json.loads(dump4[0])
        self.assertIn("gain", dump4j, "Expected 'gain' to be dumped in JSON.")

    def test_load_file_invalid(self):
        self.assertRaises(xgb.core.XGBoostError, xgb.Booster,
                          model_file='incorrect_path')

        self.assertRaises(xgb.core.XGBoostError, xgb.Booster,
                          model_file=u'不正なパス')

    def test_dmatrix_numpy_init_omp(self):

        rows = [1000, 11326, 15000]
        cols = 50
        for row in rows:
            X = np.random.randn(row, cols)
            y = np.random.randn(row).astype('f')
            dm = xgb.DMatrix(X, y, nthread=0)
            np.testing.assert_array_equal(dm.get_label(), y)
            assert dm.num_row() == row
            assert dm.num_col() == cols

            dm = xgb.DMatrix(X, y, nthread=10)
            np.testing.assert_array_equal(dm.get_label(), y)
            assert dm.num_row() == row
            assert dm.num_col() == cols

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
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_no_shuffle(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, shuffle=False, nfold=10, as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_explicit_fold_indices(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        folds = [
            # Train        Test
            ([1, 3], [5, 8]),
            ([7, 9], [23, 43]),
        ]

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, folds=folds, as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_explicit_fold_indices_labels(self):
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
        N = 100
        F = 3
        dm = xgb.DMatrix(data=np.random.randn(N, F), label=np.arange(N))
        folds = [
            # Train        Test
            ([1, 3], [5, 8]),
            ([7, 9], [23, 43, 11]),
        ]

        # Use callback to log the test labels in each fold
        def cb(cbackenv):
            print([fold.dtest.get_label() for fold in cbackenv.cvfolds])

        # Run cross validation and capture standard out to test callback result
        with captured_output() as (out, err):
            xgb.cv(
                params, dm, num_boost_round=1, folds=folds, callbacks=[cb],
                as_pandas=False
            )
            output = out.getvalue().strip()
        assert output == '[array([5., 8.], dtype=float32), array([23., 43., 11.], dtype=float32)]'

    def test_get_info(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtrain.get_float_info('label')
        dtrain.get_float_info('weight')
        dtrain.get_float_info('base_margin')
        dtrain.get_uint_info('root_index')
