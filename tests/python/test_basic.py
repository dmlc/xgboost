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
from pathlib import Path

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


@contextmanager
def captured_output():
    """
    Reassign stdout temporarily in order to test printed statements
    Taken from: https://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python

    Also works for pytest.
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
        param = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                 'objective': 'binary:logistic'}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
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
        param = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                 'objective': 'binary:logistic'}
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
        param = {'max_depth': 2, 'eta': 1, 'verbosity': 0, 'num_class': 2}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds))
                  if preds[i] != labels[i]) / float(len(preds))
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


    def test_cv(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic'}

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_no_shuffle(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic'}

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, shuffle=False, nfold=10,
                    as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_explicit_fold_indices(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0, 'objective':
                  'binary:logistic'}
        folds = [
            # Train        Test
            ([1, 3], [5, 8]),
            ([7, 9], [23, 43]),
        ]

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, folds=folds,
                    as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_explicit_fold_indices_labels(self):
        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0, 'objective':
                  'reg:squarederror'}
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
        solution = ('[array([5., 8.], dtype=float32), array([23., 43., 11.],' +
                    ' dtype=float32)]')
        assert output == solution


class TestBasicPathLike(unittest.TestCase):
    """Unit tests using the os_fspath and pathlib.Path for file interaction."""

    def test_DMatrix_init_from_path(self):
        """Initialization from the data path."""
        dpath = Path('demo/data')
        dtrain = xgb.DMatrix(dpath / 'agaricus.txt.train')
        assert dtrain.num_row() == 6513
        assert dtrain.num_col() == 127


    def test_DMatrix_save_to_path(self):
        """Saving to a binary file using pathlib from a DMatrix."""
        data = np.random.randn(100, 2)
        target = np.array([0, 1] * 50)
        features = ['Feature1', 'Feature2']

        dm = xgb.DMatrix(data, label=target, feature_names=features)

        # save, assert exists, remove file
        binary_path = Path("dtrain.bin")
        dm.save_binary(binary_path)
        assert binary_path.exists()
        Path.unlink(binary_path)


    def test_Booster_init_invalid_path(self):
        """An invalid model_file path should raise XGBoostError."""
        self.assertRaises(xgb.core.XGBoostError, xgb.Booster,
                          model_file=Path("invalidpath"))


    def test_Booster_save_and_load(self):
        """Saving and loading model files from paths."""
        save_path = Path("saveload.model")

        data = np.random.randn(100, 2)
        target = np.array([0, 1] * 50)
        features = ['Feature1', 'Feature2']

        dm = xgb.DMatrix(data, label=target, feature_names=features)
        params = {'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'eta': 0.3,
                  'max_depth': 1}

        bst = xgb.train(params, dm, num_boost_round=1)

        # save, assert exists
        bst.save_model(save_path)
        assert save_path.exists()

        def dump_assertions(dump):
            """Assertions for the expected dump from Booster"""
            assert len(dump) == 1, 'Exepcted only 1 tree to be dumped.'
            assert len(dump[0].splitlines()) == 3, 'Expected 1 root and 2 leaves - 3 lines.'

        # load the model again using Path
        bst2 = xgb.Booster(model_file=save_path)
        dump2 = bst2.get_dump()
        dump_assertions(dump2)

        # load again using load_model
        bst3 = xgb.Booster()
        bst3.load_model(save_path)
        dump3= bst3.get_dump()
        dump_assertions(dump3)

        # remove file
        Path.unlink(save_path)

    def test_os_fspath(self):
        """Core properties of the os_fspath function."""
        # strings are returned unmodified
        assert '' == xgb.compat.os_fspath('')
        assert '/this/path' == xgb.compat.os_fspath('/this/path')

        # bytes are returned unmodified
        assert b'/this/path' == xgb.compat.os_fspath(b'/this/path')

        # path objects are returned as string representation
        path_test = Path('this') / 'path'
        assert str(path_test) == xgb.compat.os_fspath(path_test)

        # invalid values raise Type error
        self.assertRaises(TypeError, xgb.compat.os_fspath, 123)
