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
import pytest
import testing as tm

# This is the select basic tests from test_basic.py with modifications for os.PathLike
# instead of string file paths throughout.
dpath = 'demo/data/'

try:
    from pathlib import Path
    dpath = Path(dpath)
except ImportError:
    pass

pytestmark = pytest.mark.skipif(**tm.no_pathlike())

rng = np.random.RandomState(1994)

class TestBasicPathLike(unittest.TestCase):

    def test_basic_PathLike(self):
        dtrain = xgb.DMatrix(dpath / 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath / 'agaricus.txt.test')
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
        dtest.save_binary(Path('dtest.buffer'))
        # save model
        bst.save_model(Path('xgb.model'))
        # load model and data in
        bst2 = xgb.Booster(model_file=Path('xgb.model'))
        dtest2 = xgb.DMatrix(Path('dtest.buffer'))
        preds2 = bst2.predict(dtest2)
        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0

    def test_record_results_PathLike(self):
        dtrain = xgb.DMatrix(dpath / 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath / 'agaricus.txt.test')
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

    def test_multiclass_PathLike(self):
        dtrain = xgb.DMatrix(dpath / 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath / 'agaricus.txt.test')
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
        dtest.save_binary(Path('dtest.buffer'))
        # save model
        bst.save_model(Path('xgb.model'))
        # load model and data in
        bst2 = xgb.Booster(model_file=Path('xgb.model'))
        dtest2 = xgb.DMatrix(Path('dtest.buffer'))
        preds2 = bst2.predict(dtest2)
        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0


    def test_load_file_invalid_PathLike(self):
        self.assertRaises(xgb.core.XGBoostError, xgb.Booster,
                          model_file=Path('incorrect_path'))

    def test_get_info_PathLike(self):
        dtrain = xgb.DMatrix(dpath / 'agaricus.txt.train')
        dtrain.get_float_info('label')
        dtrain.get_float_info('weight')
        dtrain.get_float_info('base_margin')
        dtrain.get_uint_info('root_index')
