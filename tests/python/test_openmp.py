# -*- coding: utf-8 -*-
from scipy.sparse import csr_matrix
import xgboost as xgb
import unittest
import numpy as np


class TestOMP(unittest.TestCase):
    def test_omp(self):
        dpath = 'demo/data/'
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

        # use loss-guide grow policy to train a tree
        param = {'booster': 'gbtree',
                 'objective': 'binary:logistic',
                 'grow_policy': 'lossguide',
                 'tree_method': 'hist',
                 'eval_metric': 'auc',
                 'max_depth': 0,
                 'max_leaves': 1024,
                 'min_child_weight': 0,
                 'nthread': 3}

        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 5

        def run_trial():
            res = {}
            bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=res)
            auc = [res['train']['auc'][-1], res['eval']['auc'][-1]]
            assert auc[0] > 0.99 and auc[1] > 0.99
            preds = bst.predict(dtest)
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            # error must be smaller than 10%
            assert err < 0.1
            return auc, preds

        auc1, pred1 = run_trial()

        # vary number of threads and test whether you get the same result
        param['nthread'] = 1
        auc2, pred2 = run_trial()
        assert auc1 == auc2
        assert np.array_equal(pred1, pred2)

        param['nthread'] = 2
        auc3, pred3 = run_trial()
        assert auc1 == auc3
        assert np.array_equal(pred1, pred3)

        # use depth-wise grow policy to train a tree
        param.update({
            'grow_policy': 'depthwise',
            'max_depth': 5,
            'max_leaves': 0,
            'nthread': 3
        })
        auc1, pred1 = run_trial()

        # vary number of threads and test whether you get the same result
        param['nthread'] = 1
        auc2, pred2 = run_trial()
        assert auc1 == auc2
        assert np.array_equal(pred1, pred2)
