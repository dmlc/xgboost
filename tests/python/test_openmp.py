# -*- coding: utf-8 -*-
from scipy.sparse import csr_matrix
import xgboost as xgb
import unittest


class TestOMP(unittest.TestCase):
    def test_omp(self):
        # a contrived example where one node has an instance set of size 2.
        data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        indices = [2, 1, 1, 2, 0, 0, 2, 0, 1, 3]
        indptr = [0, 1, 2, 4, 5, 7, 9, 10]
        A = csr_matrix((data, indices, indptr), shape=(7, 4))
        y = [1, 1, 0, 0, 0, 1, 1]
        dtrain = xgb.DMatrix(A, label=y)

        # 1. use 3 threads to train a tree with an instance set of size 2
        param = {'booster': 'gbtree',
                 'objective': 'binary:logistic',
                 'grow_policy': 'lossguide',
                 'tree_method': 'hist',
                 'eval_metric': 'auc',
                 'max_depth': 0,
                 'max_leaves': 1024,
                 'min_child_weight': 0,
                 'nthread': 3}

        watchlist = [(dtrain, 'train')]
        num_round = 1
        res = {}
        xgb.train(param, dtrain, num_round, watchlist, evals_result=res)
        assert res['train']['auc'][-1] > 0.99

        # 2. vary number of threads and test whether you get the same result
        param['nthread'] = 1
        res2 = {}
        xgb.train(param, dtrain, num_round, watchlist, evals_result=res2)
        assert res['train']['auc'][-1] == res2['train']['auc'][-1]

        param['nthread'] = 2
        res3 = {}
        xgb.train(param, dtrain, num_round, watchlist, evals_result=res3)
        assert res['train']['auc'][-1] == res3['train']['auc'][-1]
