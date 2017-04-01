# -*- coding: utf-8 -*-
from scipy.sparse import csr_matrix
import xgboost as xgb
import unittest
import json

class TestOMP(unittest.TestCase):
    def test_omp(self):
        # a contrived example where one node has an instance set of size 3.
        data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        indices = [2, 1, 1, 2, 0, 0, 2, 0, 1, 3]
        indptr = [0, 1, 2, 4, 5, 7, 9, 10]
        A = csr_matrix((data, indices, indptr), shape=(7,4))
        y = [1, 1, 0, 0, 0, 1, 1]
        dtrain = xgb.DMatrix(A, label=y)
        param = {'booster': 'gbtree',
                 'objective': 'binary:logistic',
                 'grow_policy': 'lossguide',
                 'tree_method': 'hist',
                 'eval_metric': 'auc',
                 'max_depth': 0,
                 'max_leaves': 1024,
                 'min_child_weight': 0}

        watchlist = [(dtrain, 'train')]
        num_round = 1
        res = {}
        bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=res)
        assert res['train']['auc'][-1] > 0.6
