import numpy as np
import unittest
import xgboost as xgb
import sys
from sklearn.datasets import load_boston


class TestOptimizers(unittest.TestCase):
    def test_momentum(self):
        data = load_boston()
        dtrain = xgb.DMatrix(data.data, label=data.target)
        param = {}
        res = {}
        num_rounds = 10
        xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res)
        baseline = res['train']['rmse']
        param["optimizer"] = "momentum_optimizer"
        param["momentum"] = 0.5
        xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res)
        momentum = res['train']['rmse']

        np.testing.assert_array_less(momentum[1: -1], baseline[1:-1])
