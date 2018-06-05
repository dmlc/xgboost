import numpy as np
import unittest
import xgboost as xgb
import testing as tm

try:
    from sklearn.datasets import load_boston
except ImportError:
    None


class TestOptimizers(unittest.TestCase):
    def test_momentum(self):
        tm._skip_if_no_sklearn()
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

    def test_nesterov(self):
        tm._skip_if_no_sklearn()
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
        param["optimizer"] = "nesterov_optimizer"
        xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res)
        nesterov = res['train']['rmse']

        np.testing.assert_array_less(nesterov[1: -1], baseline[1:-1])
        assert nesterov[- 1] < momentum[- 1]
