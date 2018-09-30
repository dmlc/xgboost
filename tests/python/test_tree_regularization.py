import numpy as np
import unittest
import xgboost as xgb

from numpy.testing import assert_approx_equal

train_data = xgb.DMatrix(np.array([[1]]), label=np.array([1]))


class TestTreeRegularization(unittest.TestCase):
    def test_alpha(self):
        params = {
            'tree_method': 'exact', 'silent': 1, 'objective': 'reg:linear',
            'eta': 1,
            'lambda': 0,
            'alpha': 0.1
        }

        model = xgb.train(params, train_data, 1)
        preds = model.predict(train_data)

        # Default prediction (with no trees) is 0.5
        # sum_grad = (0.5 - 1.0)
        # sum_hess = 1.0
        # 0.9 = 0.5 - (sum_grad - alpha * sgn(sum_grad)) / sum_hess
        assert_approx_equal(preds[0], 0.9)

    def test_lambda(self):
        params = {
            'tree_method': 'exact', 'silent': 1, 'objective': 'reg:linear',
            'eta': 1,
            'lambda': 1,
            'alpha': 0
        }

        model = xgb.train(params, train_data, 1)
        preds = model.predict(train_data)

        # Default prediction (with no trees) is 0.5
        # sum_grad = (0.5 - 1.0)
        # sum_hess = 1.0
        # 0.75 = 0.5 - sum_grad / (sum_hess + lambda)
        assert_approx_equal(preds[0], 0.75)

    def test_alpha_and_lambda(self):
        params = {
            'tree_method': 'exact', 'silent': 1, 'objective': 'reg:linear',
            'eta': 1,
            'lambda': 1,
            'alpha': 0.1
        }

        model = xgb.train(params, train_data, 1)
        preds = model.predict(train_data)

        # Default prediction (with no trees) is 0.5
        # sum_grad = (0.5 - 1.0)
        # sum_hess = 1.0
        # 0.7 = 0.5 - (sum_grad - alpha * sgn(sum_grad)) / (sum_hess + lambda)
        assert_approx_equal(preds[0], 0.7)
