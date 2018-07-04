import numpy as np
import unittest
import xgboost as xgb

from numpy.testing import assert_approx_equal

train_data = xgb.DMatrix(np.array([[1]]), label=np.array([1]))


class TestElasticNet(unittest.TestCase):
    def test_lasso(self):
        params = {
            'tree_method': 'exact', 'silent': 1, 'objective': 'reg:linear',
            'eta': 1,
            'lambda': 0,
            'alpha': 0.1
        }

        model = xgb.train(params, train_data, 1)
        preds = model.predict(train_data)

        assert_approx_equal(preds[0], 0.9)

    def test_ridge(self):
        params = {
            'tree_method': 'exact', 'silent': 1, 'objective': 'reg:linear',
            'eta': 1,
            'lambda': 1,
            'alpha': 0
        }

        model = xgb.train(params, train_data, 1)
        preds = model.predict(train_data)

        assert_approx_equal(preds[0], 0.75)

    def test_elastic_net(self):
        params = {
            'tree_method': 'exact', 'silent': 1, 'objective': 'reg:linear',
            'eta': 1,
            'lambda': 1,
            'alpha': 0.1
        }

        model = xgb.train(params, train_data, 1)
        preds = model.predict(train_data)

        assert_approx_equal(preds[0], 0.7)
