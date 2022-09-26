import numpy as np
from numpy.testing import assert_approx_equal

import xgboost as xgb

train_data = xgb.DMatrix(np.array([[1]]), label=np.array([1]))


class TestTreeRegularization:
    def test_alpha(self):
        params = {
            "tree_method": "exact",
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eta": 1,
            "lambda": 0,
            "alpha": 0.1,
            "base_score": 0.5,
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
            "tree_method": "exact",
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eta": 1,
            "lambda": 1,
            "alpha": 0,
            "base_score": 0.5,
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
            "tree_method": "exact",
            "verbosity": 1,
            "objective": "reg:squarederror",
            "eta": 1,
            "lambda": 1,
            "alpha": 0.1,
            "base_score": 0.5,
        }

        model = xgb.train(params, train_data, 1)
        preds = model.predict(train_data)

        # Default prediction (with no trees) is 0.5
        # sum_grad = (0.5 - 1.0)
        # sum_hess = 1.0
        # 0.7 = 0.5 - (sum_grad - alpha * sgn(sum_grad)) / (sum_hess + lambda)
        assert_approx_equal(preds[0], 0.7)

    def test_unlimited_depth(self):
        x = np.array([[0], [1], [2], [3]])
        y = np.array([0, 1, 2, 3])

        model = xgb.XGBRegressor(
            n_estimators=1,
            eta=1,
            tree_method="hist",
            grow_policy="lossguide",
            reg_lambda=0,
            max_leaves=128,
            max_depth=0,
        ).fit(x, y)
        assert np.array_equal(model.predict(x), y)
