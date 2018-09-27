from __future__ import print_function

import numpy as np
import sys
import unittest
import xgboost as xgb
from nose.plugins.attrib import attr

rng = np.random.RandomState(1994)


@attr('gpu')
class TestGPUPredict(unittest.TestCase):
    def test_predict(self):
        iterations = 10
        np.random.seed(1)
        test_num_rows = [10, 1000, 5000]
        test_num_cols = [10, 50, 500]
        for num_rows in test_num_rows:
            for num_cols in test_num_cols:
                dtrain = xgb.DMatrix(np.random.randn(num_rows, num_cols), label=[0, 1] * int(num_rows / 2))
                dval = xgb.DMatrix(np.random.randn(num_rows, num_cols), label=[0, 1] * int(num_rows / 2))
                dtest = xgb.DMatrix(np.random.randn(num_rows, num_cols), label=[0, 1] * int(num_rows / 2))
                watchlist = [(dtrain, 'train'), (dval, 'validation')]
                res = {}
                param = {
                    "objective": "binary:logistic",
                    "predictor": "gpu_predictor",
                    'eval_metric': 'auc',
                }
                bst = xgb.train(param, dtrain, iterations, evals=watchlist, evals_result=res)
                assert self.non_decreasing(res["train"]["auc"])
                gpu_pred_train = bst.predict(dtrain, output_margin=True)
                gpu_pred_test = bst.predict(dtest, output_margin=True)
                gpu_pred_val = bst.predict(dval, output_margin=True)

                param["predictor"] = "cpu_predictor"
                bst_cpu = xgb.train(param, dtrain, iterations, evals=watchlist)
                cpu_pred_train = bst_cpu.predict(dtrain, output_margin=True)
                cpu_pred_test = bst_cpu.predict(dtest, output_margin=True)
                cpu_pred_val = bst_cpu.predict(dval, output_margin=True)
                np.testing.assert_allclose(cpu_pred_train, gpu_pred_train, rtol=1e-5)
                np.testing.assert_allclose(cpu_pred_val, gpu_pred_val, rtol=1e-5)
                np.testing.assert_allclose(cpu_pred_test, gpu_pred_test, rtol=1e-5)

    def non_decreasing(self, L):
        return all((x - y) < 0.001 for x, y in zip(L, L[1:]))

    # Test case for a bug where multiple batch predictions made on a test set produce incorrect results
    def test_multi_predict(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        n = 1000
        X, y = make_regression(n, random_state=rng)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        params = {}
        params["tree_method"] = "gpu_hist"

        params['predictor'] = "gpu_predictor"
        bst_gpu_predict = xgb.train(params, dtrain)

        params['predictor'] = "cpu_predictor"
        bst_cpu_predict = xgb.train(params, dtrain)

        predict0 = bst_gpu_predict.predict(dtest)
        predict1 = bst_gpu_predict.predict(dtest)
        cpu_predict = bst_cpu_predict.predict(dtest)

        assert np.allclose(predict0, predict1)
        assert np.allclose(predict0, cpu_predict)

    def test_sklearn(self):
        m, n = 15000, 14
        tr_size = 2500
        X = np.random.rand(m, n)
        y = 200 * np.matmul(X, np.arange(-3, -3 + n))
        X_train, y_train = X[:tr_size, :], y[:tr_size]
        X_test, y_test = X[tr_size:, :], y[tr_size:]

        # First with cpu_predictor
        params = {'tree_method': 'gpu_hist',
                  'predictor': 'cpu_predictor',
                  'n_jobs': -1,
                  'seed': 123
                  }
        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        cpu_train_score = m.score(X_train, y_train)
        cpu_test_score = m.score(X_test, y_test)

        # Now with gpu_predictor
        params['predictor'] = 'gpu_predictor'

        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        gpu_train_score = m.score(X_train, y_train)
        gpu_test_score = m.score(X_test, y_test)

        assert np.allclose(cpu_train_score, gpu_train_score)
        assert np.allclose(cpu_test_score, gpu_test_score)
