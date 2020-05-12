import sys
import unittest
import pytest

import numpy as np
import xgboost as xgb

sys.path.append("tests/python")
import testing as tm
from test_predict import run_threaded_predict  # noqa

rng = np.random.RandomState(1994)


class TestGPUPredict(unittest.TestCase):
    def test_predict(self):
        iterations = 10
        np.random.seed(1)
        test_num_rows = [10, 1000, 5000]
        test_num_cols = [10, 50, 500]
        # This test passes for tree_method=gpu_hist and tree_method=exact. but
        # for `hist` and `approx` the floating point error accumulates faster
        # and fails even tol is set to 1e-4.  For `hist`, the mismatching rate
        # with 5000 rows is 0.04.
        for num_rows in test_num_rows:
            for num_cols in test_num_cols:
                dtrain = xgb.DMatrix(np.random.randn(num_rows, num_cols),
                                     label=[0, 1] * int(num_rows / 2))
                dval = xgb.DMatrix(np.random.randn(num_rows, num_cols),
                                   label=[0, 1] * int(num_rows / 2))
                dtest = xgb.DMatrix(np.random.randn(num_rows, num_cols),
                                    label=[0, 1] * int(num_rows / 2))
                watchlist = [(dtrain, 'train'), (dval, 'validation')]
                res = {}
                param = {
                    "objective": "binary:logistic",
                    "predictor": "gpu_predictor",
                    'eval_metric': 'logloss',
                    'tree_method': 'gpu_hist',
                    'max_depth': 1
                }
                bst = xgb.train(param, dtrain, iterations, evals=watchlist,
                                evals_result=res)
                assert self.non_increasing(res["train"]["logloss"])
                gpu_pred_train = bst.predict(dtrain, output_margin=True)
                gpu_pred_test = bst.predict(dtest, output_margin=True)
                gpu_pred_val = bst.predict(dval, output_margin=True)

                param["predictor"] = "cpu_predictor"
                bst_cpu = xgb.train(param, dtrain, iterations, evals=watchlist)
                cpu_pred_train = bst_cpu.predict(dtrain, output_margin=True)
                cpu_pred_test = bst_cpu.predict(dtest, output_margin=True)
                cpu_pred_val = bst_cpu.predict(dval, output_margin=True)

                np.testing.assert_allclose(cpu_pred_train, gpu_pred_train,
                                           rtol=1e-6)
                np.testing.assert_allclose(cpu_pred_val, gpu_pred_val,
                                           rtol=1e-6)
                np.testing.assert_allclose(cpu_pred_test, gpu_pred_test,
                                           rtol=1e-6)

    def non_increasing(self, L):
        return all((y - x) < 0.001 for x, y in zip(L, L[1:]))

    # Test case for a bug where multiple batch predictions made on a
    # test set produce incorrect results
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_multi_predict(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        n = 1000
        X, y = make_regression(n, random_state=rng)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=123)
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

    @pytest.mark.skipif(**tm.no_sklearn())
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
                  'seed': 123}
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

    @pytest.mark.skipif(**tm.no_cupy())
    def test_inplace_predict_cupy(self):
        import cupy as cp
        cp.cuda.runtime.setDevice(0)
        rows = 1000
        cols = 10
        cp_rng = cp.random.RandomState(1994)
        cp.random.set_random_state(cp_rng)
        X = cp.random.randn(rows, cols)
        y = cp.random.randn(rows)

        dtrain = xgb.DMatrix(X, y)

        booster = xgb.train({'tree_method': 'gpu_hist'},
                            dtrain, num_boost_round=10)
        test = xgb.DMatrix(X[:10, ...])
        predt_from_array = booster.inplace_predict(X[:10, ...])
        predt_from_dmatrix = booster.predict(test)

        cp.testing.assert_allclose(predt_from_array, predt_from_dmatrix)

        def predict_dense(x):
            inplace_predt = booster.inplace_predict(x)
            d = xgb.DMatrix(x)
            copied_predt = cp.array(booster.predict(d))
            return cp.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, rows, predict_dense)

    @pytest.mark.skipif(**tm.no_cudf())
    def test_inplace_predict_cudf(self):
        import cupy as cp
        import cudf
        import pandas as pd
        rows = 1000
        cols = 10
        rng = np.random.RandomState(1994)
        X = rng.randn(rows, cols)
        X = pd.DataFrame(X)
        y = rng.randn(rows)

        X = cudf.from_pandas(X)

        dtrain = xgb.DMatrix(X, y)

        booster = xgb.train({'tree_method': 'gpu_hist'},
                            dtrain, num_boost_round=10)
        test = xgb.DMatrix(X)
        predt_from_array = booster.inplace_predict(X)
        predt_from_dmatrix = booster.predict(test)

        cp.testing.assert_allclose(predt_from_array, predt_from_dmatrix)

        def predict_df(x):
            inplace_predt = booster.inplace_predict(x)
            d = xgb.DMatrix(x)
            copied_predt = cp.array(booster.predict(d))
            return cp.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, rows, predict_df)
