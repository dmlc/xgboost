import sys
import unittest
import pytest

import numpy as np
import xgboost as xgb
from hypothesis import given, strategies, assume, settings, note

from xgboost import testing as tm

rng = np.random.RandomState(1994)

shap_parameter_strategy = strategies.fixed_dictionaries(
    {
        "max_depth": strategies.integers(1, 11),
        "max_leaves": strategies.integers(0, 256),
        "num_parallel_tree": strategies.sampled_from([1, 10]),
    }
).filter(lambda x: x["max_depth"] > 0 or x["max_leaves"] > 0)


class TestSYCLPredict(unittest.TestCase):
    def test_predict(self):
        iterations = 10
        np.random.seed(1)
        test_num_rows = [10, 1000, 5000]
        test_num_cols = [10, 50, 500]
        for num_rows in test_num_rows:
            for num_cols in test_num_cols:
                dtrain = xgb.DMatrix(
                    np.random.randn(num_rows, num_cols),
                    label=[0, 1] * int(num_rows / 2),
                )
                dval = xgb.DMatrix(
                    np.random.randn(num_rows, num_cols),
                    label=[0, 1] * int(num_rows / 2),
                )
                dtest = xgb.DMatrix(
                    np.random.randn(num_rows, num_cols),
                    label=[0, 1] * int(num_rows / 2),
                )
                watchlist = [(dtrain, "train"), (dval, "validation")]
                res = {}
                param = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "device": "cpu",
                    "max_depth": 1,
                    "verbosity": 0,
                }
                bst = xgb.train(
                    param, dtrain, iterations, evals=watchlist, evals_result=res
                )
                assert tm.non_increasing(res["train"]["logloss"])
                cpu_pred_train = bst.predict(dtrain, output_margin=True)
                cpu_pred_test = bst.predict(dtest, output_margin=True)
                cpu_pred_val = bst.predict(dval, output_margin=True)

                bst.set_param({"device": "sycl"})
                sycl_pred_train = bst.predict(dtrain, output_margin=True)
                sycl_pred_test = bst.predict(dtest, output_margin=True)
                sycl_pred_val = bst.predict(dval, output_margin=True)

                np.testing.assert_allclose(cpu_pred_train, sycl_pred_train, rtol=1e-6)
                np.testing.assert_allclose(cpu_pred_val, sycl_pred_val, rtol=1e-6)
                np.testing.assert_allclose(cpu_pred_test, sycl_pred_test, rtol=1e-6)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_multi_predict(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        n = 1000
        X, y = make_regression(n, random_state=rng)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        params = {}
        params["tree_method"] = "hist"
        params["device"] = "cpu"

        bst = xgb.train(params, dtrain)
        cpu_predict = bst.predict(dtest)

        bst.set_param({"device": "sycl"})

        predict0 = bst.predict(dtest)
        predict1 = bst.predict(dtest)

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
        params = {
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": -1,
            "verbosity": 0,
            "seed": 123,
        }
        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        cpu_train_score = m.score(X_train, y_train)
        cpu_test_score = m.score(X_test, y_test)

        # Now with sycl_predictor
        params["device"] = "sycl"
        m.set_params(**params)

        sycl_train_score = m.score(X_train, y_train)
        sycl_test_score = m.score(X_test, y_test)

        assert np.allclose(cpu_train_score, sycl_train_score)
        assert np.allclose(cpu_test_score, sycl_test_score)

    @given(
        strategies.integers(1, 10), tm.make_dataset_strategy(), shap_parameter_strategy
    )
    @settings(deadline=None)
    def test_shap(self, num_rounds, dataset, param):
        if dataset.name.endswith("-l1"):  # not supported by the exact tree method
            return
        param.update({"tree_method": "hist", "device": "cpu"})
        param = dataset.set_params(param)
        dmat = dataset.get_dmat()
        bst = xgb.train(param, dmat, num_rounds)
        test_dmat = xgb.DMatrix(dataset.X, dataset.y, dataset.w, dataset.margin)
        bst.set_param({"device": "sycl"})
        shap = bst.predict(test_dmat, pred_contribs=True)
        margin = bst.predict(test_dmat, output_margin=True)
        assume(len(dataset.y) > 0)
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-3, 1e-3)

    @given(
        strategies.integers(1, 10), tm.make_dataset_strategy(), shap_parameter_strategy
    )
    @settings(deadline=None, max_examples=20)
    def test_shap_interactions(self, num_rounds, dataset, param):
        if dataset.name.endswith("-l1"):  # not supported by the exact tree method
            return
        param.update({"tree_method": "hist", "device": "cpu"})
        param = dataset.set_params(param)
        dmat = dataset.get_dmat()
        bst = xgb.train(param, dmat, num_rounds)
        test_dmat = xgb.DMatrix(dataset.X, dataset.y, dataset.w, dataset.margin)
        bst.set_param({"device": "sycl"})
        shap = bst.predict(test_dmat, pred_interactions=True)
        margin = bst.predict(test_dmat, output_margin=True)
        assume(len(dataset.y) > 0)
        assert np.allclose(
            np.sum(shap, axis=(len(shap.shape) - 1, len(shap.shape) - 2)),
            margin,
            1e-3,
            1e-3,
        )
