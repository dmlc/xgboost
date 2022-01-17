import sys
import unittest
import pytest

import numpy as np
import xgboost as xgb
from hypothesis import given, strategies, assume, settings, note

sys.path.append("tests/python")
import testing as tm

rng = np.random.RandomState(1994)

shap_parameter_strategy = strategies.fixed_dictionaries({
    'max_depth': strategies.integers(0, 11),
    'max_leaves': strategies.integers(0, 256),
    'num_parallel_tree': strategies.sampled_from([1, 10]),
}).filter(lambda x: x['max_depth'] > 0 or x['max_leaves'] > 0)


class TestOneAPIPredict(unittest.TestCase):
    def test_predict(self):
        iterations = 10
        np.random.seed(1)
        test_num_rows = [10, 1000, 5000]
        test_num_cols = [10, 50, 500]
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
                    "objective": "binary:logistic_oneapi",
                    "predictor": "oneapi_predictor",
                    'eval_metric': 'logloss',
                    'tree_method': 'hist',
                    'updater': 'grow_quantile_histmaker_oneapi',
                    'max_depth': 1
                }
                bst = xgb.train(param, dtrain, iterations, evals=watchlist,
                                evals_result=res)
                assert self.non_increasing(res["train"]["logloss"])
                oneapi_pred_train = bst.predict(dtrain, output_margin=True)
                oneapi_pred_test = bst.predict(dtest, output_margin=True)
                oneapi_pred_val = bst.predict(dval, output_margin=True)

                param["predictor"] = "cpu_predictor"
                bst_cpu = xgb.train(param, dtrain, iterations, evals=watchlist)
                cpu_pred_train = bst_cpu.predict(dtrain, output_margin=True)
                cpu_pred_test = bst_cpu.predict(dtest, output_margin=True)
                cpu_pred_val = bst_cpu.predict(dval, output_margin=True)

                np.testing.assert_allclose(cpu_pred_train, oneapi_pred_train,
                                           rtol=1e-6)
                np.testing.assert_allclose(cpu_pred_val, oneapi_pred_val,
                                           rtol=1e-6)
                np.testing.assert_allclose(cpu_pred_test, oneapi_pred_test,
                                           rtol=1e-6)

    def non_increasing(self, L):
        return all((y - x) < 0.001 for x, y in zip(L, L[1:]))

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
        params["tree_method"] = "hist"
        params["updater"] = "grow_quantile_histmaker_oneapi"

        params['predictor'] = "oneapi_predictor"
        bst_oneapi_predict = xgb.train(params, dtrain)

        params['predictor'] = "cpu_predictor"
        bst_cpu_predict = xgb.train(params, dtrain)

        predict0 = bst_oneapi_predict.predict(dtest)
        predict1 = bst_oneapi_predict.predict(dtest)
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
        params = {'tree_method': 'hist',
                  'predictor': 'cpu_predictor',
                  'n_jobs': -1,
                  'seed': 123}
        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        cpu_train_score = m.score(X_train, y_train)
        cpu_test_score = m.score(X_test, y_test)

        # Now with oneapi_predictor
        params['predictor'] = 'oneapi_predictor'

        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        oneapi_train_score = m.score(X_train, y_train)
        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        oneapi_test_score = m.score(X_test, y_test)

        assert np.allclose(cpu_train_score, oneapi_train_score)
        assert np.allclose(cpu_test_score, oneapi_test_score)

    @given(strategies.integers(1, 10),
           tm.dataset_strategy.filter(lambda x: x.name != "empty"), shap_parameter_strategy)
    @settings(deadline=None)
    def test_shap(self, num_rounds, dataset, param):
        param.update({"predictor": "oneapi_predictor"})
        param = dataset.set_params(param)
        dmat = dataset.get_dmat()
        bst = xgb.train(param, dmat, num_rounds)
        test_dmat = xgb.DMatrix(dataset.X, dataset.y, dataset.w, dataset.margin)
        shap = bst.predict(test_dmat, pred_contribs=True)
        margin = bst.predict(test_dmat, output_margin=True)
        assume(len(dataset.y) > 0)
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-3, 1e-3)

    @given(strategies.integers(1, 10),
           tm.dataset_strategy.filter(lambda x: x.name != "empty"), shap_parameter_strategy)
    @settings(deadline=None, max_examples=20)
    def test_shap_interactions(self, num_rounds, dataset, param):
        param.update({"predictor": "oneapi_predictor"})
        param = dataset.set_params(param)
        dmat = dataset.get_dmat()
        bst = xgb.train(param, dmat, num_rounds)
        test_dmat = xgb.DMatrix(dataset.X, dataset.y, dataset.w, dataset.margin)
        shap = bst.predict(test_dmat, pred_interactions=True)
        margin = bst.predict(test_dmat, output_margin=True)
        assume(len(dataset.y) > 0)
        assert np.allclose(np.sum(shap, axis=(len(shap.shape) - 1, len(shap.shape) - 2)), margin,
                           1e-3, 1e-3)
