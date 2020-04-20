'''Tests for running inplace prediction.'''
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import sparse

import xgboost as xgb


def run_threaded_predict(X, rows, predict_func):
    results = []
    per_thread = 20
    with ThreadPoolExecutor(max_workers=10) as e:
        for i in range(0, rows, int(rows / per_thread)):
            if hasattr(X, 'iloc'):
                predictor = X.iloc[i:i+per_thread, :]
            else:
                predictor = X[i:i+per_thread, ...]
            f = e.submit(predict_func, predictor)
            results.append(f)

    for f in results:
        assert f.result()


class TestInplacePredict(unittest.TestCase):
    '''Tests for running inplace prediction'''
    def test_predict(self):
        rows = 1000
        cols = 10

        np.random.seed(1994)

        X = np.random.randn(rows, cols)
        y = np.random.randn(rows)
        dtrain = xgb.DMatrix(X, y)

        booster = xgb.train({'tree_method': 'hist'},
                            dtrain, num_boost_round=10)

        test = xgb.DMatrix(X[:10, ...])
        predt_from_array = booster.inplace_predict(X[:10, ...])
        predt_from_dmatrix = booster.predict(test)

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        def predict_dense(x):
            inplace_predt = booster.inplace_predict(x)
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, rows, predict_dense)

        def predict_csr(x):
            inplace_predt = booster.inplace_predict(sparse.csr_matrix(x))
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, rows, predict_csr)
