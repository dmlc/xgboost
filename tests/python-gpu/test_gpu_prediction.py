from __future__ import print_function

import numpy as np
import unittest
import xgboost as xgb
from nose.plugins.attrib import attr

rng = np.random.RandomState(1994)

@attr('gpu')
class TestGPUPredict(unittest.TestCase):
    def test_predict(self):
        iterations = 1
        np.random.seed(1)
        test_num_rows = [10, 1000, 5000]
        test_num_cols = [10, 50, 500]
        for num_rows in test_num_rows:
            for num_cols in test_num_cols:
                dm = xgb.DMatrix(np.random.randn(num_rows, num_cols), label=[0, 1] * int(num_rows / 2))
                watchlist = [(dm, 'train')]
                res = {}
                param = {
                    "objective": "binary:logistic",
                    "predictor": "gpu_predictor",
                    'eval_metric': 'auc',
                }
                bst = xgb.train(param, dm, iterations, evals=watchlist, evals_result=res)
                assert self.non_decreasing(res["train"]["auc"])
                gpu_pred = bst.predict(dm, output_margin=True)
                bst.set_param({"predictor": "cpu_predictor"})
                cpu_pred = bst.predict(dm, output_margin=True)
                np.testing.assert_allclose(cpu_pred, gpu_pred, rtol=1e-5)

    def non_decreasing(self, L):
        return all((x - y) < 0.001 for x, y in zip(L, L[1:]))
