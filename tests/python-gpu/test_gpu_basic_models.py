import sys
import os
import unittest
import numpy as np
import xgboost as xgb
sys.path.append("tests/python")
# Don't import the test class, otherwise they will run twice.
import test_basic_models as test_bm  # noqa
rng = np.random.RandomState(1994)


class TestGPUBasicModels(unittest.TestCase):
    cputest = test_bm.TestModels()

    def test_eta_decay_gpu_hist(self):
        self.cputest.run_eta_decay('gpu_hist')

    def test_deterministic_gpu_hist(self):
        kRows = 1000
        kCols = 64
        kClasses = 4
        # Create large values to force rounding.
        X = np.random.randn(kRows, kCols) * 1e4
        y = np.random.randint(0, kClasses, size=kRows)

        cls = xgb.XGBClassifier(tree_method='gpu_hist',
                                deterministic_histogram=True,
                                single_precision_histogram=True)
        cls.fit(X, y)
        cls.get_booster().save_model('test_deterministic_gpu_hist-0.json')

        cls = xgb.XGBClassifier(tree_method='gpu_hist',
                                deterministic_histogram=True,
                                single_precision_histogram=True)
        cls.fit(X, y)
        cls.get_booster().save_model('test_deterministic_gpu_hist-1.json')

        with open('test_deterministic_gpu_hist-0.json', 'r') as fd:
            model_0 = fd.read()
        with open('test_deterministic_gpu_hist-1.json', 'r') as fd:
            model_1 = fd.read()

        assert hash(model_0) == hash(model_1)

        os.remove('test_deterministic_gpu_hist-0.json')
        os.remove('test_deterministic_gpu_hist-1.json')
