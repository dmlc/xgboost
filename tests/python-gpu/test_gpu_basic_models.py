import sys
import os
import numpy as np
import xgboost as xgb
import pytest
sys.path.append("tests/python")
# Don't import the test class, otherwise they will run twice.
import test_callback as test_cb  # noqa
import test_basic_models as test_bm
rng = np.random.RandomState(1994)


class TestGPUBasicModels:
    cpu_test_cb = test_cb.TestCallbacks()
    cpu_test_bm = test_bm.TestModels()

    def run_cls(self, X, y, deterministic):
        cls = xgb.XGBClassifier(tree_method='gpu_hist',
                                deterministic_histogram=deterministic,
                                single_precision_histogram=True)
        cls.fit(X, y)
        cls.get_booster().save_model('test_deterministic_gpu_hist-0.json')

        cls = xgb.XGBClassifier(tree_method='gpu_hist',
                                deterministic_histogram=deterministic,
                                single_precision_histogram=True)
        cls.fit(X, y)
        cls.get_booster().save_model('test_deterministic_gpu_hist-1.json')

        with open('test_deterministic_gpu_hist-0.json', 'r') as fd:
            model_0 = fd.read()
        with open('test_deterministic_gpu_hist-1.json', 'r') as fd:
            model_1 = fd.read()

        os.remove('test_deterministic_gpu_hist-0.json')
        os.remove('test_deterministic_gpu_hist-1.json')

        return hash(model_0), hash(model_1)

    def test_custom_objective(self):
        self.cpu_test_bm.run_custom_objective("gpu_hist")

    def test_eta_decay_gpu_hist(self):
        self.cpu_test_cb.run_eta_decay('gpu_hist', True)
        self.cpu_test_cb.run_eta_decay('gpu_hist', False)

    def test_deterministic_gpu_hist(self):
        kRows = 1000
        kCols = 64
        kClasses = 4
        # Create large values to force rounding.
        X = np.random.randn(kRows, kCols) * 1e4
        y = np.random.randint(0, kClasses, size=kRows) * 1e4

        model_0, model_1 = self.run_cls(X, y, True)
        assert model_0 == model_1

    def test_invalid_gpu_id(self):
        X = np.random.randn(10, 5) * 1e4
        y = np.random.randint(0, 2, size=10) * 1e4
        # should pass with invalid gpu id
        cls1 = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=9999)
        cls1.fit(X, y)
        # should throw error with fail_on_invalid_gpu_id enabled
        cls2 = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=9999, fail_on_invalid_gpu_id=True)
        try:
            cls2.fit(X, y)
            assert False, "Should have failed with with fail_on_invalid_gpu_id enabled"
        except xgb.core.XGBoostError as err:
            assert "gpu_id 9999 is invalid" in str(err)
