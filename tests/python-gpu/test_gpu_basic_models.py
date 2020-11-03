import sys
import os
import numpy as np
import xgboost as xgb
import pytest
sys.path.append("tests/python")
# Don't import the test class, otherwise they will run twice.
import test_callback as test_cb  # noqa
rng = np.random.RandomState(1994)


class TestGPUBasicModels:
    cputest = test_cb.TestCallbacks()

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

    @pytest.mark.parametrize('deprecated_callback', [True, False],
                             ids=['old_callback', 'new_callback'])
    @pytest.mark.parametrize('tree_method', ['gpu_hist'])
    def test_eta_decay_gpu_hist(self, tree_method, deprecated_callback):
        self.cputest.test_eta_decay(tree_method, deprecated_callback)

    def test_deterministic_gpu_hist(self):
        kRows = 1000
        kCols = 64
        kClasses = 4
        # Create large values to force rounding.
        X = np.random.randn(kRows, kCols) * 1e4
        y = np.random.randint(0, kClasses, size=kRows) * 1e4

        model_0, model_1 = self.run_cls(X, y, True)
        assert model_0 == model_1

        model_0, model_1 = self.run_cls(X, y, False)
        assert model_0 != model_1
