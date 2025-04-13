import os
import sys

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.basic_models import run_custom_objective

sys.path.append("tests/python")

# Don't import the test class, otherwise they will run twice.
import test_callback as test_cb  # noqa

rng = np.random.RandomState(1994)


class TestGPUBasicModels:
    cpu_test_cb = test_cb.TestCallbacks()

    def run_cls(self, X, y):
        cls = xgb.XGBClassifier(tree_method="hist", device="cuda")
        cls.fit(X, y)
        cls.get_booster().save_model("test_deterministic_gpu_hist-0.json")

        cls = xgb.XGBClassifier(tree_method="hist", device="cuda")
        cls.fit(X, y)
        cls.get_booster().save_model("test_deterministic_gpu_hist-1.json")

        with open("test_deterministic_gpu_hist-0.json", "r") as fd:
            model_0 = fd.read()
        with open("test_deterministic_gpu_hist-1.json", "r") as fd:
            model_1 = fd.read()

        os.remove("test_deterministic_gpu_hist-0.json")
        os.remove("test_deterministic_gpu_hist-1.json")

        return hash(model_0), hash(model_1)

    def test_custom_objective(self) -> None:
        dtrain, dtest = tm.load_agaricus(__file__)
        run_custom_objective("hist", "cuda", dtrain, dtest)

    def test_eta_decay(self) -> None:
        self.cpu_test_cb.run_eta_decay("gpu_hist")

    @pytest.mark.parametrize(
        "objective", ["binary:logistic", "reg:absoluteerror", "reg:quantileerror"]
    )
    def test_eta_decay_leaf_output(self, objective) -> None:
        self.cpu_test_cb.run_eta_decay_leaf_output("gpu_hist", objective)

    def test_deterministic_gpu_hist(self):
        kRows = 1000
        kCols = 64
        kClasses = 4
        # Create large values to force rounding.
        X = np.random.randn(kRows, kCols) * 1e4
        y = np.random.randint(0, kClasses, size=kRows)

        model_0, model_1 = self.run_cls(X, y)
        assert model_0 == model_1

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_invalid_gpu_id(self):
        from sklearn.datasets import load_digits

        X, y = load_digits(return_X_y=True)
        # should pass with invalid gpu id
        cls1 = xgb.XGBClassifier(tree_method="hist", device="cuda:9999")
        cls1.fit(X, y)
        # should throw error with fail_on_invalid_gpu_id enabled
        cls2 = xgb.XGBClassifier(
            tree_method="hist", device="cuda:9999", fail_on_invalid_gpu_id=True
        )
        with pytest.raises(ValueError, match="ordinal 9999 is invalid"):
            cls2.fit(X, y)

        cls2 = xgb.XGBClassifier(
            tree_method="hist", device="cuda:9999", fail_on_invalid_gpu_id=True
        )
        with pytest.raises(ValueError, match="ordinal 9999 is invalid"):
            cls2.fit(X, y)
