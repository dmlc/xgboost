import json
import sys

import pytest

import xgboost
from xgboost import testing as tm
from xgboost.testing.metrics import check_quantile_error

sys.path.append("tests/python")
import test_eval_metrics as test_em  # noqa


class TestGPUEvalMetrics:
    cpu_test = test_em.TestEvalMetrics()

    @pytest.mark.parametrize("n_samples", [4, 100, 1000])
    def test_roc_auc_binary(self, n_samples):
        self.cpu_test.run_roc_auc_binary("gpu_hist", n_samples)

    @pytest.mark.parametrize(
        "n_samples,weighted", [(4, False), (100, False), (1000, False), (1000, True)]
    )
    def test_roc_auc_multi(self, n_samples, weighted):
        self.cpu_test.run_roc_auc_multi("gpu_hist", n_samples, weighted)

    @pytest.mark.parametrize("n_samples", [4, 100, 1000])
    def test_roc_auc_ltr(self, n_samples):
        import numpy as np

        rng = np.random.RandomState(1994)
        n_samples = n_samples
        n_features = 10
        X = rng.randn(n_samples, n_features)
        y = rng.randint(0, 16, size=n_samples)
        group = np.array([n_samples // 2, n_samples // 2])

        Xy = xgboost.DMatrix(X, y, group=group)

        booster = xgboost.train(
            {"tree_method": "hist", "eval_metric": "auc", "objective": "rank:ndcg"},
            Xy,
            num_boost_round=10,
        )
        cpu_auc = float(booster.eval(Xy).split(":")[1])
        booster.set_param({"gpu_id": "0"})
        assert json.loads(booster.save_config())["learner"]["generic_param"]["gpu_id"] == "0"
        gpu_auc = float(booster.eval(Xy).split(":")[1])
        assert json.loads(booster.save_config())["learner"]["generic_param"]["gpu_id"] == "0"

        np.testing.assert_allclose(cpu_auc, gpu_auc)

    def test_pr_auc_binary(self):
        self.cpu_test.run_pr_auc_binary("gpu_hist")

    def test_pr_auc_multi(self):
        self.cpu_test.run_pr_auc_multi("gpu_hist")

    def test_pr_auc_ltr(self):
        self.cpu_test.run_pr_auc_ltr("gpu_hist")

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_quantile_error(self) -> None:
        check_quantile_error("gpu_hist")
