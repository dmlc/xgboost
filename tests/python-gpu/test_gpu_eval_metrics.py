import json

import pytest

import xgboost
from xgboost import testing as tm
from xgboost.testing.metrics import (
    check_precision_score,
    check_quantile_error,
    run_pr_auc_binary,
    run_pr_auc_ltr,
    run_pr_auc_multi,
    run_roc_auc_binary,
    run_roc_auc_multi,
)


class TestGPUEvalMetrics:
    @pytest.mark.parametrize("n_samples", [4, 100, 1000])
    def test_roc_auc_binary(self, n_samples: int) -> None:
        run_roc_auc_binary("hist", n_samples, "cuda")

    @pytest.mark.parametrize(
        "n_samples,weighted", [(4, False), (100, False), (1000, False), (1000, True)]
    )
    def test_roc_auc_multi(self, n_samples: int, weighted: bool) -> None:
        run_roc_auc_multi("hist", n_samples, weighted, "cuda")

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
        booster.set_param({"device": "cuda:0"})
        assert (
            json.loads(booster.save_config())["learner"]["generic_param"]["device"]
            == "cuda:0"
        )
        gpu_auc = float(booster.eval(Xy).split(":")[1])
        assert (
            json.loads(booster.save_config())["learner"]["generic_param"]["device"]
            == "cuda:0"
        )

        np.testing.assert_allclose(cpu_auc, gpu_auc)

    def test_pr_auc_binary(self) -> None:
        run_pr_auc_binary("hist", "cuda")

    def test_pr_auc_multi(self) -> None:
        run_pr_auc_multi("hist", "cuda")

    def test_pr_auc_ltr(self) -> None:
        run_pr_auc_ltr("hist", "cuda")

    def test_precision_score(self) -> None:
        check_precision_score("hist", "cuda")

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_quantile_error(self) -> None:
        check_quantile_error("hist", "cuda")
