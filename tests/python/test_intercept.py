from itertools import product

import numpy as np
import pytest

import xgboost as xgb
from xgboost.testing.intercept import (
    run_adaptive,
    run_exp_family,
    run_init_estimation,
    run_logistic_degenerate,
    run_tweedie_degenerate,
)


def test_init_estimation() -> None:
    run_init_estimation("hist", "cpu")


@pytest.mark.parametrize(
    "tree_method,weighted", list(product(["approx", "hist"], [True, False]))
)
def test_adaptive(tree_method: str, weighted: bool) -> None:
    run_adaptive(tree_method, weighted, "cpu")


def test_exp_family() -> None:
    run_exp_family("cpu")


def test_logistic_degenerate() -> None:
    run_logistic_degenerate("cpu")


def test_tweedie_degenerate() -> None:
    run_tweedie_degenerate("cpu")


@pytest.mark.parametrize(
    "objective,base_score",
    [("count:poisson", -1.0), ("survival:cox", 0.0), ("survival:cox", -1.0)],
)
def test_non_positive_base_score(objective: str, base_score: float) -> None:
    # The margin is log(base_score), which used to give NaN predictions.
    X = np.random.default_rng(2026).normal(size=(16, 2))
    dtrain = xgb.DMatrix(X, label=np.arange(1, 17, dtype=np.float32))
    booster = xgb.train({"objective": objective, "base_score": base_score}, dtrain, 2)
    assert np.isfinite(booster.predict(dtrain)).all()
