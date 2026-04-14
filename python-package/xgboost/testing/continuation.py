"""Tests for training continuation."""

import json
from typing import Any, Dict, TypeVar

import numpy as np
import pytest
from hypothesis import strategies

import xgboost as xgb


# pylint: disable=too-many-locals
def run_training_continuation_model_output(device: str, tree_method: str) -> None:
    """Run training continuation test."""
    datasets = pytest.importorskip("sklearn.datasets")
    n_samples = 64
    n_features = 32
    X, y = datasets.make_regression(n_samples, n_features, random_state=1)

    dtrain = xgb.DMatrix(X, y)
    params = {
        "tree_method": tree_method,
        "max_depth": "2",
        "gamma": "0.1",
        "alpha": "0.01",
        "device": device,
    }
    bst_0 = xgb.train(params, dtrain, num_boost_round=64)
    dump_0 = bst_0.get_dump(dump_format="json")

    bst_1 = xgb.train(params, dtrain, num_boost_round=32)
    bst_1 = xgb.train(params, dtrain, num_boost_round=32, xgb_model=bst_1)
    dump_1 = bst_1.get_dump(dump_format="json")

    T = TypeVar("T", Dict[str, Any], float, str, int, list)

    def recursive_compare(obj_0: T, obj_1: T) -> None:
        if isinstance(obj_0, float):
            assert np.isclose(obj_0, obj_1, atol=1e-6)
        elif isinstance(obj_0, str):
            assert obj_0 == obj_1
        elif isinstance(obj_0, int):
            assert obj_0 == obj_1
        elif isinstance(obj_0, dict):
            for i in range(len(obj_0.items())):
                assert list(obj_0.keys())[i] == list(obj_1.keys())[i]
                if list(obj_0.keys())[i] != "missing":
                    recursive_compare(list(obj_0.values()), list(obj_1.values()))
        else:
            for i, lhs in enumerate(obj_0):
                rhs = obj_1[i]
                recursive_compare(lhs, rhs)

    assert len(dump_0) == len(dump_1)

    for i, lhs in enumerate(dump_0):
        obj_0 = json.loads(lhs)
        obj_1 = json.loads(dump_1[i])
        recursive_compare(obj_0, obj_1)


# pylint: disable=too-many-arguments
def run_training_continuation_determinism(
    *,
    device: str,
    tree_method: str,
    subsample: float,
    sampling_method: str,
    rate_drop: float,
    colsample_bylevel: float,
    num_class: int,
    seed_per_iteration: bool,
) -> None:
    """Check that 2-session training (4+4 iters) equals single-session (8 iters)."""
    datasets = pytest.importorskip("sklearn.datasets")

    n_samples = 128
    n_features = 16
    total_rounds = 8
    split_at = 4

    if num_class > 1:
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=6,
            n_classes=num_class,
            random_state=42,
        )
        objective = "multi:softprob"
    else:
        X, y = datasets.make_regression(
            n_samples=n_samples, n_features=n_features, random_state=42
        )
        objective = "reg:squarederror"

    dtrain = xgb.DMatrix(X, y)

    params: Dict[str, Any] = {
        "device": device,
        "tree_method": tree_method,
        "max_depth": 4,
        "objective": objective,
        "subsample": subsample,
        "sampling_method": sampling_method,
        "rate_drop": rate_drop,
        "colsample_bylevel": colsample_bylevel,
        "seed_per_iteration": seed_per_iteration,
    }
    if num_class > 1:
        params["num_class"] = num_class

    bst_single = xgb.train(params, dtrain, num_boost_round=total_rounds)

    bst_first = xgb.train(params, dtrain, num_boost_round=split_at)
    bst_continued = xgb.train(
        params, dtrain, num_boost_round=total_rounds - split_at, xgb_model=bst_first
    )

    config_single = json.loads(bst_single.save_config())
    config_cont = json.loads(bst_continued.save_config())

    rng_single = config_single["learner"]["generic_param"]["rng_state"]
    rng_cont = config_cont["learner"]["generic_param"]["rng_state"]
    assert rng_single == rng_cont, "RNG states diverged between single and continued."

    pred_single = bst_single.predict(dtrain)
    pred_cont = bst_continued.predict(dtrain)
    np.testing.assert_allclose(pred_single, pred_cont)


def make_determinism_strategy(tree_methods: list[str]) -> "strategies.SearchStrategy":
    """Hypothesis strategy for testing training continuation with sampling."""
    strategy = strategies.fixed_dictionaries(
        {
            "subsample": strategies.sampled_from([0.5, 1.0]),
            "sampling_method": strategies.sampled_from(["uniform", "gradient_based"]),
            "rate_drop": strategies.sampled_from([0.0, 0.5]),
            "colsample_bylevel": strategies.sampled_from([0.5, 1.0]),
            "tree_method": strategies.sampled_from(tree_methods),
            "num_class": strategies.sampled_from([1, 3]),
            "seed_per_iteration": strategies.booleans(),
        }
    ).filter(
        lambda x: (
            not (
                x["sampling_method"] == "gradient_based" and x["tree_method"] != "hist"
            )
        )
    )
    return strategy
