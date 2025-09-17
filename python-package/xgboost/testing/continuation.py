"""Tests for training continuation."""

import json
from typing import Any, Dict, TypeVar

import numpy as np
import pytest

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
