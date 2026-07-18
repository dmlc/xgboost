import json

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.monotone_constraints import is_correctly_constrained, training_dset

rng = np.random.RandomState(1994)


def non_decreasing(L: np.ndarray) -> bool:
    return all((x - y) < 0.001 for x, y in zip(L, L[1:]))


def non_increasing(L: np.ndarray) -> bool:
    return all((y - x) < 0.001 for x, y in zip(L, L[1:]))


def assert_constraint(constraint: int, tree_method: str) -> None:
    from sklearn.datasets import make_regression

    n = 1000
    X, y = make_regression(n, random_state=rng, n_features=1, n_informative=1)
    dtrain = xgb.DMatrix(X, y)
    param = {}
    param["tree_method"] = tree_method
    param["device"] = "cuda"
    param["monotone_constraints"] = "(" + str(constraint) + ")"
    bst = xgb.train(param, dtrain)
    dpredict = xgb.DMatrix(X[X[:, 0].argsort()])
    pred = bst.predict(dpredict)

    if constraint > 0:
        assert non_decreasing(pred)
    elif constraint < 0:
        assert non_increasing(pred)


@pytest.mark.skipif(**tm.no_sklearn())
def test_gpu_hist_basic() -> None:
    assert_constraint(1, "hist")
    assert_constraint(-1, "hist")


@pytest.mark.skipif(**tm.no_sklearn())
def test_gpu_approx_basic() -> None:
    assert_constraint(1, "approx")
    assert_constraint(-1, "approx")


def test_gpu_hist_depthwise() -> None:
    params = {
        "tree_method": "hist",
        "grow_policy": "depthwise",
        "device": "cuda",
        "monotone_constraints": "(1, -1)",
    }
    model = xgb.train(params, training_dset)
    is_correctly_constrained(model)


def test_gpu_hist_lossguide() -> None:
    params = {
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "device": "cuda",
        "monotone_constraints": "(1, -1)",
    }
    model = xgb.train(params, training_dset)
    is_correctly_constrained(model)


def test_gpu_hist_parent_gain_with_monotonic_constraint() -> None:
    """Test that parent gain uses the node's inherited monotonic bounds."""
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    gradients = np.array([3.0, -2.75, -0.125, 0.875], dtype=np.float32)
    hessians = np.array([1.0, 0.5, 0.25, 0.25], dtype=np.float32)
    dtrain = xgb.DMatrix(X, label=np.zeros(X.shape[0], dtype=np.float32))

    def objective(
        _predt: np.ndarray, _dtrain: xgb.DMatrix
    ) -> tuple[np.ndarray, np.ndarray]:
        return gradients, hessians

    params = {
        "tree_method": "hist",
        "monotone_constraints": "(1, 0, 0)",
        "reg_alpha": 0,
        "reg_lambda": 0,
        "max_delta_step": 0,
        "min_child_weight": 0,
        "max_depth": 3,
        "eta": 1,
        "base_score": 0,
    }

    trees = {}
    for device in ("cpu", "cuda"):
        model = xgb.train(
            {**params, "device": device}, dtrain, num_boost_round=1, obj=objective
        )
        trees[device] = json.loads(
            model.get_dump(dump_format="json", with_stats=True)[0]
        )

    assert trees["cuda"] == trees["cpu"]
    constrained_parent = trees["cuda"]["children"][1]["children"][1]
    assert constrained_parent["split"] == "f2"
    assert constrained_parent["gain"] == pytest.approx(0.25)
