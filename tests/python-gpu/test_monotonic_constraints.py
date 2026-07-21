import numpy as np
import pytest
import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.monotone_constraints import (
    is_correctly_constrained,
    is_decreasing,
    is_increasing,
    run_parent_gain,
    training_dset,
)

rng = np.random.RandomState(1994)


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
        assert is_increasing(pred)
    elif constraint < 0:
        assert is_decreasing(pred)


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


@pytest.mark.parametrize("multi_strategy", ["one_output_per_tree", "multi_output_tree"])
def test_parent_gain(multi_strategy: str) -> None:
    run_parent_gain("cuda", multi_strategy)
