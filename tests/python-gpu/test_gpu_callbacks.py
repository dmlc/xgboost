import pytest

from xgboost import testing as tm
from xgboost.testing.callbacks import (
    run_eta_decay,
    run_eta_decay_leaf_output,
    tree_methods_objs,
)


@pytest.mark.parametrize("tree_method", ["approx", "hist"])
def test_eta_decay(tree_method: str) -> None:
    dtrain, dtest = tm.load_agaricus(__file__)
    run_eta_decay(tree_method, dtrain, dtest, "cuda")


@pytest.mark.parametrize("tree_method,objective", tree_methods_objs())
def test_eta_decay_leaf_output(tree_method: str, objective: str) -> None:
    dtrain, dtest = tm.load_agaricus(__file__)
    run_eta_decay_leaf_output(tree_method, objective, dtrain, dtest, "cuda")
