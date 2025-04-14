import pytest

from xgboost.testing.callbacks import (
    run_eta_decay,
    run_eta_decay_leaf_output,
    tree_methods_objs,
)


@pytest.mark.parametrize("tree_method", ["approx", "hist"])
def test_eta_decay(tree_method: str) -> None:
    run_eta_decay(tree_method, "cuda")


@pytest.mark.parametrize("tree_method,objective", tree_methods_objs())
def test_eta_decay_leaf_output(tree_method: str, objective: str) -> None:
    run_eta_decay_leaf_output(tree_method, objective, "cuda")
