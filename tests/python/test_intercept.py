from itertools import product

import pytest

from xgboost.testing.intercept import (
    run_adaptive,
    run_exp_family,
    run_init_estimation,
    run_logistic_degenerate,
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
