from typing import Optional

import pytest

from xgboost import config_context
from xgboost.testing.multi_target import (
    run_column_sampling,
    run_deterministic,
    run_eta,
    run_multiclass,
    run_multilabel,
    run_reduced_grad,
    run_with_iter,
)


@pytest.mark.parametrize("learning_rate", [1.0, None])
def test_multiclass(learning_rate: Optional[float]) -> None:
    run_multiclass("cuda", learning_rate)


@pytest.mark.parametrize("learning_rate", [1.0, None])
def test_multilabel(learning_rate: Optional[float]) -> None:
    run_multilabel("cuda", learning_rate)


def test_reduced_grad() -> None:
    run_reduced_grad("cuda")


def test_with_iter() -> None:
    with config_context(use_rmm=True):
        run_with_iter("cuda")


def test_eta() -> None:
    run_eta("cuda")


def test_deterministic() -> None:
    run_deterministic("cuda")


def test_column_sampling() -> None:
    run_column_sampling("cuda")
