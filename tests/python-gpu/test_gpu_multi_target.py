from xgboost import config_context
from xgboost.testing.multi_target import (
    run_multiclass,
    run_multilabel,
    run_reduced_grad,
    run_with_iter,
    run_eta,
)


def test_multiclass() -> None:
    # learning_rate is not yet supported.
    run_multiclass("cuda", 1.0)


def test_multilabel() -> None:
    # learning_rate is not yet supported.
    run_multilabel("cuda", 1.0)


def test_reduced_grad() -> None:
    run_reduced_grad("cuda")


def test_with_iter() -> None:
    with config_context(use_rmm=True):
        run_with_iter("cuda")


def test_eta() -> None:
    run_eta("cuda")
