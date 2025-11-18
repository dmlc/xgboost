from xgboost.testing.multi_target import (
    run_multiclass,
    run_multilabel,
    run_reduced_grad,
    run_with_iter,
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
    run_with_iter("cuda")
