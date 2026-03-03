"""GPU tests for the built-in objective Python interface."""

from xgboost.testing.objective import (
    check_default_metrics,
    check_equivalence,
    check_sklearn_objectives,
    check_train_classification_objectives,
    check_train_positive_objectives,
    check_train_ranking_objectives,
    check_train_regression_objectives,
    check_train_survival_objectives,
)


def test_train_regression() -> None:
    check_train_regression_objectives("cuda")


def test_train_positive() -> None:
    check_train_positive_objectives("cuda")


def test_train_classification() -> None:
    check_train_classification_objectives("cuda")


def test_train_survival() -> None:
    check_train_survival_objectives("cuda")


def test_train_ranking() -> None:
    check_train_ranking_objectives("cuda")


def test_equivalence() -> None:
    check_equivalence("cuda")


def test_default_metrics() -> None:
    check_default_metrics("cuda")


def test_sklearn() -> None:
    check_sklearn_objectives("cuda")
