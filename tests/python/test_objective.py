"""CPU tests for the built-in objective Python interface."""

from xgboost.testing.objective import (
    check_builtin_objective_base,
    check_default_metrics,
    check_equivalence,
    check_sklearn_objectives,
    check_train_aft_objective,
    check_train_classification_objectives,
    check_train_positive_objectives,
    check_train_ranking_objectives,
    check_train_regression_objectives,
)


def test_base() -> None:
    check_builtin_objective_base()


def test_train_regression() -> None:
    check_train_regression_objectives("cpu")


def test_train_positive() -> None:
    check_train_positive_objectives("cpu")


def test_train_classification() -> None:
    check_train_classification_objectives("cpu")


def test_train_aft() -> None:
    check_train_aft_objective("cpu")


def test_train_ranking() -> None:
    check_train_ranking_objectives("cpu")


def test_equivalence() -> None:
    check_equivalence("cpu")


def test_default_metrics() -> None:
    check_default_metrics("cpu")


def test_sklearn() -> None:
    check_sklearn_objectives("cpu")
