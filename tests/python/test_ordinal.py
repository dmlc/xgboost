import pytest

from xgboost import testing as tm
from xgboost.testing.ordinal import (
    run_cat_container,
    run_cat_container_iter,
    run_cat_container_mixed,
    run_cat_invalid,
    run_cat_predict,
    run_cat_shap,
    run_cat_thread_safety,
)

pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_arrow(), tm.no_pandas()))


def test_cat_container() -> None:
    run_cat_container("cpu")


def test_cat_container_mixed() -> None:
    run_cat_container_mixed("cpu")


def test_cat_container_iter() -> None:
    run_cat_container_iter("cpu")


def test_cat_predict() -> None:
    run_cat_predict("cpu")


def test_cat_invalid() -> None:
    run_cat_invalid("cpu")


def test_cat_thread_safety() -> None:
    run_cat_thread_safety("cpu")


def test_cat_shap() -> None:
    run_cat_shap("cpu")
