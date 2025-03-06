import pytest

from xgboost import testing as tm
from xgboost.testing.ordinal import (
    run_cat_container,
    run_cat_container_iter,
    run_cat_container_mixed,
)

pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_arrow(), tm.no_pandas()))


def test_cat_container() -> None:
    run_cat_container("cpu")


def test_cat_container_mixed() -> None:
    run_cat_container_mixed("cpu")


def test_cat_container_iter() -> None:
    run_cat_container_iter("cpu")
