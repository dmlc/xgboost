from typing import TYPE_CHECKING

from xgboost.testing.objective import check_objectives

if TYPE_CHECKING:
    from pytest import Subtests


def test_objectives(subtests: "Subtests") -> None:
    check_objectives(subtests, "cuda")
