from typing import Callable

import pytest
from xgboost.testing.objective import all_objective_checks
from xgboost.testing.utils import Device


@pytest.mark.parametrize("obj_chk", all_objective_checks())
def test_objectives(obj_chk: Callable[[Device], None]) -> None:
    obj_chk("cuda")
