from typing import Callable

import pytest
from xgboost.objective import _BuiltInObjective
from xgboost.testing.objective import (
    all_objective_checks,
    check_equivalence,
    equivalence_parameters,
)
from xgboost.testing.utils import Device


@pytest.mark.parametrize("obj_chk", all_objective_checks())
def test_objectives(obj_chk: Callable[[Device], None]) -> None:
    obj_chk("cuda")


@pytest.mark.parametrize("obj_inst,str_params,dm_factory", equivalence_parameters())
def test_equivalence(
    obj_inst: _BuiltInObjective, str_params: dict, dm_factory: Callable
) -> None:
    check_equivalence("cuda", obj_inst, str_params, dm_factory)
