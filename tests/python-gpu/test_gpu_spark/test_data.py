import sys
from typing import List

import numpy as np
import pandas as pd
import pytest

sys.path.append("tests/python")

import testing as tm

if tm.no_spark()["condition"]:
    pytest.skip(msg=tm.no_spark()["reason"], allow_module_level=True)
if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
    pytest.skip("Skipping PySpark tests on Windows", allow_module_level=True)


from test_spark.test_data import run_dmatrix_ctor


@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.parametrize(
    "is_feature_cols,is_qdm",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_dmatrix_ctor(is_feature_cols: bool, is_qdm: bool) -> None:
    run_dmatrix_ctor(is_feature_cols, is_qdm, on_gpu=True)
