import sys

import pandas as pd
import pytest

from xgboost import testing

sys.path.append("tests/python")

import testing as tm

if testing.skip_spark()["condition"]:
    pytest.skip(msg=testing.skip_spark()["reason"], allow_module_level=True)


from test_spark.test_data import run_dmatrix_ctor


@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.parametrize(
    "is_feature_cols,is_qdm",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_dmatrix_ctor(is_feature_cols: bool, is_qdm: bool) -> None:
    run_dmatrix_ctor(is_feature_cols, is_qdm, on_gpu=True)
