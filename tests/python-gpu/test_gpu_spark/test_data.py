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
def test_qdm_ctor() -> None:
    run_dmatrix_ctor(True)
