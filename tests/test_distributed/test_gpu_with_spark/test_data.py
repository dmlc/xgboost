import pytest

from xgboost import testing as tm

pytestmark = pytest.mark.skipif(**tm.no_spark())

from ..test_with_spark.test_data import run_dmatrix_ctor


@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.parametrize(
    "is_feature_cols,is_qdm",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_dmatrix_ctor(is_feature_cols: bool, is_qdm: bool) -> None:
    run_dmatrix_ctor(is_feature_cols, is_qdm, on_gpu=True)
