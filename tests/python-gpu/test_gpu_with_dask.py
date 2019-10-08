import sys
import pytest

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)

try:
    from distributed.utils_test import client, loop, cluster_fixture
    import dask.dataframe as dd
    from xgboost import dask as dxgb
    import cudf
except ImportError:
    client = None
    loop = None
    cluster_fixture = None
    pass

sys.path.append("tests/python")
from test_with_dask import generate_array
import testing as tm


@pytest.mark.skipif(**tm.no_dask())
@pytest.mark.skipif(**tm.no_cudf())
def test_dask_dataframe(client):
    X, y = generate_array()

    X = dd.from_dask_array(X)
    y = dd.from_dask_array(y)

    X = X.map_partitions(cudf.from_pandas)
    y = y.map_partitions(cudf.from_pandas)

    dtrain = dxgb.DaskDMatrix(client, X, y)
    out = dxgb.train(client, {'tree_method': 'gpu_hist'},
                     dtrain=dtrain,
                     evals=[(dtrain, 'X')],
                     num_boost_round=2)

    assert isinstance(out['booster'], dxgb.Booster)
    assert len(out['history']['X']['rmse']) == 2
