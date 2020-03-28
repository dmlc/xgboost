import sys
import pytest
import numpy as np
import unittest
import xgboost

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)

sys.path.append("tests/python")
from test_with_dask import run_empty_dmatrix  # noqa
from test_with_dask import generate_array     # noqa
import testing as tm                          # noqa

try:
    import dask.dataframe as dd
    from xgboost import dask as dxgb
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import cudf
except ImportError:
    pass


class TestDistributedGPU(unittest.TestCase):
    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_cudf())
    @pytest.mark.skipif(**tm.no_dask_cudf())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    def test_dask_dataframe(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                import cupy
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

                predictions = dxgb.predict(client, out, dtrain).compute()
                assert isinstance(predictions, np.ndarray)

                # There's an error with cudf saying `concat_cudf` got an
                # expected argument `ignore_index`.  So the test here is just
                # place holder.

                # series_predictions = dxgb.inplace_predict(client, out, X)
                # assert isinstance(series_predictions, dd.Series)

                single_node = out['booster'].predict(
                    xgboost.DMatrix(X.compute()))
                cupy.testing.assert_allclose(single_node, predictions)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dask_array(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                import cupy
                X, y = generate_array()

                X = X.map_blocks(cupy.asarray)
                y = y.map_blocks(cupy.asarray)
                dtrain = dxgb.DaskDMatrix(client, X, y)
                out = dxgb.train(client, {'tree_method': 'gpu_hist'},
                                 dtrain=dtrain,
                                 evals=[(dtrain, 'X')],
                                 num_boost_round=2)
                from_dmatrix = dxgb.predict(client, out, dtrain).compute()
                inplace_predictions = dxgb.inplace_predict(
                    client, out, X).compute()
                single_node = out['booster'].predict(
                    xgboost.DMatrix(X.compute()))
                np.testing.assert_allclose(single_node, from_dmatrix)
                cupy.testing.assert_allclose(
                    cupy.array(single_node),
                    inplace_predictions)


    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_empty_dmatrix(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                parameters = {'tree_method': 'gpu_hist'}
                run_empty_dmatrix(client, parameters)
