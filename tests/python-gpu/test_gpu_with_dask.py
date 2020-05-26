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
    @pytest.mark.mgpu
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
                                 num_boost_round=4)

                assert isinstance(out['booster'], dxgb.Booster)
                assert len(out['history']['X']['rmse']) == 4

                predictions = dxgb.predict(client, out, dtrain).compute()
                assert isinstance(predictions, np.ndarray)

                series_predictions = dxgb.inplace_predict(client, out, X)
                assert isinstance(series_predictions, dd.Series)
                series_predictions = series_predictions.compute()

                single_node = out['booster'].predict(
                    xgboost.DMatrix(X.compute()))

                cupy.testing.assert_allclose(single_node, predictions)
                cupy.testing.assert_allclose(single_node, series_predictions)

                predt = dxgb.predict(client, out, X)
                assert isinstance(predt, dd.Series)

                def is_df(part):
                    assert isinstance(part, cudf.DataFrame), part
                    return part

                predt.map_partitions(
                    is_df,
                    meta=dd.utils.make_meta({'prediction': 'f4'}))

                cupy.testing.assert_allclose(
                    predt.values.compute(), single_node)

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.mgpu
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
                device = cupy.cuda.runtime.getDevice()
                assert device == inplace_predictions.device.id
                single_node = cupy.array(single_node)
                assert device == single_node.device.id
                cupy.testing.assert_allclose(
                    single_node,
                    inplace_predictions)


    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_empty_dmatrix(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                parameters = {'tree_method': 'gpu_hist'}
                run_empty_dmatrix(client, parameters)
