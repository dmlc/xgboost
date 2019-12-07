import sys
import pytest
import numpy as np
import unittest

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)

try:
    import dask.dataframe as dd
    from xgboost import dask as dxgb
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import cudf
except ImportError:
    pass

sys.path.append("tests/python")
from test_with_dask import generate_array  # noqa
import testing as tm                       # noqa


class TestDistributedGPU(unittest.TestCase):
    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_cudf())
    @pytest.mark.skipif(**tm.no_dask_cudf())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    def test_dask_dataframe(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
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

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_empty_dmatrix(self):

        def _check_outputs(out, predictions):
            assert isinstance(out['booster'], dxgb.Booster)
            assert len(out['history']['validation']['rmse']) == 2
            assert isinstance(predictions, np.ndarray)
            assert predictions.shape[0] == 1

        parameters = {'tree_method': 'gpu_hist', 'verbosity': 3,
                      'debug_synchronize': True}

        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                kRows, kCols = 1, 97
                X = dd.from_array(np.random.randn(kRows, kCols))
                y = dd.from_array(np.random.rand(kRows))
                dtrain = dxgb.DaskDMatrix(client, X, y)

                out = dxgb.train(client, parameters,
                                 dtrain=dtrain,
                                 evals=[(dtrain, 'validation')],
                                 num_boost_round=2)
                predictions = dxgb.predict(client=client, model=out,
                                           data=dtrain).compute()
                _check_outputs(out, predictions)

                # train has more rows than evals
                valid = dtrain
                kRows += 1
                X = dd.from_array(np.random.randn(kRows, kCols))
                y = dd.from_array(np.random.rand(kRows))
                dtrain = dxgb.DaskDMatrix(client, X, y)

                out = dxgb.train(client, parameters,
                                 dtrain=dtrain,
                                 evals=[(valid, 'validation')],
                                 num_boost_round=2)
                predictions = dxgb.predict(client=client, model=out,
                                           data=valid).compute()
                _check_outputs(out, predictions)
