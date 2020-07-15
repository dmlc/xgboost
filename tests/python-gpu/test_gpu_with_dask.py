import sys
import os
import pytest
import numpy as np
import unittest
import xgboost
import subprocess
from hypothesis import given, strategies, settings, note
from test_gpu_updaters import parameter_strategy

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
    from dask import array as da
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
                import cupy as cp
                cp.cuda.runtime.setDevice(0)
                X, y = generate_array()

                X = dd.from_dask_array(X)
                y = dd.from_dask_array(y)

                X = X.map_partitions(cudf.from_pandas)
                y = y.map_partitions(cudf.from_pandas)

                dtrain = dxgb.DaskDMatrix(client, X, y)
                out = dxgb.train(client, {'tree_method': 'gpu_hist',
                                          'debug_synchronize': True},
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

                cp.testing.assert_allclose(single_node, predictions)
                np.testing.assert_allclose(single_node,
                                           series_predictions.to_array())

                predt = dxgb.predict(client, out, X)
                assert isinstance(predt, dd.Series)

                def is_df(part):
                    assert isinstance(part, cudf.DataFrame), part
                    return part

                predt.map_partitions(
                    is_df,
                    meta=dd.utils.make_meta({'prediction': 'f4'}))

                cp.testing.assert_allclose(
                    predt.values.compute(), single_node)

    @given(parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=None)
    @pytest.mark.mgpu
    def test_gpu_hist(self, params, num_rounds, dataset):
        with LocalCUDACluster(n_workers=2) as cluster:
            with Client(cluster) as client:
                params['tree_method'] = 'gpu_hist'
                params = dataset.set_params(params)
                # multi class doesn't handle empty dataset well (empty
                # means at least 1 worker has data).
                if params['objective'] == "multi:softmax":
                    return
                # It doesn't make sense to distribute a completely
                # empty dataset.
                if dataset.X.shape[0] == 0:
                    return

                chunk = 128
                X = da.from_array(dataset.X,
                                  chunks=(chunk, dataset.X.shape[1]))
                y = da.from_array(dataset.y, chunks=(chunk, ))
                if dataset.w is not None:
                    w = da.from_array(dataset.w, chunks=(chunk, ))
                else:
                    w = None

                m = dxgb.DaskDMatrix(
                    client, data=X, label=y, weight=w)
                history = dxgb.train(client, params=params, dtrain=m,
                                     num_boost_round=num_rounds,
                                     evals=[(m, 'train')])['history']
                note(history)
                assert tm.non_increasing(history['train'][dataset.metric])

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.mgpu
    def test_dask_array(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                import cupy as cp
                cp.cuda.runtime.setDevice(0)
                X, y = generate_array()

                X = X.map_blocks(cp.asarray)
                y = y.map_blocks(cp.asarray)
                dtrain = dxgb.DaskDMatrix(client, X, y)
                out = dxgb.train(client, {'tree_method': 'gpu_hist',
                                          'debug_synchronize': True},
                                 dtrain=dtrain,
                                 evals=[(dtrain, 'X')],
                                 num_boost_round=2)
                from_dmatrix = dxgb.predict(client, out, dtrain).compute()
                inplace_predictions = dxgb.inplace_predict(
                    client, out, X).compute()
                single_node = out['booster'].predict(
                    xgboost.DMatrix(X.compute()))
                np.testing.assert_allclose(single_node, from_dmatrix)
                device = cp.cuda.runtime.getDevice()
                assert device == inplace_predictions.device.id
                single_node = cp.array(single_node)
                assert device == single_node.device.id
                cp.testing.assert_allclose(
                    single_node,
                    inplace_predictions)

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_empty_dmatrix(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                parameters = {'tree_method': 'gpu_hist',
                              'debug_synchronize': True}
                run_empty_dmatrix(client, parameters)

    def run_quantile(self, name):
        if sys.platform.startswith("win"):
            pytest.skip("Skipping dask tests on Windows")

        exe = None
        for possible_path in {'./testxgboost', './build/testxgboost',
                              '../build/testxgboost', '../gpu-build/testxgboost'}:
            if os.path.exists(possible_path):
                exe = possible_path
        assert exe, 'No testxgboost executable found.'
        test = "--gtest_filter=GPUQuantile." + name

        def runit(worker_addr, rabit_args):
            port = None
            # setup environment for running the c++ part.
            for arg in rabit_args:
                if arg.decode('utf-8').startswith('DMLC_TRACKER_PORT'):
                    port = arg.decode('utf-8')
            port = port.split('=')
            env = os.environ.copy()
            env[port[0]] = port[1]
            return subprocess.run([exe, test], env=env, stdout=subprocess.PIPE)

        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                workers = list(dxgb._get_client_workers(client).keys())
                rabit_args = dxgb._get_rabit_args(workers, client)
                futures = client.map(runit,
                                     workers,
                                     pure=False,
                                     workers=workers,
                                     rabit_args=rabit_args)
                results = client.gather(futures)
                for ret in results:
                    msg = ret.stdout.decode('utf-8')
                    assert msg.find('1 test from GPUQuantile') != -1, msg
                    assert ret.returncode == 0, msg

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.mgpu
    @pytest.mark.gtest
    def test_quantile_basic(self):
        self.run_quantile('AllReduceBasic')

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.mgpu
    @pytest.mark.gtest
    def test_quantile_same_on_all_workers(self):
        self.run_quantile('SameOnAllWorkers')
