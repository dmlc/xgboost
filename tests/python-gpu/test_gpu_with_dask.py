import sys
import os
import pytest
import numpy as np
import asyncio
import unittest
import xgboost
import subprocess
from hypothesis import given, strategies, settings, note
from hypothesis._settings import duration
from test_gpu_updaters import parameter_strategy

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)

sys.path.append("tests/python")
from test_with_dask import run_empty_dmatrix_reg  # noqa
from test_with_dask import run_empty_dmatrix_cls  # noqa
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


def run_with_dask_dataframe(DMatrixT, client):
    import cupy as cp
    cp.cuda.runtime.setDevice(0)
    X, y = generate_array()

    X = dd.from_dask_array(X)
    y = dd.from_dask_array(y)

    X = X.map_partitions(cudf.from_pandas)
    y = y.map_partitions(cudf.from_pandas)

    dtrain = DMatrixT(client, X, y)
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


def run_with_dask_array(DMatrixT, client):
    import cupy as cp
    cp.cuda.runtime.setDevice(0)
    X, y = generate_array()

    X = X.map_blocks(cp.asarray)
    y = y.map_blocks(cp.asarray)
    dtrain = DMatrixT(client, X, y)
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


def to_cp(x, DMatrixT):
    import cupy
    if isinstance(x, np.ndarray) and \
       DMatrixT is dxgb.DaskDeviceQuantileDMatrix:
        X = cupy.array(x)
    else:
        X = x
    return X


def run_gpu_hist(params, num_rounds, dataset, DMatrixT, client):
    params['tree_method'] = 'gpu_hist'
    params = dataset.set_params(params)
    # It doesn't make sense to distribute a completely
    # empty dataset.
    if dataset.X.shape[0] == 0:
        return

    chunk = 128
    X = to_cp(dataset.X, DMatrixT)
    X = da.from_array(X,
                      chunks=(chunk, dataset.X.shape[1]))
    y = to_cp(dataset.y, DMatrixT)
    y = da.from_array(y, chunks=(chunk, ))
    if dataset.w is not None:
        w = to_cp(dataset.w, DMatrixT)
        w = da.from_array(w, chunks=(chunk, ))
    else:
        w = None

    if DMatrixT is dxgb.DaskDeviceQuantileDMatrix:
        m = DMatrixT(client, data=X, label=y, weight=w,
                     max_bin=params.get('max_bin', 256))
    else:
        m = DMatrixT(client, data=X, label=y, weight=w)
    history = dxgb.train(client, params=params, dtrain=m,
                         num_boost_round=num_rounds,
                         evals=[(m, 'train')])['history']
    note(history)
    assert tm.non_increasing(history['train'][dataset.metric])


class TestDistributedGPU(unittest.TestCase):
    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_cudf())
    @pytest.mark.skipif(**tm.no_dask_cudf())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_dask_dataframe(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                run_with_dask_dataframe(dxgb.DaskDMatrix, client)
                run_with_dask_dataframe(dxgb.DaskDeviceQuantileDMatrix, client)

    @given(parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=duration(seconds=120))
    @pytest.mark.mgpu
    def test_gpu_hist(self, params, num_rounds, dataset):
        with LocalCUDACluster(n_workers=2) as cluster:
            with Client(cluster) as client:
                run_gpu_hist(params, num_rounds, dataset, dxgb.DaskDMatrix,
                             client)
                run_gpu_hist(params, num_rounds, dataset,
                             dxgb.DaskDeviceQuantileDMatrix, client)

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.mgpu
    def test_dask_array(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                run_with_dask_array(dxgb.DaskDMatrix, client)
                run_with_dask_array(dxgb.DaskDeviceQuantileDMatrix, client)

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_empty_dmatrix(self):
        with LocalCUDACluster() as cluster:
            with Client(cluster) as client:
                parameters = {'tree_method': 'gpu_hist',
                              'debug_synchronize': True}
                run_empty_dmatrix_reg(client, parameters)
                run_empty_dmatrix_cls(client, parameters)

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
                rabit_args = client.sync(dxgb._get_rabit_args, workers, client)
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
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    @pytest.mark.gtest
    def test_quantile_basic(self):
        self.run_quantile('AllReduceBasic')

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    @pytest.mark.gtest
    def test_quantile_same_on_all_workers(self):
        self.run_quantile('SameOnAllWorkers')


async def run_from_dask_array_asyncio(scheduler_address):
    async with Client(scheduler_address, asynchronous=True) as client:
        import cupy as cp
        X, y = generate_array()
        X = X.map_blocks(cp.array)
        y = y.map_blocks(cp.array)

        m = await xgboost.dask.DaskDeviceQuantileDMatrix(client, X, y)
        output = await xgboost.dask.train(client, {'tree_method': 'gpu_hist'},
                                          dtrain=m)

        with_m = await xgboost.dask.predict(client, output, m)
        with_X = await xgboost.dask.predict(client, output, X)
        inplace = await xgboost.dask.inplace_predict(client, output, X)
        assert isinstance(with_m, da.Array)
        assert isinstance(with_X, da.Array)
        assert isinstance(inplace, da.Array)

        cp.testing.assert_allclose(await client.compute(with_m),
                                   await client.compute(with_X))
        cp.testing.assert_allclose(await client.compute(with_m),
                                   await client.compute(inplace))

        client.shutdown()
        return output


@pytest.mark.skipif(**tm.no_dask())
@pytest.mark.mgpu
def test_with_asyncio():
    with LocalCUDACluster() as cluster:
        with Client(cluster) as client:
            address = client.scheduler.address
            output = asyncio.run(run_from_dask_array_asyncio(address))
            assert isinstance(output['booster'], xgboost.Booster)
            assert isinstance(output['history'], dict)
