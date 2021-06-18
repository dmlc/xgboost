import sys
import os
from typing import Type, TypeVar, Any, Dict, List, Tuple
import pytest
import numpy as np
import asyncio
import xgboost
import subprocess
import tempfile
import json
from collections import OrderedDict
from inspect import signature
from hypothesis import given, strategies, settings, note
from hypothesis._settings import duration
from test_gpu_updaters import parameter_strategy

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)

sys.path.append("tests/python")
from test_with_dask import run_empty_dmatrix_reg      # noqa
from test_with_dask import run_empty_dmatrix_auc      # noqa
from test_with_dask import run_auc                    # noqa
from test_with_dask import run_boost_from_prediction  # noqa
from test_with_dask import run_dask_classifier        # noqa
from test_with_dask import run_empty_dmatrix_cls      # noqa
from test_with_dask import _get_client_workers        # noqa
from test_with_dask import generate_array             # noqa
from test_with_dask import kCols as random_cols       # noqa
from test_with_dask import suppress                   # noqa
import testing as tm                                  # noqa


try:
    import dask.dataframe as dd
    from xgboost import dask as dxgb
    import xgboost as xgb
    from dask.distributed import Client
    from dask import array as da
    from dask_cuda import LocalCUDACluster
    import cudf
except ImportError:
    pass


def make_categorical(
    client: Client,
    n_samples: int,
    n_features: int,
    n_categories: int,
    onehot: bool = False,
) -> Tuple[dd.DataFrame, dd.Series]:
    workers = _get_client_workers(client)
    n_workers = len(workers)
    dfs = []

    def pack(**kwargs: Any) -> dd.DataFrame:
        X, y = tm.make_categorical(**kwargs)
        X["label"] = y
        return X

    meta = pack(
        n_samples=1, n_features=n_features, n_categories=n_categories, onehot=False
    )

    for i, worker in enumerate(workers):
        l_n_samples = min(
            n_samples // n_workers, n_samples - i * (n_samples // n_workers)
        )
        future = client.submit(
            pack,
            n_samples=l_n_samples,
            n_features=n_features,
            n_categories=n_categories,
            onehot=False,
            workers=[worker],
        )
        dfs.append(future)

    df = dd.from_delayed(dfs, meta=meta)
    y = df["label"]
    X = df[df.columns.difference(["label"])]

    if onehot:
        return dd.get_dummies(X), y
    return X, y


def run_with_dask_dataframe(DMatrixT: Type, client: Client) -> None:
    import cupy as cp
    cp.cuda.runtime.setDevice(0)
    X, y, _ = generate_array()

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

    predictions = dxgb.predict(client, out, dtrain)
    assert isinstance(predictions.compute(), np.ndarray)

    series_predictions = dxgb.inplace_predict(client, out, X)
    assert isinstance(series_predictions, dd.Series)

    single_node = out['booster'].predict(xgboost.DMatrix(X.compute()))

    cp.testing.assert_allclose(single_node, predictions.compute())
    np.testing.assert_allclose(single_node,
                               series_predictions.compute().to_array())

    predt = dxgb.predict(client, out, X)
    assert isinstance(predt, dd.Series)

    T = TypeVar('T')

    def is_df(part: T) -> T:
        assert isinstance(part, cudf.DataFrame), part
        return part

    predt.map_partitions(
        is_df,
        meta=dd.utils.make_meta({'prediction': 'f4'}))

    cp.testing.assert_allclose(
        predt.values.compute(), single_node)

    # Make sure the output can be integrated back to original dataframe
    X["predict"] = predictions
    X["inplace_predict"] = series_predictions

    has_null = X.isnull().values.any().compute()
    assert bool(has_null) is False


def run_with_dask_array(DMatrixT: Type, client: Client) -> None:
    import cupy as cp
    cp.cuda.runtime.setDevice(0)
    X, y, _ = generate_array()

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


@pytest.mark.skipif(**tm.no_dask_cudf())
def test_categorical(local_cuda_cluster: LocalCUDACluster) -> None:
    with Client(local_cuda_cluster) as client:
        import dask_cudf

        rounds = 10
        X, y = make_categorical(client, 10000, 30, 13)
        X = dask_cudf.from_dask_dataframe(X)

        X_onehot, _ = make_categorical(client, 10000, 30, 13, True)
        X_onehot = dask_cudf.from_dask_dataframe(X_onehot)

        parameters = {"tree_method": "gpu_hist"}

        m = dxgb.DaskDMatrix(client, X_onehot, y, enable_categorical=True)
        by_etl_results = dxgb.train(
            client,
            parameters,
            m,
            num_boost_round=rounds,
            evals=[(m, "Train")],
        )["history"]

        m = dxgb.DaskDMatrix(client, X, y, enable_categorical=True)
        output = dxgb.train(
            client,
            parameters,
            m,
            num_boost_round=rounds,
            evals=[(m, "Train")],
        )
        by_builtin_results = output["history"]

        np.testing.assert_allclose(
            np.array(by_etl_results["Train"]["rmse"]),
            np.array(by_builtin_results["Train"]["rmse"]),
            rtol=1e-3,
        )
        assert tm.non_increasing(by_builtin_results["Train"]["rmse"])

        model = output["booster"]
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "model.json")
            model.save_model(path)
            with open(path, "r") as fd:
                categorical = json.load(fd)

            categories_sizes = np.array(
                categorical["learner"]["gradient_booster"]["model"]["trees"][-1][
                    "categories_sizes"
                ]
            )
            assert categories_sizes.shape[0] != 0
            np.testing.assert_allclose(categories_sizes, 1)


def to_cp(x: Any, DMatrixT: Type) -> Any:
    import cupy
    if isinstance(x, np.ndarray) and \
       DMatrixT is dxgb.DaskDeviceQuantileDMatrix:
        X = cupy.array(x)
    else:
        X = x
    return X


def run_gpu_hist(
    params: Dict,
    num_rounds: int,
    dataset: tm.TestDataset,
    DMatrixT: Type,
    client: Client,
) -> None:
    params["tree_method"] = "gpu_hist"
    params = dataset.set_params(params)
    # It doesn't make sense to distribute a completely
    # empty dataset.
    if dataset.X.shape[0] == 0:
        return

    chunk = 128
    X = to_cp(dataset.X, DMatrixT)
    X = da.from_array(X, chunks=(chunk, dataset.X.shape[1]))
    y = to_cp(dataset.y, DMatrixT)
    y = da.from_array(y, chunks=(chunk,))
    if dataset.w is not None:
        w = to_cp(dataset.w, DMatrixT)
        w = da.from_array(w, chunks=(chunk,))
    else:
        w = None

    if DMatrixT is dxgb.DaskDeviceQuantileDMatrix:
        m = DMatrixT(
            client, data=X, label=y, weight=w, max_bin=params.get("max_bin", 256)
        )
    else:
        m = DMatrixT(client, data=X, label=y, weight=w)
    history = dxgb.train(
        client,
        params=params,
        dtrain=m,
        num_boost_round=num_rounds,
        evals=[(m, "train")],
    )["history"]
    note(history)
    assert tm.non_increasing(history["train"][dataset.metric])


@pytest.mark.skipif(**tm.no_cudf())
def test_boost_from_prediction(local_cuda_cluster: LocalCUDACluster) -> None:
    import cudf
    from sklearn.datasets import load_breast_cancer
    with Client(local_cuda_cluster) as client:
        X_, y_ = load_breast_cancer(return_X_y=True)
        X = dd.from_array(X_, chunksize=100).map_partitions(cudf.from_pandas)
        y = dd.from_array(y_, chunksize=100).map_partitions(cudf.from_pandas)
        run_boost_from_prediction(X, y, "gpu_hist", client)


class TestDistributedGPU:
    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_cudf())
    @pytest.mark.skipif(**tm.no_dask_cudf())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_dask_dataframe(self, local_cuda_cluster: LocalCUDACluster) -> None:
        with Client(local_cuda_cluster) as client:
            run_with_dask_dataframe(dxgb.DaskDMatrix, client)
            run_with_dask_dataframe(dxgb.DaskDeviceQuantileDMatrix, client)

    @given(
        params=parameter_strategy,
        num_rounds=strategies.integers(1, 20),
        dataset=tm.dataset_strategy,
    )
    @settings(deadline=duration(seconds=120), suppress_health_check=suppress)
    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.parametrize(
        "local_cuda_cluster", [{"n_workers": 2}], indirect=["local_cuda_cluster"]
    )
    @pytest.mark.mgpu
    def test_gpu_hist(
        self,
        params: Dict,
        num_rounds: int,
        dataset: tm.TestDataset,
        local_cuda_cluster: LocalCUDACluster,
    ) -> None:
        with Client(local_cuda_cluster) as client:
            run_gpu_hist(params, num_rounds, dataset, dxgb.DaskDMatrix, client)
            run_gpu_hist(
                params, num_rounds, dataset, dxgb.DaskDeviceQuantileDMatrix, client
            )

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_dask_array(self, local_cuda_cluster: LocalCUDACluster) -> None:
        with Client(local_cuda_cluster) as client:
            run_with_dask_array(dxgb.DaskDMatrix, client)
            run_with_dask_array(dxgb.DaskDeviceQuantileDMatrix, client)

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    def test_early_stopping(self, local_cuda_cluster: LocalCUDACluster) -> None:
        from sklearn.datasets import load_breast_cancer
        with Client(local_cuda_cluster) as client:
            X, y = load_breast_cancer(return_X_y=True)
            X, y = da.from_array(X), da.from_array(y)

            m = dxgb.DaskDMatrix(client, X, y)

            valid = dxgb.DaskDMatrix(client, X, y)
            early_stopping_rounds = 5
            booster = dxgb.train(client, {'objective': 'binary:logistic',
                                          'eval_metric': 'error',
                                          'tree_method': 'gpu_hist'}, m,
                                 evals=[(valid, 'Valid')],
                                 num_boost_round=1000,
                                 early_stopping_rounds=early_stopping_rounds)[
                                     'booster']
            assert hasattr(booster, 'best_score')
            dump = booster.get_dump(dump_format='json')
            assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

            valid_X = X
            valid_y = y
            cls = dxgb.DaskXGBClassifier(objective='binary:logistic',
                                         tree_method='gpu_hist',
                                         n_estimators=100)
            cls.client = client
            cls.fit(X, y, early_stopping_rounds=early_stopping_rounds,
                    eval_set=[(valid_X, valid_y)])
            booster = cls.get_booster()
            dump = booster.get_dump(dump_format='json')
            assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    @pytest.mark.skipif(**tm.no_cudf())
    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.parametrize("model", ["boosting"])
    def test_dask_classifier(
        self, model: str, local_cuda_cluster: LocalCUDACluster
    ) -> None:
        import dask_cudf
        with Client(local_cuda_cluster) as client:
            X_, y_, w_ = generate_array(with_weights=True)
            y_ = (y_ * 10).astype(np.int32)
            X = dask_cudf.from_dask_dataframe(dd.from_dask_array(X_))
            y = dask_cudf.from_dask_dataframe(dd.from_dask_array(y_))
            w = dask_cudf.from_dask_dataframe(dd.from_dask_array(w_))
            run_dask_classifier(X, y, w, model, "gpu_hist", client, 10)

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    def test_empty_dmatrix(self, local_cuda_cluster: LocalCUDACluster) -> None:
        with Client(local_cuda_cluster) as client:
            parameters = {'tree_method': 'gpu_hist',
                          'debug_synchronize': True}
            run_empty_dmatrix_reg(client, parameters)
            run_empty_dmatrix_cls(client, parameters)

    def test_empty_dmatrix_auc(self, local_cuda_cluster: LocalCUDACluster) -> None:
        with Client(local_cuda_cluster) as client:
            n_workers = len(_get_client_workers(client))
            run_empty_dmatrix_auc(client, "gpu_hist", n_workers)

    def test_auc(self, local_cuda_cluster: LocalCUDACluster) -> None:
        with Client(local_cuda_cluster) as client:
            run_auc(client, "gpu_hist")

    def test_data_initialization(self, local_cuda_cluster: LocalCUDACluster) -> None:
        with Client(local_cuda_cluster) as client:
            X, y, _ = generate_array()
            fw = da.random.random((random_cols, ))
            fw = fw - fw.min()
            m = dxgb.DaskDMatrix(client, X, y, feature_weights=fw)

            workers = _get_client_workers(client)
            rabit_args = client.sync(dxgb._get_rabit_args, len(workers), client)

            def worker_fn(worker_addr: str, data_ref: Dict) -> None:
                with dxgb.RabitContext(rabit_args):
                    local_dtrain = dxgb._dmatrix_from_list_of_parts(**data_ref)
                    fw_rows = local_dtrain.get_float_info("feature_weights").shape[0]
                    assert fw_rows == local_dtrain.num_col()

            futures = []
            for i in range(len(workers)):
                futures.append(
                    client.submit(
                        worker_fn,
                        workers[i],
                        m._create_fn_args(workers[i]),
                        pure=False,
                        workers=[workers[i]]
                    )
                )
            client.gather(futures)

    def test_interface_consistency(self) -> None:
        sig = OrderedDict(signature(dxgb.DaskDMatrix).parameters)
        del sig["client"]
        ddm_names = list(sig.keys())
        sig = OrderedDict(signature(dxgb.DaskDeviceQuantileDMatrix).parameters)
        del sig["client"]
        del sig["max_bin"]
        ddqdm_names = list(sig.keys())
        assert len(ddm_names) == len(ddqdm_names)

        # between dask
        for i in range(len(ddm_names)):
            assert ddm_names[i] == ddqdm_names[i]

        sig = OrderedDict(signature(xgb.DMatrix).parameters)
        del sig["nthread"]      # no nthread in dask
        dm_names = list(sig.keys())
        sig = OrderedDict(signature(xgb.DeviceQuantileDMatrix).parameters)
        del sig["nthread"]
        del sig["max_bin"]
        dqdm_names = list(sig.keys())

        # between single node
        assert len(dm_names) == len(dqdm_names)
        for i in range(len(dm_names)):
            assert dm_names[i] == dqdm_names[i]

        # ddm <-> dm
        for i in range(len(ddm_names)):
            assert ddm_names[i] == dm_names[i]

        # dqdm <-> ddqdm
        for i in range(len(ddqdm_names)):
            assert ddqdm_names[i] == dqdm_names[i]

        sig = OrderedDict(signature(xgb.XGBRanker.fit).parameters)
        ranker_names = list(sig.keys())
        sig = OrderedDict(signature(xgb.dask.DaskXGBRanker.fit).parameters)
        dranker_names = list(sig.keys())

        for rn, drn in zip(ranker_names, dranker_names):
            assert rn == drn

    def run_quantile(self, name: str, local_cuda_cluster: LocalCUDACluster) -> None:
        if sys.platform.startswith("win"):
            pytest.skip("Skipping dask tests on Windows")

        exe = None
        for possible_path in {'./testxgboost', './build/testxgboost',
                              '../build/testxgboost', '../gpu-build/testxgboost'}:
            if os.path.exists(possible_path):
                exe = possible_path
        assert exe, 'No testxgboost executable found.'
        test = "--gtest_filter=GPUQuantile." + name

        def runit(
            worker_addr: str, rabit_args: List[bytes]
        ) -> subprocess.CompletedProcess:
            port_env = ''
            # setup environment for running the c++ part.
            for arg in rabit_args:
                if arg.decode('utf-8').startswith('DMLC_TRACKER_PORT'):
                    port_env = arg.decode('utf-8')
            port = port_env.split('=')
            env = os.environ.copy()
            env[port[0]] = port[1]
            return subprocess.run([str(exe), test], env=env, stdout=subprocess.PIPE)

        with Client(local_cuda_cluster) as client:
            workers = _get_client_workers(client)
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
    def test_quantile_basic(self, local_cuda_cluster: LocalCUDACluster) -> None:
        self.run_quantile('AllReduceBasic', local_cuda_cluster)

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_dask_cuda())
    @pytest.mark.mgpu
    @pytest.mark.gtest
    def test_quantile_same_on_all_workers(
        self, local_cuda_cluster: LocalCUDACluster
    ) -> None:
        self.run_quantile('SameOnAllWorkers', local_cuda_cluster)


async def run_from_dask_array_asyncio(scheduler_address: str) -> dxgb.TrainReturnT:
    async with Client(scheduler_address, asynchronous=True) as client:
        import cupy as cp
        X, y, _ = generate_array()
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
@pytest.mark.skipif(**tm.no_dask_cuda())
@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.mgpu
def test_with_asyncio(local_cuda_cluster: LocalCUDACluster) -> None:
    with Client(local_cuda_cluster) as client:
        address = client.scheduler.address
        output = asyncio.run(run_from_dask_array_asyncio(address))
        assert isinstance(output['booster'], xgboost.Booster)
        assert isinstance(output['history'], dict)
