"""Copyright 2019-2023, XGBoost contributors"""

import asyncio
import json
from collections import OrderedDict
from inspect import signature
from typing import Any, Dict, Type, TypeVar

import numpy as np
import pytest
from hypothesis import given, note, settings, strategies
from hypothesis._settings import duration
from packaging.version import parse as parse_version

import xgboost as xgb
from xgboost import testing as tm
from xgboost.collective import CommunicatorContext
from xgboost.testing.params import hist_parameter_strategy

pytestmark = [
    pytest.mark.skipif(**tm.no_dask()),
    pytest.mark.skipif(**tm.no_dask_cuda()),
    tm.timeout(120),
]

from ..test_with_dask.test_with_dask import generate_array
from ..test_with_dask.test_with_dask import kCols as random_cols
from ..test_with_dask.test_with_dask import (
    make_categorical,
    run_auc,
    run_boost_from_prediction,
    run_boost_from_prediction_multi_class,
    run_categorical,
    run_dask_classifier,
    run_empty_dmatrix_auc,
    run_empty_dmatrix_cls,
    run_empty_dmatrix_reg,
    run_tree_stats,
    suppress,
)

try:
    import cudf
    import dask.dataframe as dd
    from dask import __version__ as dask_version
    from dask import array as da
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster

    from xgboost import dask as dxgb
    from xgboost.testing.dask import check_init_estimation, check_uneven_nan
except ImportError:
    dask_version = None


dask_version_ge110 = dask_version and parse_version(dask_version) >= parse_version(
    "2024.11.0"
)


def run_with_dask_dataframe(DMatrixT: Type, client: Client) -> None:
    import cupy as cp

    cp.cuda.runtime.setDevice(0)
    _X, _y, _ = generate_array()

    X = dd.from_dask_array(_X)
    y = dd.from_dask_array(_y)

    X = X.map_partitions(cudf.from_pandas)
    y = y.map_partitions(cudf.from_pandas)

    dtrain = DMatrixT(client, X, y)
    out = dxgb.train(
        client,
        {"tree_method": "hist", "debug_synchronize": True, "device": "cuda"},
        dtrain=dtrain,
        evals=[(dtrain, "X")],
        num_boost_round=4,
    )

    assert isinstance(out["booster"], dxgb.Booster)
    assert len(out["history"]["X"]["rmse"]) == 4

    predictions = dxgb.predict(client, out, dtrain)
    assert isinstance(predictions.compute(), np.ndarray)

    series_predictions = dxgb.inplace_predict(client, out, X)
    assert isinstance(series_predictions, dd.Series)

    single_node = out["booster"].predict(xgb.DMatrix(X.compute()))

    cp.testing.assert_allclose(single_node, predictions.compute())
    np.testing.assert_allclose(single_node, series_predictions.compute().to_numpy())

    predt = dxgb.predict(client, out, X)
    assert isinstance(predt, dd.Series)

    T = TypeVar("T")

    def is_df(part: T) -> T:
        assert isinstance(part, cudf.DataFrame), part
        return part

    predt.map_partitions(is_df, meta=dd.utils.make_meta({"prediction": "f4"}))

    cp.testing.assert_allclose(predt.values.compute(), single_node)

    # Work around https://github.com/dmlc/xgboost/issues/10752
    X.columns = X.columns.astype("object")
    # Make sure the output can be integrated back to original dataframe
    X["predict"] = predictions
    X["inplace_predict"] = series_predictions

    has_null = X.isnull().values.any().compute()
    assert bool(has_null) is False


def run_with_dask_array(DMatrixT: Type, client: Client) -> None:
    import cupy as cp

    cp.cuda.runtime.setDevice(0)
    X, y, _ = generate_array()

    X = X.map_blocks(cp.asarray)  # type: ignore
    y = y.map_blocks(cp.asarray)  # type: ignore
    dtrain = DMatrixT(client, X, y)
    out = dxgb.train(
        client,
        {"tree_method": "hist", "debug_synchronize": True, "device": "cuda"},
        dtrain=dtrain,
        evals=[(dtrain, "X")],
        num_boost_round=2,
    )
    from_dmatrix = dxgb.predict(client, out, dtrain).compute()
    assert (
        json.loads(out["booster"].save_config())["learner"]["gradient_booster"][
            "updater"
        ][0]["name"]
        == "grow_gpu_hist"
    )
    inplace_predictions = dxgb.inplace_predict(client, out, X).compute()
    single_node = out["booster"].predict(xgb.DMatrix(X.compute()))
    np.testing.assert_allclose(single_node, from_dmatrix)
    device = cp.cuda.runtime.getDevice()
    assert device == inplace_predictions.device.id
    single_node = cp.array(single_node)
    assert device == single_node.device.id
    cp.testing.assert_allclose(single_node, inplace_predictions)


def to_cp(x: Any, DMatrixT: Type) -> Any:
    import cupy

    if isinstance(x, np.ndarray) and DMatrixT is dxgb.DaskQuantileDMatrix:
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
    params["device"] = "cuda"
    params = dataset.set_params(params)
    # It doesn't make sense to distribute a completely
    # empty dataset.
    if dataset.X.shape[0] == 0:
        return

    chunk = 128
    X = to_cp(dataset.X, DMatrixT)
    X = da.from_array(X, chunks=(chunk, dataset.X.shape[1]))
    y = to_cp(dataset.y, DMatrixT)
    y_chunk = chunk if len(dataset.y.shape) == 1 else (chunk, dataset.y.shape[1])
    y = da.from_array(y, chunks=y_chunk)

    if dataset.w is not None:
        w = to_cp(dataset.w, DMatrixT)
        w = da.from_array(w, chunks=(chunk,))
    else:
        w = None

    if DMatrixT is dxgb.DaskQuantileDMatrix:
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
    )["history"]["train"][dataset.metric]
    note(str(history))

    # See note on `ObjFunction::UpdateTreeLeaf`.
    update_leaf = dataset.name.endswith("-l1")
    if update_leaf:
        assert history[0] + 1e-2 >= history[-1]
        return
    else:
        assert tm.non_increasing(history)


def test_tree_stats() -> None:
    with LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster) as client:
            local = run_tree_stats(client, "hist", "cuda")

    with LocalCUDACluster(n_workers=2) as cluster:
        with Client(cluster) as client:
            distributed = run_tree_stats(client, "hist", "cuda")

    assert local == distributed


class TestDistributedGPU:
    @pytest.mark.skipif(**tm.no_cudf())
    def test_boost_from_prediction(self, local_cuda_client: Client) -> None:
        import cudf
        from sklearn.datasets import load_breast_cancer, load_iris

        X_, y_ = load_breast_cancer(return_X_y=True)
        X = dd.from_array(X_, chunksize=100).map_partitions(cudf.from_pandas)
        y = dd.from_array(y_, chunksize=100).map_partitions(cudf.from_pandas)
        run_boost_from_prediction(X, y, "hist", "cuda", local_cuda_client)

        X_, y_ = load_iris(return_X_y=True)
        X = dd.from_array(X_, chunksize=50).map_partitions(cudf.from_pandas)
        y = dd.from_array(y_, chunksize=50).map_partitions(cudf.from_pandas)
        run_boost_from_prediction_multi_class(X, y, "hist", "cuda", local_cuda_client)

    def test_init_estimation(self, local_cuda_client: Client) -> None:
        check_init_estimation("gpu_hist", local_cuda_client)

    def test_uneven_nan(self) -> None:
        n_workers = 2
        with LocalCUDACluster(n_workers=n_workers) as cluster:
            with Client(cluster) as client:
                check_uneven_nan(client, "gpu_hist", n_workers)

    @pytest.mark.skipif(**tm.no_dask_cudf())
    def test_dask_dataframe(self, local_cuda_client: Client) -> None:
        run_with_dask_dataframe(dxgb.DaskDMatrix, local_cuda_client)
        run_with_dask_dataframe(dxgb.DaskQuantileDMatrix, local_cuda_client)

    @pytest.mark.skipif(**tm.no_dask_cudf())
    def test_categorical(self, local_cuda_client: Client) -> None:
        import dask_cudf

        X, y = make_categorical(local_cuda_client, 10000, 30, 13)
        X = X.to_backend("cudf")

        X_onehot, _ = make_categorical(local_cuda_client, 10000, 30, 13, True)
        X_onehot = X_onehot.to_backend("cudf")
        run_categorical(local_cuda_client, "hist", "cuda", X, X_onehot, y)

    @given(
        params=hist_parameter_strategy,
        num_rounds=strategies.integers(1, 20),
        dataset=tm.make_dataset_strategy(),
        dmatrix_type=strategies.sampled_from(
            [dxgb.DaskDMatrix, dxgb.DaskQuantileDMatrix]
        ),
    )
    @settings(
        deadline=duration(seconds=120),
        max_examples=20,
        suppress_health_check=suppress,
        print_blob=True,
    )
    @pytest.mark.skipif(**tm.no_cupy())
    def test_gpu_hist(
        self,
        params: Dict,
        num_rounds: int,
        dataset: tm.TestDataset,
        dmatrix_type: type,
        local_cuda_client: Client,
    ) -> None:
        params["tree_method"] = "hist"
        run_gpu_hist(params, num_rounds, dataset, dmatrix_type, local_cuda_client)

    @given(
        params=hist_parameter_strategy,
        num_rounds=strategies.integers(1, 20),
        dataset=tm.make_dataset_strategy(),
    )
    @settings(
        deadline=duration(seconds=120),
        max_examples=20,
        suppress_health_check=suppress,
        print_blob=True,
    )
    @pytest.mark.skipif(**tm.no_cupy())
    def test_gpu_approx(
        self,
        params: Dict,
        num_rounds: int,
        dataset: tm.TestDataset,
        local_cuda_client: Client,
    ) -> None:
        params["tree_method"] = "approx"
        run_gpu_hist(params, num_rounds, dataset, dxgb.DaskDMatrix, local_cuda_client)

    def test_empty_quantile_dmatrix(self, local_cuda_client: Client) -> None:
        client = local_cuda_client
        X, y = make_categorical(client, 1, 30, 13)
        X_valid, y_valid = make_categorical(client, 10000, 30, 13)

        Xy = xgb.dask.DaskQuantileDMatrix(client, X, y, enable_categorical=True)
        Xy_valid = xgb.dask.DaskQuantileDMatrix(
            client, X_valid, y_valid, ref=Xy, enable_categorical=True
        )
        result = xgb.dask.train(
            client,
            {"tree_method": "hist", "device": "cuda", "debug_synchronize": True},
            Xy,
            num_boost_round=10,
            evals=[(Xy_valid, "Valid")],
        )
        predt = xgb.dask.inplace_predict(client, result["booster"], X).compute()
        np.testing.assert_allclose(y.compute(), predt)
        rmse = result["history"]["Valid"]["rmse"][-1]
        assert rmse < 32.0

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dask_array(self, local_cuda_client: Client) -> None:
        run_with_dask_array(dxgb.DaskDMatrix, local_cuda_client)
        run_with_dask_array(dxgb.DaskQuantileDMatrix, local_cuda_client)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_early_stopping(self, local_cuda_client: Client) -> None:
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)

        m = dxgb.DaskDMatrix(local_cuda_client, X, y)

        valid = dxgb.DaskDMatrix(local_cuda_client, X, y)
        early_stopping_rounds = 5
        booster = dxgb.train(
            local_cuda_client,
            {
                "objective": "binary:logistic",
                "eval_metric": "error",
                "tree_method": "hist",
                "device": "cuda",
            },
            m,
            evals=[(valid, "Valid")],
            num_boost_round=1000,
            early_stopping_rounds=early_stopping_rounds,
        )["booster"]
        assert hasattr(booster, "best_score")
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

        valid_X = X
        valid_y = y
        cls = dxgb.DaskXGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cuda",
            eval_metric="error",
            n_estimators=100,
            early_stopping_rounds=early_stopping_rounds,
        )
        cls.client = local_cuda_client
        cls.fit(
            X,
            y,
            eval_set=[(valid_X, valid_y)],
        )
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    @pytest.mark.xfail(
        dask_version_ge110, reason="Test cannot pass with Dask 2024.11.0+"
    )
    @pytest.mark.skipif(**tm.no_cudf())
    @pytest.mark.parametrize("model", ["boosting"])
    def test_dask_classifier(self, model: str, local_cuda_client: Client) -> None:
        import dask_cudf

        X_, y_, w_ = generate_array(with_weights=True)
        y_ = (y_ * 10).astype(np.int32)
        X = dd.from_dask_array(X_).to_backend("cudf")
        y = dd.from_dask_array(y_).to_backend("cudf")
        w = dd.from_dask_array(w_).to_backend("cudf")
        run_dask_classifier(X, y, w, model, "gpu_hist", local_cuda_client, 10)

    def test_empty_dmatrix(self, local_cuda_client: Client) -> None:
        parameters = {
            "tree_method": "hist",
            "debug_synchronize": True,
            "device": "cuda",
        }
        run_empty_dmatrix_reg(local_cuda_client, parameters)
        run_empty_dmatrix_cls(local_cuda_client, parameters)

    @pytest.mark.skipif(**tm.no_dask_cudf())
    def test_empty_partition(self, local_cuda_client: Client) -> None:
        import cudf
        import cupy
        import dask_cudf

        mult = 100
        df = cudf.DataFrame(
            {
                "a": [1, 2, 3, 4, 5.1] * mult,
                "b": [10, 15, 29.3, 30, 31] * mult,
                "y": [10, 20, 30, 40.0, 50] * mult,
            }
        )
        parameters = {
            "tree_method": "hist",
            "debug_synchronize": True,
            "device": "cuda",
        }

        empty = df.iloc[:0]
        ddf = dask_cudf.concat(
            [dask_cudf.from_cudf(empty, npartitions=1)]
            + [dask_cudf.from_cudf(df, npartitions=3)]
            + [dask_cudf.from_cudf(df, npartitions=3)]
        )
        X = ddf[ddf.columns.difference(["y"])]
        y = ddf[["y"]]
        dtrain = dxgb.DaskQuantileDMatrix(local_cuda_client, X, y)
        bst_empty = xgb.dask.train(
            local_cuda_client, parameters, dtrain, evals=[(dtrain, "train")]
        )
        predt_empty = dxgb.predict(local_cuda_client, bst_empty, X).compute().values

        ddf = dask_cudf.concat(
            [dask_cudf.from_cudf(df, npartitions=3)]
            + [dask_cudf.from_cudf(df, npartitions=3)]
        )
        X = ddf[ddf.columns.difference(["y"])]
        y = ddf[["y"]]
        dtrain = dxgb.DaskQuantileDMatrix(local_cuda_client, X, y)
        bst = xgb.dask.train(
            local_cuda_client, parameters, dtrain, evals=[(dtrain, "train")]
        )

        predt = dxgb.predict(local_cuda_client, bst, X).compute().values
        cupy.testing.assert_allclose(predt, predt_empty)

        predt = dxgb.predict(local_cuda_client, bst, dtrain).compute()
        cupy.testing.assert_allclose(predt, predt_empty)

        predt = dxgb.inplace_predict(local_cuda_client, bst, X).compute().values
        cupy.testing.assert_allclose(predt, predt_empty)

        df = df.to_pandas()
        empty = df.iloc[:0]
        ddf = dd.concat(
            [dd.from_pandas(empty, npartitions=1)]
            + [dd.from_pandas(df, npartitions=3)]
            + [dd.from_pandas(df, npartitions=3)]
        )
        X = ddf[ddf.columns.difference(["y"])]
        y = ddf[["y"]]

        predt_empty = cupy.asnumpy(predt_empty)

        predt = dxgb.predict(local_cuda_client, bst_empty, X).compute().values
        np.testing.assert_allclose(predt, predt_empty)

        in_predt = (
            dxgb.inplace_predict(local_cuda_client, bst_empty, X).compute().values
        )
        np.testing.assert_allclose(predt, in_predt)

    def test_empty_dmatrix_auc(self, local_cuda_client: Client) -> None:
        n_workers = len(tm.get_client_workers(local_cuda_client))
        run_empty_dmatrix_auc(local_cuda_client, "cuda", n_workers)

    def test_auc(self, local_cuda_client: Client) -> None:
        run_auc(local_cuda_client, "cuda")

    def test_invalid_ordinal(self, local_cuda_client: Client) -> None:
        """One should not specify the device ordinal with dask."""
        with pytest.raises(ValueError, match="device=cuda"):
            X, y, _ = generate_array()
            m = dxgb.DaskDMatrix(local_cuda_client, X, y)
            dxgb.train(local_cuda_client, {"device": "cuda:0"}, m)

        booster = dxgb.train(local_cuda_client, {"device": "cuda"}, m)["booster"]
        assert (
            json.loads(booster.save_config())["learner"]["generic_param"]["device"]
            == "cuda:0"
        )

    def test_data_initialization(self, local_cuda_client: Client) -> None:
        X, y, _ = generate_array()
        fw = da.random.random((random_cols,))
        fw = fw - fw.min()
        m = dxgb.DaskDMatrix(local_cuda_client, X, y, feature_weights=fw)

        workers = tm.get_client_workers(local_cuda_client)
        rabit_args = local_cuda_client.sync(
            dxgb._get_rabit_args, len(workers), None, local_cuda_client
        )

        def worker_fn(worker_addr: str, data_ref: Dict) -> None:
            with dxgb.CommunicatorContext(**rabit_args):
                local_dtrain = dxgb._dmatrix_from_list_of_parts(**data_ref, nthread=7)
                fw_rows = local_dtrain.get_float_info("feature_weights").shape[0]
                assert fw_rows == local_dtrain.num_col()

        futures = []
        for i in range(len(workers)):
            futures.append(
                local_cuda_client.submit(
                    worker_fn,
                    workers[i],
                    m._create_fn_args(workers[i]),
                    pure=False,
                    workers=[workers[i]],
                )
            )
        local_cuda_client.gather(futures)

    def test_interface_consistency(self) -> None:
        sig = OrderedDict(signature(dxgb.DaskDMatrix).parameters)
        del sig["client"]
        ddm_names = list(sig.keys())
        sig = OrderedDict(signature(dxgb.DaskQuantileDMatrix).parameters)
        del sig["client"]
        del sig["max_bin"]
        del sig["ref"]
        ddqdm_names = list(sig.keys())
        assert len(ddm_names) == len(ddqdm_names)

        # between dask
        for i in range(len(ddm_names)):
            assert ddm_names[i] == ddqdm_names[i]

        sig = OrderedDict(signature(xgb.DMatrix).parameters)
        del sig["nthread"]  # no nthread in dask
        dm_names = list(sig.keys())
        sig = OrderedDict(signature(xgb.QuantileDMatrix).parameters)
        del sig["nthread"]
        del sig["max_bin"]
        del sig["ref"]
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


@pytest.mark.skipif(**tm.no_cupy())
def test_with_asyncio(local_cuda_client: Client) -> None:
    address = local_cuda_client.scheduler.address
    output = asyncio.run(run_from_dask_array_asyncio(address))
    assert isinstance(output["booster"], xgb.Booster)
    assert isinstance(output["history"], dict)


@pytest.mark.skipif(
    condition=not xgb.build_info()["USE_DLOPEN_NCCL"],
    reason="Not compiled with dlopen.",
)
def test_invalid_nccl(local_cuda_client: Client) -> None:
    client = local_cuda_client
    workers = tm.get_client_workers(client)
    args = client.sync(
        dxgb._get_rabit_args, len(workers), dxgb._get_dask_config(), client
    )

    def run(wid: int) -> None:
        ctx = CommunicatorContext(dmlc_nccl_path="foo", **args)
        X, y, w = tm.make_regression(n_samples=10, n_features=10, use_cupy=True)

        with ctx:
            with pytest.raises(ValueError, match=r"pip install"):
                xgb.QuantileDMatrix(X, y, weight=w)

    futures = client.map(run, range(len(workers)), workers=workers)
    client.gather(futures)


@pytest.mark.skipif(
    condition=not xgb.build_info()["USE_DLOPEN_NCCL"],
    reason="Not compiled with dlopen.",
)
@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_nccl_load(local_cuda_client: Client, tree_method: str) -> None:
    X, y, w = tm.make_regression(128, 16, use_cupy=True)

    def make_model() -> None:
        xgb.XGBRegressor(
            device="cuda",
            tree_method=tree_method,
            objective="reg:quantileerror",
            verbosity=2,
            quantile_alpha=[0.2, 0.8],
        ).fit(X, y, sample_weight=w)

    # no nccl load when using single-node.
    with tm.captured_output() as (out, err):
        make_model()
        assert out.getvalue().find("NCCL") == -1
        assert err.getvalue().find("NCCL") == -1

    client = local_cuda_client
    workers = tm.get_client_workers(client)
    args = client.sync(
        dxgb._get_rabit_args, len(workers), dxgb._get_dask_config(), client
    )

    # nccl is loaded
    def run(wid: int) -> None:
        # FIXME(jiamingy): https://github.com/dmlc/xgboost/issues/9147
        from xgboost.core import _LIB, _register_log_callback

        _register_log_callback(_LIB)

        with CommunicatorContext(**args):
            with tm.captured_output() as (out, err):
                make_model()
                assert out.getvalue().find("Loaded shared NCCL") != -1, out.getvalue()

    futures = client.map(run, range(len(workers)), workers=workers)
    client.gather(futures)


async def run_from_dask_array_asyncio(scheduler_address: str) -> dxgb.TrainReturnT:
    async with Client(scheduler_address, asynchronous=True) as client:
        import cupy as cp

        X, y, _ = generate_array()
        X = X.map_blocks(cp.array)  # type: ignore
        y = y.map_blocks(cp.array)  # type: ignore

        m = await xgb.dask.DaskQuantileDMatrix(client, X, y)
        output = await xgb.dask.train(
            client, {"tree_method": "hist", "device": "cuda"}, dtrain=m
        )

        with_m = await xgb.dask.predict(client, output, m)
        with_X = await xgb.dask.predict(client, output, X)
        inplace = await xgb.dask.inplace_predict(client, output, X)
        assert isinstance(with_m, da.Array)
        assert isinstance(with_X, da.Array)
        assert isinstance(inplace, da.Array)

        cp.testing.assert_allclose(
            await client.compute(with_m), await client.compute(with_X)
        )
        cp.testing.assert_allclose(
            await client.compute(with_m), await client.compute(inplace)
        )

        client.shutdown()
        return output
