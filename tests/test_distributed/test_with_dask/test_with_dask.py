"""Copyright 2019-2024, XGBoost contributors"""

import asyncio
import json
import os
import pickle
import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, Literal, Optional, Tuple, Type, Union

import dask
import dask.array as da
import dask.dataframe as dd
import hypothesis
import numpy as np
import pytest
import scipy
import sklearn
from distributed import Client, LocalCluster, Nanny, Worker
from distributed.scheduler import KilledWorker, Scheduler
from distributed.utils_test import async_poll_for, gen_cluster
from hypothesis import HealthCheck, assume, given, note, settings
from sklearn.datasets import make_classification, make_regression

import xgboost as xgb
from xgboost import collective as coll
from xgboost import dask as dxgb
from xgboost import testing as tm
from xgboost.collective import Config as CollConfig
from xgboost.dask import DaskDMatrix
from xgboost.testing.dask import check_init_estimation, check_uneven_nan, get_rabit_args
from xgboost.testing.params import hist_cache_strategy, hist_parameter_strategy
from xgboost.testing.shared import (
    get_feature_weights,
    validate_data_initialization,
    validate_leaf_output,
)

dask.config.set({"distributed.scheduler.allowed-failures": False})

pytestmark = tm.timeout(60)


if hasattr(HealthCheck, "function_scoped_fixture"):
    suppress = [HealthCheck.function_scoped_fixture]
else:
    suppress = hypothesis.utils.conventions.not_set  # type:ignore


@pytest.fixture(scope="module")
def cluster() -> Generator:
    n_threads = os.cpu_count()
    assert n_threads is not None
    with LocalCluster(
        n_workers=2, threads_per_worker=n_threads // 2, dashboard_address=":0"
    ) as dask_cluster:
        yield dask_cluster


@pytest.fixture
def client(cluster: "LocalCluster") -> Generator:
    with Client(cluster) as dask_client:
        yield dask_client


kRows = 1000
kCols = 10
kWorkers = 5


def make_categorical(
    client: Client,
    n_samples: int,
    n_features: int,
    n_categories: int,
    onehot: bool = False,
) -> Tuple[dd.DataFrame, dd.Series]:
    workers = tm.dask.get_client_workers(client)
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
        # make sure there's at least one sample for testing empty DMatrix
        if n_samples == 1 and i == 0:
            l_n_samples = 1
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


def generate_array(
    with_weights: bool = False,
) -> Tuple[da.Array, da.Array, Optional[da.Array]]:
    chunk_size = 20
    rng = da.random.RandomState(1994)
    X = rng.random_sample((kRows, kCols), chunks=(chunk_size, -1))
    y = rng.random_sample(kRows, chunks=chunk_size)
    if with_weights:
        w = rng.random_sample(kRows, chunks=chunk_size)
        return X, y, w
    return X, y, None


@pytest.mark.parametrize("to_frame", [True, False])
def test_xgbclassifier_classes_type_and_value(to_frame: bool, client: "Client"):
    X, y = make_classification(n_samples=1000, n_features=4, random_state=123)
    if to_frame:
        import pandas as pd

        feats = [f"var_{i}" for i in range(4)]
        df = pd.DataFrame(X, columns=feats)
        df["target"] = y
        df = dd.from_pandas(df, npartitions=1)
        X, y = df[feats], df["target"]
    else:
        X = da.from_array(X)
        y = da.from_array(y)

    est = dxgb.DaskXGBClassifier(n_estimators=10).fit(X, y)
    assert isinstance(est.classes_, np.ndarray)
    np.testing.assert_array_equal(est.classes_, np.array([0, 1]))


def test_from_dask_dataframe() -> None:
    with LocalCluster(n_workers=kWorkers, dashboard_address=":0") as cluster:
        with Client(cluster) as client:
            X_, y_, _ = generate_array()

            X = dd.from_dask_array(X_)
            y = dd.from_dask_array(y_)

            dtrain = DaskDMatrix(client, X, y)
            booster = dxgb.train(client, {}, dtrain, num_boost_round=2)["booster"]

            prediction = dxgb.predict(client, model=booster, data=dtrain)

            assert prediction.ndim == 1
            assert isinstance(prediction, da.Array)
            assert prediction.shape[0] == kRows

            with pytest.raises(TypeError):
                # evals_result is not supported in dask interface.
                dxgb.train(  # type:ignore
                    client, {}, dtrain, num_boost_round=2, evals_result={}
                )
            # force prediction to be computed
            from_dmatrix = prediction.compute()

            prediction = dxgb.predict(client, model=booster, data=X)
            from_df = prediction.compute()

            assert isinstance(prediction, dd.Series)
            assert np.all(prediction.compute().values == from_dmatrix)
            assert np.all(from_dmatrix == from_df.to_numpy())

            series_predictions = dxgb.inplace_predict(client, booster, X)
            assert isinstance(series_predictions, dd.Series)
            np.testing.assert_allclose(
                series_predictions.compute().values, from_dmatrix
            )

            # Make sure the output can be integrated back to original dataframe
            X["predict"] = prediction
            X["inplace_predict"] = series_predictions

            assert bool(X.isnull().values.any().compute()) is False


def test_from_dask_array() -> None:
    with LocalCluster(
        n_workers=kWorkers, threads_per_worker=5, dashboard_address=":0"
    ) as cluster:
        with Client(cluster) as client:
            X, y, _ = generate_array()
            dtrain = DaskDMatrix(client, X, y)
            # results is {'booster': Booster, 'history': {...}}
            result = dxgb.train(client, {}, dtrain)

            prediction = dxgb.predict(client, result, dtrain)
            assert prediction.shape[0] == kRows

            assert isinstance(prediction, da.Array)
            # force prediction to be computed
            prediction = prediction.compute()

            booster: xgb.Booster = result["booster"]
            single_node_predt = booster.predict(xgb.DMatrix(X.compute()))
            np.testing.assert_allclose(prediction, single_node_predt)

            config = json.loads(booster.save_config())
            assert int(config["learner"]["generic_param"]["nthread"]) == 5

            from_arr = dxgb.predict(client, model=booster, data=X)

            assert isinstance(from_arr, da.Array)
            assert np.all(single_node_predt == from_arr.compute())


def test_dask_sparse(client: "Client") -> None:
    X_, y_ = make_classification(n_samples=1000, n_informative=5, n_classes=3)
    rng = np.random.default_rng(seed=0)
    idx = rng.integers(low=0, high=X_.shape[0], size=X_.shape[0] // 4)
    X_[idx, :] = np.nan

    # numpy
    X, y = da.from_array(X_), da.from_array(y_)
    clf = dxgb.DaskXGBClassifier(tree_method="hist", n_estimators=10)
    clf.client = client
    clf.fit(X, y, eval_set=[(X, y)])
    dense_results = clf.evals_result()

    # scipy sparse
    X, y = da.from_array(X_).map_blocks(scipy.sparse.csr_matrix), da.from_array(y_)
    clf = dxgb.DaskXGBClassifier(tree_method="hist", n_estimators=10)
    clf.client = client
    clf.fit(X, y, eval_set=[(X, y)])
    sparse_results = clf.evals_result()
    np.testing.assert_allclose(
        dense_results["validation_0"]["mlogloss"],
        sparse_results["validation_0"]["mlogloss"],
    )


def run_categorical(
    client: "Client", tree_method: str, device: str, X, X_onehot, y
) -> None:
    # Force onehot
    parameters = {
        "tree_method": tree_method,
        "device": device,
        "max_cat_to_onehot": 9999,
    }
    rounds = 10
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

    def check_model_output(model: dxgb.Booster) -> None:
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

    check_model_output(output["booster"])
    reg = dxgb.DaskXGBRegressor(
        enable_categorical=True,
        n_estimators=10,
        tree_method=tree_method,
        device=device,
        # force onehot
        max_cat_to_onehot=9999,
    )
    reg.fit(X, y)

    check_model_output(reg.get_booster())

    reg = dxgb.DaskXGBRegressor(
        enable_categorical=True, n_estimators=10, tree_method="exact"
    )
    with pytest.raises(ValueError, match="categorical data"):
        reg.fit(X, y)
    # check partition based
    reg = dxgb.DaskXGBRegressor(
        enable_categorical=True,
        n_estimators=10,
        tree_method=tree_method,
        device=device,
    )
    reg.fit(X, y, eval_set=[(X, y)])
    assert tm.non_increasing(reg.evals_result()["validation_0"]["rmse"])

    booster = reg.get_booster()
    predt = dxgb.predict(client, booster, X).compute().values
    inpredt = dxgb.inplace_predict(client, booster, X).compute().values

    if hasattr(predt, "get"):
        predt = predt.get()
    if hasattr(inpredt, "get"):
        inpredt = inpredt.get()

    np.testing.assert_allclose(predt, inpredt)


def test_categorical(client: "Client") -> None:
    X, y = make_categorical(client, 10000, 30, 13)
    X_onehot, _ = make_categorical(client, 10000, 30, 13, onehot=True)
    run_categorical(client, "approx", "cpu", X, X_onehot, y)
    run_categorical(client, "hist", "cpu", X, X_onehot, y)

    ft = ["c"] * X.shape[1]
    reg = dxgb.DaskXGBRegressor(
        tree_method="hist", feature_types=ft, enable_categorical=True
    )
    reg.fit(X, y)
    assert reg.get_booster().feature_types == ft


def test_dask_predict_shape_infer(client: "Client") -> None:
    X, y = make_classification(n_samples=kRows, n_informative=5, n_classes=3)
    X_ = dd.from_array(X, chunksize=100)
    y_ = dd.from_array(y, chunksize=100)
    dtrain = dxgb.DaskDMatrix(client, data=X_, label=y_)

    model = dxgb.train(
        client, {"objective": "multi:softprob", "num_class": 3}, dtrain=dtrain
    )

    preds = dxgb.predict(client, model, dtrain)
    assert preds.shape[0] == preds.compute().shape[0]
    assert preds.shape[1] == preds.compute().shape[1]

    prediction = dxgb.predict(client, model, X_, output_margin=True)
    assert isinstance(prediction, dd.DataFrame)

    prediction = prediction.compute()
    assert prediction.ndim == 2
    assert prediction.shape[0] == kRows
    assert prediction.shape[1] == 3

    prediction = dxgb.inplace_predict(client, model, X_, predict_type="margin")
    assert isinstance(prediction, dd.DataFrame)
    prediction = prediction.compute()
    assert prediction.ndim == 2
    assert prediction.shape[0] == kRows
    assert prediction.shape[1] == 3


def run_boost_from_prediction_multi_class(
    X: dd.DataFrame,
    y: dd.Series,
    tree_method: str,
    device: str,
    client: "Client",
) -> None:
    model_0 = dxgb.DaskXGBClassifier(
        learning_rate=0.3,
        n_estimators=4,
        tree_method=tree_method,
        max_bin=768,
        device=device,
    )
    model_0.fit(X=X, y=y, eval_set=[(X, y)])
    margin = dxgb.inplace_predict(
        client, model_0.get_booster(), X, predict_type="margin"
    )
    margin.columns = [f"m_{i}" for i in range(margin.shape[1])]

    model_1 = dxgb.DaskXGBClassifier(
        learning_rate=0.3,
        n_estimators=4,
        tree_method=tree_method,
        max_bin=768,
        device=device,
    )
    model_1.fit(
        X=X, y=y, base_margin=margin, eval_set=[(X, y)], base_margin_eval_set=[margin]
    )
    predictions_1 = dxgb.predict(
        client,
        model_1.get_booster(),
        dxgb.DaskDMatrix(client, X, base_margin=margin),
        output_margin=True,
    )

    model_2 = dxgb.DaskXGBClassifier(
        learning_rate=0.3,
        n_estimators=8,
        tree_method=tree_method,
        max_bin=768,
        device=device,
    )
    model_2.fit(X=X, y=y, eval_set=[(X, y)])
    predictions_2 = dxgb.inplace_predict(
        client, model_2.get_booster(), X, predict_type="margin"
    )
    a = predictions_1.compute()
    b = predictions_2.compute()
    # cupy/cudf
    if hasattr(a, "get"):
        a = a.get()
    if hasattr(b, "values"):
        b = b.values
    if hasattr(b, "get"):
        b = b.get()
    np.testing.assert_allclose(a, b, atol=1e-5)


def run_boost_from_prediction(
    X: dd.DataFrame,
    y: dd.Series,
    tree_method: str,
    device: str,
    client: "Client",
) -> None:
    X, y = client.persist([X, y])

    model_0 = dxgb.DaskXGBClassifier(
        learning_rate=0.3,
        n_estimators=3,
        tree_method=tree_method,
        max_bin=512,
        device=device,
    )
    model_0.fit(X=X, y=y, eval_set=[(X, y)])
    margin: dd.Series = model_0.predict(X, output_margin=True)

    model_1 = dxgb.DaskXGBClassifier(
        learning_rate=0.3,
        n_estimators=3,
        tree_method=tree_method,
        max_bin=512,
        device=device,
    )
    model_1.fit(
        X=X, y=y, base_margin=margin, eval_set=[(X, y)], base_margin_eval_set=[margin]
    )
    predictions_1: dd.Series = model_1.predict(X, base_margin=margin)

    model_2 = dxgb.DaskXGBClassifier(
        learning_rate=0.3,
        n_estimators=6,
        tree_method=tree_method,
        max_bin=512,
        device=device,
    )
    model_2.fit(X=X, y=y, eval_set=[(X, y)])
    predictions_2: dd.Series = model_2.predict(X)

    logloss_concat = (
        model_0.evals_result()["validation_0"]["logloss"]
        + model_1.evals_result()["validation_0"]["logloss"]
    )
    logloss_2 = model_2.evals_result()["validation_0"]["logloss"]
    np.testing.assert_allclose(logloss_concat, logloss_2, rtol=1e-4)

    margined = dxgb.DaskXGBClassifier(n_estimators=4)
    margined.fit(
        X=X, y=y, base_margin=margin, eval_set=[(X, y)], base_margin_eval_set=[margin]
    )

    unmargined = dxgb.DaskXGBClassifier(n_estimators=4)
    unmargined.fit(X=X, y=y, eval_set=[(X, y)], base_margin=margin)

    margined_res = margined.evals_result()["validation_0"]["logloss"]
    unmargined_res = unmargined.evals_result()["validation_0"]["logloss"]

    assert len(margined_res) == len(unmargined_res)
    for i in range(len(margined_res)):
        # margined is correct one, so smaller error.
        assert margined_res[i] < unmargined_res[i]


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_boost_from_prediction(tree_method: str) -> None:
    from sklearn.datasets import load_breast_cancer, load_digits

    n_threads = os.cpu_count()
    assert n_threads is not None
    # This test has strict reproducibility requirements. However, Dask is freed to move
    # partitions between workers and modify the partitions' size during the test. Given
    # the lack of control over the partitioning logic, here we use a single worker as a
    # workaround.
    n_workers = 1

    with LocalCluster(
        n_workers=n_workers, threads_per_worker=n_threads // n_workers
    ) as cluster:
        with Client(cluster) as client:
            X_, y_ = load_breast_cancer(return_X_y=True)
            X, y = dd.from_array(X_, chunksize=200), dd.from_array(y_, chunksize=200)
            run_boost_from_prediction(X, y, tree_method, "cpu", client)

            X_, y_ = load_digits(return_X_y=True)
            X, y = dd.from_array(X_, chunksize=100), dd.from_array(y_, chunksize=100)
            run_boost_from_prediction_multi_class(X, y, tree_method, "cpu", client)


def test_inplace_predict(client: "Client") -> None:
    from sklearn.datasets import load_diabetes

    X_, y_ = load_diabetes(return_X_y=True)
    X, y = dd.from_array(X_, chunksize=32), dd.from_array(y_, chunksize=32)
    reg = dxgb.DaskXGBRegressor(n_estimators=4).fit(X, y)
    booster = reg.get_booster()
    base_margin = y

    inplace = dxgb.inplace_predict(
        client, booster, X, base_margin=base_margin
    ).compute()
    Xy = dxgb.DaskDMatrix(client, X, base_margin=base_margin)
    copied = dxgb.predict(client, booster, Xy).compute()
    np.testing.assert_allclose(inplace, copied)


def test_dask_missing_value_reg(client: "Client") -> None:
    X_0 = np.ones((20 // 2, kCols))
    X_1 = np.zeros((20 // 2, kCols))
    X = np.concatenate([X_0, X_1], axis=0)
    np.random.shuffle(X)
    X = da.from_array(X)
    X = X.rechunk(20, 1)
    y = da.random.randint(0, 3, size=20)
    y.rechunk(20)
    regressor = dxgb.DaskXGBRegressor(verbosity=1, n_estimators=2, missing=0.0)
    regressor.client = client
    regressor.set_params(tree_method="hist")
    regressor.fit(X, y, eval_set=[(X, y)])
    dd_predt = regressor.predict(X).compute()

    np_X = X.compute()
    np_predt = regressor.get_booster().predict(xgb.DMatrix(np_X, missing=0.0))
    np.testing.assert_allclose(np_predt, dd_predt)


def test_dask_missing_value_cls(client: "Client") -> None:
    X_0 = np.ones((kRows // 2, kCols))
    X_1 = np.zeros((kRows // 2, kCols))
    X = np.concatenate([X_0, X_1], axis=0)
    np.random.shuffle(X)
    X = da.from_array(X)
    X = X.rechunk(20, None)
    y = da.random.randint(0, 3, size=kRows)
    y = y.rechunk(20, 1)
    cls = dxgb.DaskXGBClassifier(
        verbosity=1, n_estimators=2, tree_method="hist", missing=0.0
    )
    cls.client = client
    cls.fit(X, y, eval_set=[(X, y)])
    dd_pred_proba = cls.predict_proba(X).compute()

    np_X = X.compute()
    np_pred_proba = cls.get_booster().predict(xgb.DMatrix(np_X, missing=0.0))
    np.testing.assert_allclose(np_pred_proba, dd_pred_proba)

    cls = dxgb.DaskXGBClassifier()
    assert hasattr(cls, "missing")


@pytest.mark.parametrize("model", ["boosting", "rf"])
def test_dask_regressor(model: str, client: "Client") -> None:
    X, y, w = generate_array(with_weights=True)
    if model == "boosting":
        regressor = dxgb.DaskXGBRegressor(verbosity=1, n_estimators=2)
    else:
        regressor = dxgb.DaskXGBRFRegressor(verbosity=1, n_estimators=2)

    assert regressor._estimator_type == "regressor"
    assert sklearn.base.is_regressor(regressor)

    regressor.set_params(tree_method="hist")
    regressor.client = client
    regressor.fit(X, y, sample_weight=w, eval_set=[(X, y)])
    prediction = regressor.predict(X)

    assert prediction.ndim == 1
    assert prediction.shape[0] == kRows

    history = regressor.evals_result()

    assert isinstance(prediction, da.Array)
    assert isinstance(history, dict)

    assert list(history["validation_0"].keys())[0] == "rmse"
    forest = int(
        json.loads(regressor.get_booster().save_config())["learner"][
            "gradient_booster"
        ]["gbtree_model_param"]["num_parallel_tree"]
    )

    if model == "boosting":
        assert len(history["validation_0"]["rmse"]) == 2
        assert forest == 1
    else:
        assert len(history["validation_0"]["rmse"]) == 1
        assert forest == 2


def run_dask_classifier(
    X: dxgb._DaskCollection,
    y: dxgb._DaskCollection,
    w: dxgb._DaskCollection,
    model: str,
    tree_method: Optional[str],
    device: Literal["cpu", "cuda"],
    client: "Client",
    n_classes: int,
) -> None:
    metric = "merror" if n_classes > 2 else "logloss"

    if model == "boosting":
        classifier = dxgb.DaskXGBClassifier(
            verbosity=1,
            n_estimators=2,
            eval_metric=metric,
            tree_method=tree_method,
            device=device,
        )
    else:
        classifier = dxgb.DaskXGBRFClassifier(
            verbosity=1,
            n_estimators=2,
            eval_metric=metric,
            tree_method=tree_method,
            device=device,
        )

    assert classifier._estimator_type == "classifier"
    assert sklearn.base.is_classifier(classifier)

    classifier.client = client
    classifier.fit(X, y, sample_weight=w, eval_set=[(X, y)])
    prediction = classifier.predict(X).compute()

    assert prediction.ndim == 1
    assert prediction.shape[0] == kRows

    history = classifier.evals_result()

    assert isinstance(history, dict)

    assert list(history.keys())[0] == "validation_0"
    assert list(history["validation_0"].keys())[0] == metric
    assert len(list(history["validation_0"])) == 1

    config = json.loads(classifier.get_booster().save_config())
    n_threads = int(config["learner"]["generic_param"]["nthread"])
    assert n_threads != 0 and n_threads != os.cpu_count()

    forest = int(
        config["learner"]["gradient_booster"]["gbtree_model_param"]["num_parallel_tree"]
    )
    if model == "boosting":
        assert len(history["validation_0"][metric]) == 2
        assert forest == 1
    else:
        assert len(history["validation_0"][metric]) == 1
        assert forest == 2

    # Test .predict_proba()
    probas = classifier.predict_proba(X).compute()
    assert classifier.n_classes_ == n_classes
    assert probas.ndim == 2
    assert probas.shape[0] == kRows
    assert probas.shape[1] == n_classes

    if n_classes > 2:
        cls_booster = classifier.get_booster()
        single_node_proba = cls_booster.inplace_predict(X.compute())

        # test shared by CPU and GPU
        if isinstance(single_node_proba, np.ndarray):
            np.testing.assert_allclose(single_node_proba, probas)
        else:
            import cupy

            cupy.testing.assert_allclose(single_node_proba, probas)

    # Test with dataframe, not shared with GPU as cupy doesn't work well with da.unique.
    if isinstance(X, da.Array) and n_classes > 2:
        X_d: dd.DataFrame = X.to_dask_dataframe()

        assert classifier.n_classes_ == n_classes
        prediction_df = classifier.predict(X_d).compute()

        assert prediction_df.ndim == 1
        assert prediction_df.shape[0] == kRows
        np.testing.assert_allclose(prediction_df, prediction)

        probas = classifier.predict_proba(X).compute()
        np.testing.assert_allclose(single_node_proba, probas)


@pytest.mark.parametrize("model", ["boosting", "rf"])
def test_dask_classifier(model: str, client: "Client") -> None:
    X, y, w = generate_array(with_weights=True)
    y = (y * 10).astype(np.int32)
    assert w is not None
    run_dask_classifier(X, y, w, model, None, "cpu", client, 10)

    y_bin = y.copy()
    y_bin[y > 5] = 1.0
    y_bin[y <= 5] = 0.0
    run_dask_classifier(X, y_bin, w, model, None, "cpu", client, 2)


def test_empty_dmatrix_training_continuation(client: "Client") -> None:
    kRows, kCols = 1, 97
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.rand(kRows))
    X.columns = ["X" + str(i) for i in range(0, kCols)]
    dtrain = dxgb.DaskDMatrix(client, X, y)

    kRows += 1000
    X = dd.from_array(np.random.randn(kRows, kCols), chunksize=10)
    X.columns = ["X" + str(i) for i in range(0, kCols)]
    y = dd.from_array(np.random.rand(kRows), chunksize=10)
    valid = dxgb.DaskDMatrix(client, X, y)

    out = dxgb.train(
        client,
        {"tree_method": "hist"},
        dtrain=dtrain,
        num_boost_round=2,
        evals=[(valid, "validation")],
    )

    out = dxgb.train(
        client,
        {"tree_method": "hist"},
        dtrain=dtrain,
        xgb_model=out["booster"],
        num_boost_round=2,
        evals=[(valid, "validation")],
    )
    assert dxgb.predict(client, out, dtrain).compute().shape[0] == 1


def run_empty_dmatrix_reg(client: "Client", parameters: dict) -> None:
    def _check_outputs(out: dxgb.TrainReturnT, predictions: np.ndarray) -> None:
        assert isinstance(out["booster"], dxgb.Booster)
        for _, v in out["history"]["validation"].items():
            assert len(v) == 2
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 1

    kRows, kCols = 1, 97
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.rand(kRows))
    dtrain = dxgb.DaskDMatrix(client, X, y)

    out = dxgb.train(
        client,
        parameters,
        dtrain=dtrain,
        evals=[(dtrain, "validation")],
        num_boost_round=2,
    )
    predictions = dxgb.predict(client=client, model=out, data=dtrain).compute()
    _check_outputs(out, predictions)

    # valid has more rows than train
    kRows += 1
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.rand(kRows))
    valid = dxgb.DaskDMatrix(client, X, y)
    out = dxgb.train(
        client,
        parameters,
        dtrain=dtrain,
        evals=[(valid, "validation")],
        num_boost_round=2,
    )
    predictions = dxgb.predict(client=client, model=out, data=dtrain).compute()
    _check_outputs(out, predictions)

    # train has more rows than evals
    valid = dtrain
    kRows += 1
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.rand(kRows))
    dtrain = dxgb.DaskDMatrix(client, X, y)

    out = dxgb.train(
        client,
        parameters,
        dtrain=dtrain,
        evals=[(valid, "validation")],
        num_boost_round=2,
    )
    predictions = dxgb.predict(client=client, model=out, data=valid).compute()
    _check_outputs(out, predictions)


def run_empty_dmatrix_cls(client: "Client", parameters: dict) -> None:
    n_classes = 4

    def _check_outputs(out: dxgb.TrainReturnT, predictions: np.ndarray) -> None:
        assert isinstance(out["booster"], dxgb.Booster)
        assert len(out["history"]["validation"]["merror"]) == 2
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[1] == n_classes, predictions.shape

    kRows, kCols = 1, 97
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.randint(low=0, high=n_classes, size=kRows))
    dtrain = dxgb.DaskDMatrix(client, X, y)
    parameters["objective"] = "multi:softprob"
    parameters["eval_metric"] = "merror"
    parameters["num_class"] = n_classes

    out = dxgb.train(
        client,
        parameters,
        dtrain=dtrain,
        evals=[(dtrain, "validation")],
        num_boost_round=2,
    )
    predictions = dxgb.predict(client=client, model=out, data=dtrain)
    assert predictions.shape[1] == n_classes
    predictions = predictions.compute()
    _check_outputs(out, predictions)

    # train has more rows than evals
    valid = dtrain
    kRows += 1
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.randint(low=0, high=n_classes, size=kRows))
    dtrain = dxgb.DaskDMatrix(client, X, y)

    out = dxgb.train(
        client,
        parameters,
        dtrain=dtrain,
        evals=[(valid, "validation")],
        num_boost_round=2,
    )
    predictions = dxgb.predict(client=client, model=out, data=valid).compute()
    _check_outputs(out, predictions)


def run_empty_dmatrix_auc(client: "Client", device: str, n_workers: int) -> None:
    from sklearn import datasets

    n_samples = 100
    n_features = 7
    rng = np.random.RandomState(1994)

    make_classification = partial(
        datasets.make_classification, n_features=n_features, random_state=rng
    )

    # binary
    X_, y_ = make_classification(n_samples=n_samples, random_state=rng)
    X = dd.from_array(X_, chunksize=10)
    y = dd.from_array(y_, chunksize=10)

    n_samples = n_workers - 1
    valid_X_, valid_y_ = make_classification(n_samples=n_samples, random_state=rng)
    valid_X = dd.from_array(valid_X_, chunksize=n_samples)
    valid_y = dd.from_array(valid_y_, chunksize=n_samples)

    cls = dxgb.DaskXGBClassifier(
        device=device, n_estimators=2, eval_metric=["auc", "aucpr"]
    )
    cls.fit(X, y, eval_set=[(valid_X, valid_y)])

    # multiclass
    X_, y_ = make_classification(
        n_samples=n_samples,
        n_classes=n_workers,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
    )
    for i in range(y_.shape[0]):
        y_[i] = i % n_workers
    X = dd.from_array(X_, chunksize=10)
    y = dd.from_array(y_, chunksize=10)

    n_samples = n_workers - 1
    valid_X_, valid_y_ = make_classification(
        n_samples=n_samples,
        n_classes=n_workers,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
    )
    for i in range(valid_y_.shape[0]):
        valid_y_[i] = i % n_workers
    valid_X = dd.from_array(valid_X_, chunksize=n_samples)
    valid_y = dd.from_array(valid_y_, chunksize=n_samples)

    # Specify base score in case if there are only two workers and one sample.
    cls = dxgb.DaskXGBClassifier(
        device=device, n_estimators=2, eval_metric=["auc", "aucpr"], base_score=0.5
    )
    cls.fit(X, y, eval_set=[(valid_X, valid_y)])


def test_empty_dmatrix_auc() -> None:
    with LocalCluster(n_workers=4, dashboard_address=":0") as cluster:
        with Client(cluster) as client:
            run_empty_dmatrix_auc(client, "cpu", 4)


def run_auc(client: "Client", device: str) -> None:
    from sklearn import datasets

    n_samples = 100
    n_features = 97
    rng = np.random.RandomState(1994)
    X_, y_ = datasets.make_classification(
        n_samples=n_samples, n_features=n_features, random_state=rng
    )
    X = dd.from_array(X_, chunksize=10)
    y = dd.from_array(y_, chunksize=10)

    valid_X_, valid_y_ = datasets.make_classification(
        n_samples=n_samples, n_features=n_features, random_state=rng
    )
    valid_X = dd.from_array(valid_X_, chunksize=10)
    valid_y = dd.from_array(valid_y_, chunksize=10)

    cls = xgb.XGBClassifier(device=device, n_estimators=2, eval_metric="auc")
    cls.fit(X_, y_, eval_set=[(valid_X_, valid_y_)])

    dcls = dxgb.DaskXGBClassifier(device=device, n_estimators=2, eval_metric="auc")
    dcls.fit(X, y, eval_set=[(valid_X, valid_y)])

    approx = dcls.evals_result()["validation_0"]["auc"]
    exact = cls.evals_result()["validation_0"]["auc"]
    for i in range(2):
        # approximated test.
        assert np.abs(approx[i] - exact[i]) <= 0.06


def test_auc(client: "Client") -> None:
    run_auc(client, "cpu")


# No test for Exact, as empty DMatrix handling are mostly for distributed
# environment and Exact doesn't support it.
@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_empty_dmatrix(tree_method) -> None:
    with LocalCluster(n_workers=kWorkers, dashboard_address=":0") as cluster:
        with Client(cluster) as client:
            parameters = {"tree_method": tree_method}
            run_empty_dmatrix_reg(client, parameters)
            run_empty_dmatrix_cls(client, parameters)
            parameters = {"tree_method": tree_method, "objective": "reg:absoluteerror"}
            run_empty_dmatrix_reg(client, parameters)


async def run_from_dask_array_asyncio(scheduler_address: str) -> dxgb.TrainReturnT:
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y, _ = generate_array()
        m = await DaskDMatrix(client, X, y)  # type: ignore
        output = await dxgb.train(client, {}, dtrain=m)

        with_m = await dxgb.predict(client, output, m)
        with_X = await dxgb.predict(client, output, X)
        inplace = await dxgb.inplace_predict(client, output, X)
        assert isinstance(with_m, da.Array)
        assert isinstance(with_X, da.Array)
        assert isinstance(inplace, da.Array)

        np.testing.assert_allclose(
            await client.compute(with_m), await client.compute(with_X)
        )
        np.testing.assert_allclose(
            await client.compute(with_m), await client.compute(inplace)
        )
    return output


async def run_dask_regressor_asyncio(scheduler_address: str) -> None:
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y, _ = generate_array()
        regressor = await dxgb.DaskXGBRegressor(verbosity=1, n_estimators=2)
        regressor.set_params(tree_method="hist")
        regressor.client = client
        await regressor.fit(X, y, eval_set=[(X, y)])
        prediction = await regressor.predict(X)

        assert prediction.ndim == 1
        assert prediction.shape[0] == kRows

        history = regressor.evals_result()

        assert isinstance(prediction, da.Array)
        assert isinstance(history, dict)

        assert list(history["validation_0"].keys())[0] == "rmse"
        assert len(history["validation_0"]["rmse"]) == 2

        awaited = await client.compute(prediction)
        assert awaited.shape[0] == kRows


async def run_dask_classifier_asyncio(scheduler_address: str) -> None:
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y, _ = generate_array()
        y = (y * 10).astype(np.int32)
        classifier = await dxgb.DaskXGBClassifier(
            verbosity=1, n_estimators=2, eval_metric="merror"
        )
        classifier.client = client
        await classifier.fit(X, y, eval_set=[(X, y)])
        prediction = await classifier.predict(X)

        assert prediction.ndim == 1
        assert prediction.shape[0] == kRows

        history = classifier.evals_result()

        assert isinstance(prediction, da.Array)
        assert isinstance(history, dict)

        assert list(history.keys())[0] == "validation_0"
        assert list(history["validation_0"].keys())[0] == "merror"
        assert len(list(history["validation_0"])) == 1
        assert len(history["validation_0"]["merror"]) == 2

        # Test .predict_proba()
        probas = await classifier.predict_proba(X)
        assert classifier.n_classes_ == 10
        assert probas.ndim == 2
        assert probas.shape[0] == kRows
        assert probas.shape[1] == 10

        # Test with dataframe.
        X_d = dd.from_dask_array(X)
        y_d = dd.from_dask_array(y)
        await classifier.fit(X_d, y_d)

        assert classifier.n_classes_ == 10
        prediction = await client.compute(await classifier.predict(X_d))

        assert prediction.ndim == 1
        assert prediction.shape[0] == kRows


def test_with_asyncio() -> None:
    with LocalCluster(n_workers=2, dashboard_address=":0") as cluster:
        with Client(cluster) as client:
            address = client.scheduler.address
            output = asyncio.run(run_from_dask_array_asyncio(address))
            assert isinstance(output["booster"], xgb.Booster)
            assert isinstance(output["history"], dict)

            asyncio.run(run_dask_regressor_asyncio(address))
            asyncio.run(run_dask_classifier_asyncio(address))


async def generate_concurrent_trainings() -> None:
    async def train() -> None:
        async with LocalCluster(
            n_workers=2, threads_per_worker=1, asynchronous=True, dashboard_address=":0"
        ) as cluster:
            async with Client(cluster, asynchronous=True) as client:
                X, y, w = generate_array(with_weights=True)
                dtrain = await DaskDMatrix(client, X, y, weight=w)  # type: ignore
                dvalid = await DaskDMatrix(client, X, y, weight=w)  # type: ignore
                output = await dxgb.train(client, {}, dtrain=dtrain)
                await dxgb.predict(client, output, data=dvalid)

    await asyncio.gather(train(), train())


def test_concurrent_trainings() -> None:
    asyncio.run(generate_concurrent_trainings())


def test_predict(client: "Client") -> None:
    X, y, _ = generate_array()
    dtrain = DaskDMatrix(client, X, y)
    booster = dxgb.train(client, {}, dtrain, num_boost_round=2)["booster"]

    predt_0 = dxgb.predict(client, model=booster, data=dtrain)
    assert predt_0.ndim == 1
    assert predt_0.shape[0] == kRows

    margin = dxgb.predict(client, model=booster, data=dtrain, output_margin=True)
    assert margin.ndim == 1
    assert margin.shape[0] == kRows

    shap = dxgb.predict(client, model=booster, data=dtrain, pred_contribs=True)
    assert shap.ndim == 2
    assert shap.shape[0] == kRows
    assert shap.shape[1] == kCols + 1

    booster_f = client.scatter(booster, broadcast=True)

    predt_1 = dxgb.predict(client, booster_f, X).compute()
    predt_2 = dxgb.inplace_predict(client, booster_f, X).compute()
    np.testing.assert_allclose(predt_0, predt_1)
    np.testing.assert_allclose(predt_0, predt_2)


def test_predict_with_meta(client: "Client") -> None:
    X, y, w = generate_array(with_weights=True)
    assert w is not None
    partition_size = 20
    margin = da.random.random(kRows, partition_size) + 1e4

    dtrain = DaskDMatrix(client, X, y, weight=w, base_margin=margin)
    booster: xgb.Booster = dxgb.train(client, {}, dtrain, num_boost_round=4)["booster"]

    prediction = dxgb.predict(client, model=booster, data=dtrain)
    assert prediction.ndim == 1
    assert prediction.shape[0] == kRows

    prediction = client.compute(prediction).result()
    assert np.all(prediction > 1e3)

    m = xgb.DMatrix(X.compute())
    m.set_info(label=y.compute(), weight=w.compute(), base_margin=margin.compute())
    single = booster.predict(m)  # Make sure the ordering is correct.
    assert np.all(prediction == single)


def run_aft_survival(client: "Client", dmatrix_t: Type) -> None:
    df = dd.read_csv(os.path.join(tm.data_dir(__file__), "veterans_lung_cancer.csv"))
    y_lower_bound = df["Survival_label_lower_bound"]
    y_upper_bound = df["Survival_label_upper_bound"]
    X = df.drop(["Survival_label_lower_bound", "Survival_label_upper_bound"], axis=1)
    m = dmatrix_t(
        client, X, label_lower_bound=y_lower_bound, label_upper_bound=y_upper_bound
    )
    base_params = {
        "verbosity": 0,
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "learning_rate": 0.05,
        "aft_loss_distribution_scale": 1.20,
        "max_depth": 6,
        "lambda": 0.01,
        "alpha": 0.02,
    }

    nloglik_rec = {}
    dists = ["normal", "logistic", "extreme"]
    for dist in dists:
        params = base_params
        params.update({"aft_loss_distribution": dist})
        evals_result = {}
        out = dxgb.train(client, params, m, num_boost_round=100, evals=[(m, "train")])
        evals_result = out["history"]
        nloglik_rec[dist] = evals_result["train"]["aft-nloglik"]
        # AFT metric (negative log likelihood) improve monotonically
        assert all(p >= q for p, q in zip(nloglik_rec[dist], nloglik_rec[dist][:1]))
    # For this data, normal distribution works the best
    assert nloglik_rec["normal"][-1] < 4.9
    assert nloglik_rec["logistic"][-1] > 4.9
    assert nloglik_rec["extreme"][-1] > 4.9


def test_dask_aft_survival() -> None:
    with LocalCluster(n_workers=kWorkers, dashboard_address=":0") as cluster:
        with Client(cluster) as client:
            run_aft_survival(client, DaskDMatrix)


@pytest.mark.parametrize("booster", ["dart", "gbtree"])
def test_dask_predict_leaf(booster: str, client: "Client") -> None:
    from sklearn.datasets import load_digits

    X_, y_ = load_digits(return_X_y=True)
    num_parallel_tree = 4
    X, y = dd.from_array(X_, chunksize=32), dd.from_array(y_, chunksize=32)
    rounds = 4
    cls = dxgb.DaskXGBClassifier(
        n_estimators=rounds, num_parallel_tree=num_parallel_tree, booster=booster
    )
    cls.client = client
    cls.fit(X, y)
    leaf = dxgb.predict(
        client,
        cls.get_booster(),
        X.to_dask_array(),  # we can't map_blocks on dataframe when output is 4-dim.
        pred_leaf=True,
        strict_shape=True,
        validate_features=False,
    ).compute()

    assert leaf.shape[0] == X_.shape[0]
    assert leaf.shape[1] == rounds
    assert leaf.shape[2] == cls.n_classes_
    assert leaf.shape[3] == num_parallel_tree

    leaf_from_apply = cls.apply(X).reshape(leaf.shape).compute()
    np.testing.assert_allclose(leaf_from_apply, leaf)

    validate_leaf_output(leaf, num_parallel_tree)


def test_dask_iteration_range(client: "Client"):
    X, y, _ = generate_array()
    n_rounds = 10

    Xy = xgb.DMatrix(X.compute(), y.compute())

    dXy = dxgb.DaskDMatrix(client, X, y)
    booster = dxgb.train(
        client, {"tree_method": "hist"}, dXy, num_boost_round=n_rounds
    )["booster"]

    for i in range(0, n_rounds):
        iter_range = (0, i)
        native_predt = booster.predict(Xy, iteration_range=iter_range)

        with_dask_dmatrix = dxgb.predict(
            client, booster, dXy, iteration_range=iter_range
        )
        with_dask_collection = dxgb.predict(
            client, booster, X, iteration_range=iter_range
        )
        with_inplace = dxgb.inplace_predict(
            client, booster, X, iteration_range=iter_range
        )
        np.testing.assert_allclose(native_predt, with_dask_dmatrix.compute())
        np.testing.assert_allclose(native_predt, with_dask_collection.compute())
        np.testing.assert_allclose(native_predt, with_inplace.compute())

    full_predt = dxgb.predict(client, booster, X, iteration_range=(0, n_rounds))
    default = dxgb.predict(client, booster, X)
    np.testing.assert_allclose(full_predt.compute(), default.compute())


def test_killed_task_wo_hang():
    # Test that aborting a worker doesn't lead to hang.
    class Eve(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch: int, evals_log) -> bool:
            if coll.get_rank() == 1:
                os.abort()
            return False

    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster) as client:
            X, y, _ = generate_array()
            n_rounds = 10
            dXy = dxgb.DaskDMatrix(client, X, y)
            # The precise error message depends on Dask scheduler.
            try:
                dxgb.train(
                    client,
                    {"tree_method": "hist"},
                    dXy,
                    num_boost_round=n_rounds,
                    callbacks=[Eve()],
                )
            except (ValueError, KilledWorker):
                pass


def test_invalid_config(client: "Client") -> None:
    X, y, _ = generate_array()
    dtrain = DaskDMatrix(client, X, y)

    with dask.config.set({"xgboost.foo": "bar"}):
        with pytest.raises(ValueError, match=r"Unknown configuration.*"):
            dxgb.train(client, {}, dtrain, num_boost_round=4)

    with dask.config.set({"xgboost.scheduler_address": "127.0.0.1:foo"}):
        with pytest.raises(socket.gaierror, match=r".*not known.*"):
            dxgb.train(client, {}, dtrain, num_boost_round=1)

    # No failure only because we are also using the Dask scheduler address.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        cfg = CollConfig(tracker_host_ip="127.0.0.1", tracker_port=port)
        dxgb.train(client, {}, dtrain, num_boost_round=1, coll_cfg=cfg)

    with pytest.raises(ValueError, match=r"comm_group.*timeout >= 0.*"):
        cfg = CollConfig(tracker_host_ip="127.0.0.1", tracker_port=0, timeout=-1)
        dxgb.train(client, {}, dtrain, num_boost_round=1, coll_cfg=cfg)


class TestWithDask:
    def test_dmatrix_binary(self, client: "Client") -> None:
        def save_dmatrix(rabit_args: Dict[str, Union[int, str]], tmpdir: str) -> None:
            with dxgb.CommunicatorContext(**rabit_args):
                rank = xgb.collective.get_rank()
                X, y = tm.make_categorical(100, 4, 4, onehot=False)
                Xy = xgb.DMatrix(X, y, enable_categorical=True)
                path = os.path.join(tmpdir, f"{rank}.bin")
                Xy.save_binary(path)

        def load_dmatrix(rabit_args: Dict[str, Union[int, str]], tmpdir: str) -> None:
            with dxgb.CommunicatorContext(**rabit_args):
                rank = xgb.collective.get_rank()
                path = os.path.join(tmpdir, f"{rank}.bin")
                Xy = xgb.DMatrix(path)
                assert Xy.num_row() == 100
                assert Xy.num_col() == 4

        with tempfile.TemporaryDirectory() as tmpdir:
            workers = tm.dask.get_client_workers(client)
            rabit_args = get_rabit_args(client, len(workers))
            futures = []
            for w in workers:
                # same argument for each worker, must set pure to False otherwise dask
                # will try to reuse the result from the first worker and hang waiting
                # for it.
                f = client.submit(
                    save_dmatrix, rabit_args, tmpdir, workers=[w], pure=False
                )
                futures.append(f)
            client.gather(futures)

            rabit_args = get_rabit_args(client, len(workers))
            futures = []
            for w in workers:
                f = client.submit(
                    load_dmatrix, rabit_args, tmpdir, workers=[w], pure=False
                )
                futures.append(f)
            client.gather(futures)

    @pytest.mark.parametrize(
        "config_key,config_value", [("verbosity", 0), ("use_rmm", True)]
    )
    def test_global_config(
        self, client: "Client", config_key: str, config_value: Any
    ) -> None:
        X, y, _ = generate_array()
        xgb.config.set_config(**{config_key: config_value})
        dtrain = DaskDMatrix(client, X, y)
        before_fname = "./before_training-test_global_config"
        after_fname = "./after_training-test_global_config"

        class TestCallback(xgb.callback.TrainingCallback):
            def write_file(self, fname: str) -> None:
                with open(fname, "w") as fd:
                    fd.write(str(xgb.config.get_config()[config_key]))

            def before_training(self, model: xgb.Booster) -> xgb.Booster:
                self.write_file(before_fname)
                assert xgb.config.get_config()[config_key] == config_value
                return model

            def after_training(self, model: xgb.Booster) -> xgb.Booster:
                assert xgb.config.get_config()[config_key] == config_value
                return model

            def before_iteration(
                self, model: xgb.Booster, epoch: int, evals_log: Dict
            ) -> bool:
                assert xgb.config.get_config()[config_key] == config_value
                return False

            def after_iteration(
                self, model: xgb.Booster, epoch: int, evals_log: Dict
            ) -> bool:
                self.write_file(after_fname)
                assert xgb.config.get_config()[config_key] == config_value
                return False

        dxgb.train(client, {}, dtrain, num_boost_round=4, callbacks=[TestCallback()])[
            "booster"
        ]

        with open(before_fname, "r") as before, open(after_fname, "r") as after:
            assert before.read() == str(config_value)
            assert after.read() == str(config_value)

        os.remove(before_fname)
        os.remove(after_fname)

    def run_updater_test(
        self,
        client: "Client",
        params: Dict,
        num_rounds: int,
        dataset: tm.TestDataset,
        tree_method: str,
    ) -> None:
        params["tree_method"] = tree_method
        params["debug_synchronize"] = True
        params = dataset.set_params(params)

        # It doesn't make sense to distribute a completely empty dataset.
        assume(dataset.X.shape[0] != 0)

        chunk = 128
        y_chunk = chunk if len(dataset.y.shape) == 1 else (chunk, dataset.y.shape[1])
        X = da.from_array(dataset.X, chunks=(chunk, dataset.X.shape[1]))
        y = da.from_array(dataset.y, chunks=y_chunk)
        if dataset.w is not None:
            w = da.from_array(dataset.w, chunks=(chunk,))
        else:
            w = None

        m = dxgb.DaskDMatrix(client, data=X, label=y, weight=w)
        history = dxgb.train(
            client,
            params=params,
            dtrain=m,
            num_boost_round=num_rounds,
            evals=[(m, "train")],
        )["history"]
        note(str(history))
        history = history["train"][dataset.metric]

        def is_stump():
            return (
                params.get("max_depth", None) == 1
                or params.get("max_leaves", None) == 1
            )

        def minimum_bin() -> bool:
            return "max_bin" in params and params["max_bin"] == 2

        # See note on `ObjFunction::UpdateTreeLeaf`.
        update_leaf = dataset.name.endswith("-l1")
        if update_leaf and (is_stump() or minimum_bin()):
            assert tm.non_increasing(history, tolerance=1e-2)
            return
        elif minimum_bin() and is_stump():
            assert tm.non_increasing(history, tolerance=1e-3)
        else:
            assert tm.non_increasing(history)
        # Make sure that it's decreasing
        if is_stump():
            # we might have already got the best score with base_score.
            assert history[-1] <= history[0] + 1e-3
        else:
            assert history[-1] < history[0]

    @given(
        params=hist_parameter_strategy,
        cache_param=hist_cache_strategy,
        dataset=tm.make_dataset_strategy(),
    )
    @settings(
        deadline=None, max_examples=10, suppress_health_check=suppress, print_blob=True
    )
    def test_hist(
        self,
        params: Dict[str, Any],
        cache_param: Dict[str, Any],
        dataset: tm.TestDataset,
        client: "Client",
    ) -> None:
        num_rounds = 10
        params.update(cache_param)
        self.run_updater_test(client, params, num_rounds, dataset, "hist")

    def test_quantile_dmatrix(self, client: Client) -> None:
        X, y = make_categorical(client, 10000, 30, 13)

        Xy = dxgb.DaskDMatrix(client, X, y, enable_categorical=True)
        valid_Xy = dxgb.DaskDMatrix(client, X, y, enable_categorical=True)

        output = dxgb.train(
            client,
            {"tree_method": "hist"},
            Xy,
            num_boost_round=10,
            evals=[(Xy, "Train"), (valid_Xy, "Valid")],
        )
        dmatrix_hist = output["history"]

        Xy = dxgb.DaskQuantileDMatrix(client, X, y, enable_categorical=True)
        valid_Xy = dxgb.DaskQuantileDMatrix(
            client, X, y, enable_categorical=True, ref=Xy
        )

        output = dxgb.train(
            client,
            {"tree_method": "hist"},
            Xy,
            num_boost_round=10,
            evals=[(Xy, "Train"), (valid_Xy, "Valid")],
        )
        quantile_hist = output["history"]

        np.testing.assert_allclose(
            quantile_hist["Train"]["rmse"], dmatrix_hist["Train"]["rmse"]
        )
        np.testing.assert_allclose(
            quantile_hist["Valid"]["rmse"], dmatrix_hist["Valid"]["rmse"]
        )

    def test_empty_quantile_dmatrix(self, client: Client) -> None:
        X, y = make_categorical(client, 2, 30, 13)
        X_valid, y_valid = make_categorical(client, 10000, 30, 13)

        Xy = dxgb.DaskQuantileDMatrix(client, X, y, enable_categorical=True)
        Xy_valid = dxgb.DaskQuantileDMatrix(
            client, X_valid, y_valid, ref=Xy, enable_categorical=True
        )
        result = dxgb.train(
            client,
            {"tree_method": "hist"},
            Xy,
            num_boost_round=10,
            evals=[(Xy_valid, "Valid")],
        )
        predt = dxgb.inplace_predict(client, result["booster"], X).compute()
        np.testing.assert_allclose(y.compute(), predt)
        rmse = result["history"]["Valid"]["rmse"][-1]
        assert rmse < 32.0

    @given(
        params=hist_parameter_strategy,
        cache_param=hist_cache_strategy,
        dataset=tm.make_dataset_strategy(),
    )
    @settings(
        deadline=None, max_examples=10, suppress_health_check=suppress, print_blob=True
    )
    def test_approx(
        self,
        client: "Client",
        params: Dict,
        cache_param: Dict[str, Any],
        dataset: tm.TestDataset,
    ) -> None:
        num_rounds = 10
        params.update(cache_param)
        self.run_updater_test(client, params, num_rounds, dataset, "approx")

    def test_adaptive(self) -> None:
        def get_score(config: Dict) -> float:
            return float(config["learner"]["learner_model_param"]["base_score"])

        def local_test(rabit_args: Dict[str, Union[int, str]], worker_id: int) -> bool:
            with dxgb.CommunicatorContext(**rabit_args):
                if worker_id == 0:
                    y = np.array([0.0, 0.0, 0.0])
                    x = np.array([[0.0]] * 3)
                else:
                    y = np.array([1000.0])
                    x = np.array(
                        [
                            [0.0],
                        ]
                    )

                Xy = xgb.DMatrix(x, y)
                booster = xgb.train(
                    {"tree_method": "hist", "objective": "reg:absoluteerror"},
                    Xy,
                    num_boost_round=1,
                )
                config = json.loads(booster.save_config())
                base_score = get_score(config)
                assert base_score == 250.0
                return True

        with LocalCluster(n_workers=2, dashboard_address=":0") as cluster:
            with Client(cluster) as client:
                workers = tm.dask.get_client_workers(client)
                rabit_args = get_rabit_args(client, len(workers))
                futures = []
                for i, _ in enumerate(workers):
                    f = client.submit(local_test, rabit_args, i)
                    futures.append(f)

                results = client.gather(futures)
                assert all(results)

    def test_n_workers(self) -> None:
        with LocalCluster(n_workers=2, dashboard_address=":0") as cluster:
            with Client(cluster) as client:
                workers = tm.dask.get_client_workers(client)
                from sklearn.datasets import load_breast_cancer

                X, y = load_breast_cancer(return_X_y=True)
                dX = client.submit(da.from_array, X, workers=[workers[0]]).result()
                dy = client.submit(da.from_array, y, workers=[workers[0]]).result()
                train = dxgb.DaskDMatrix(client, dX, dy)

                dX = dd.from_array(X)
                dX = client.persist(dX, workers=workers[1])
                dy = dd.from_array(y)
                dy = client.persist(dy, workers=workers[1])
                valid = dxgb.DaskDMatrix(client, dX, dy)

                merged = dxgb._get_workers_from_data(train, evals=[(valid, "Valid")])
                assert len(merged) == 2

    @pytest.mark.skipif(**tm.no_dask())
    def test_feature_weights(self, client: "Client") -> None:
        kRows = 1024
        kCols = 64
        rng = da.random.RandomState(1994)
        X = rng.random_sample((kRows, kCols), chunks=(32, -1))
        y = rng.random_sample(kRows, chunks=32)

        fw = np.ones(shape=(kCols,))
        for i in range(kCols):
            fw[i] *= float(i)
        fw = da.from_array(fw)
        parser = os.path.join(tm.demo_dir(__file__), "guide-python", "model_parser.py")
        poly_increasing = get_feature_weights(
            X=X,
            y=y,
            fw=fw,
            parser_path=parser,
            tree_method="approx",
            model=dxgb.DaskXGBRegressor,
        )

        fw = np.ones(shape=(kCols,))
        for i in range(kCols):
            fw[i] *= float(kCols - i)
        fw = da.from_array(fw)
        poly_decreasing = get_feature_weights(
            X=X,
            y=y,
            fw=fw,
            parser_path=parser,
            tree_method="approx",
            model=dxgb.DaskXGBRegressor,
        )

        # Approxmated test, this is dependent on the implementation of random
        # number generator in std library.
        assert poly_increasing[0] > 0.08
        assert poly_decreasing[0] < -0.08

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_custom_objective(self, client: "Client") -> None:
        from sklearn.datasets import fetch_california_housing

        X, y = fetch_california_housing(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)
        rounds = 20

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log")

            def sqr(
                labels: np.ndarray, predts: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
                with open(path, "a") as fd:
                    print("Running sqr", file=fd)
                grad = predts - labels
                hess = np.ones(shape=labels.shape[0])
                return grad, hess

            reg = dxgb.DaskXGBRegressor(
                n_estimators=rounds, objective=sqr, tree_method="hist"
            )
            reg.fit(X, y, eval_set=[(X, y)])

            # Check the obj is ran for rounds.
            with open(path, "r") as fd:
                out = fd.readlines()
                assert len(out) == rounds

            results_custom = reg.evals_result()

            reg = dxgb.DaskXGBRegressor(
                n_estimators=rounds, tree_method="hist", base_score=0.5
            )
            reg.fit(X, y, eval_set=[(X, y)])
            results_native = reg.evals_result()

            np.testing.assert_allclose(
                results_custom["validation_0"]["rmse"],
                results_native["validation_0"]["rmse"],
            )
            tm.non_increasing(results_native["validation_0"]["rmse"])

            reg = dxgb.DaskXGBRegressor(
                n_estimators=rounds, objective=tm.ls_obj, tree_method="hist"
            )
            rng = da.random.RandomState(1994)
            w = rng.uniform(low=0.0, high=1.0, size=y.shape[0])
            reg.fit(
                X, y, sample_weight=w, eval_set=[(X, y)], sample_weight_eval_set=[w]
            )
            results_custom = reg.evals_result()
            tm.non_increasing(results_custom["validation_0"]["rmse"])

    def test_no_duplicated_partition(self) -> None:
        """Assert each worker has the correct amount of data, and DMatrix initialization
        doesn't generate unnecessary copies of data.

        """
        with LocalCluster(n_workers=2, dashboard_address=":0") as cluster:
            with Client(cluster) as client:
                X, y, _ = generate_array()
                n_partitions = X.npartitions
                m = dxgb.DaskDMatrix(client, X, y)
                workers = tm.dask.get_client_workers(client)
                rabit_args = get_rabit_args(client, len(workers))
                n_workers = len(workers)

                def worker_fn(worker_addr: str, data_ref: Dict) -> None:
                    with dxgb.CommunicatorContext(**rabit_args):
                        local_dtrain = dxgb._dmatrix_from_list_of_parts(
                            **data_ref, nthread=7
                        )
                        total = np.array([local_dtrain.num_row()])
                        total = xgb.collective.allreduce(total, xgb.collective.Op.SUM)
                        assert total[0] == kRows

                futures = []
                for i in range(len(workers)):
                    futures.append(
                        client.submit(
                            worker_fn,
                            workers[i],
                            m._create_fn_args(workers[i]),
                            pure=False,
                            workers=[workers[i]],
                        )
                    )
                client.gather(futures)

                has_what = client.has_what()
                cnt = 0
                data = set()
                for k, v in has_what.items():
                    for d in v:
                        cnt += 1
                        data.add(d)

                assert len(data) == cnt
                # Subtract the on disk resource from each worker
                assert cnt - n_workers == n_partitions

    def test_data_initialization(self, client: "Client") -> None:
        """assert that we don't create duplicated DMatrix"""
        from sklearn.datasets import load_digits

        X, y = load_digits(return_X_y=True)
        X, y = dd.from_array(X, chunksize=32), dd.from_array(y, chunksize=32)
        validate_data_initialization(
            dxgb.DaskQuantileDMatrix, dxgb.DaskXGBClassifier, X, y
        )

    def run_shap(
        self, X: Any, y: Any, params: Dict[str, Any], client: "Client"
    ) -> None:
        rows = X.shape[0]
        cols = X.shape[1]

        def assert_shape(shape: Tuple[int, ...]) -> None:
            assert shape[0] == rows
            if "num_class" in params.keys():
                assert shape[1] == params["num_class"]
                assert shape[2] == cols + 1
            else:
                assert shape[1] == cols + 1

        X, y = da.from_array(X, chunks=(32, -1)), da.from_array(y, chunks=32)
        Xy = dxgb.DaskDMatrix(client, X, y)
        booster = dxgb.train(client, params, Xy, num_boost_round=10)["booster"]

        test_Xy = dxgb.DaskDMatrix(client, X, y)

        shap = dxgb.predict(client, booster, test_Xy, pred_contribs=True).compute()
        margin = dxgb.predict(client, booster, test_Xy, output_margin=True).compute()
        assert_shape(shap.shape)
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-5, 1e-5)

        shap = dxgb.predict(client, booster, X, pred_contribs=True).compute()
        margin = dxgb.predict(client, booster, X, output_margin=True).compute()
        assert_shape(shap.shape)
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-5, 1e-5)

        if "num_class" not in params.keys():
            X = dd.from_dask_array(X).repartition(npartitions=32)
            y = dd.from_dask_array(y).repartition(npartitions=32)
            shap_df = dxgb.predict(
                client, booster, X, pred_contribs=True, validate_features=False
            ).compute()
            assert_shape(shap_df.shape)
            assert np.allclose(
                np.sum(shap_df, axis=len(shap_df.shape) - 1), margin, 1e-5, 1e-5
            )

    def run_shap_cls_sklearn(self, X: Any, y: Any, client: "Client") -> None:
        X, y = da.from_array(X, chunks=(32, -1)), da.from_array(y, chunks=32)
        cls = dxgb.DaskXGBClassifier(n_estimators=4)
        cls.client = client
        cls.fit(X, y)
        booster = cls.get_booster()

        test_Xy = dxgb.DaskDMatrix(client, X, y)

        shap = dxgb.predict(client, booster, test_Xy, pred_contribs=True).compute()
        margin = dxgb.predict(client, booster, test_Xy, output_margin=True).compute()
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-5, 1e-5)

        shap = dxgb.predict(client, booster, X, pred_contribs=True).compute()
        margin = dxgb.predict(client, booster, X, output_margin=True).compute()
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-5, 1e-5)

    def test_shap(self, client: "Client") -> None:
        from sklearn.datasets import load_diabetes, load_iris

        X, y = load_diabetes(return_X_y=True)
        params: Dict[str, Any] = {"objective": "reg:squarederror"}
        self.run_shap(X, y, params, client)

        X, y = load_iris(return_X_y=True)
        params = {"objective": "multi:softmax", "num_class": 3}
        self.run_shap(X, y, params, client)

        params = {"objective": "multi:softprob", "num_class": 3}
        self.run_shap(X, y, params, client)

        self.run_shap_cls_sklearn(X, y, client)

    def run_shap_interactions(
        self, X: Any, y: Any, params: Dict[str, Any], client: "Client"
    ) -> None:
        rows = X.shape[0]
        cols = X.shape[1]
        X, y = da.from_array(X, chunks=(32, -1)), da.from_array(y, chunks=32)

        Xy = dxgb.DaskDMatrix(client, X, y)
        booster = dxgb.train(client, params, Xy, num_boost_round=10)["booster"]

        test_Xy = dxgb.DaskDMatrix(client, X, y)

        shap = dxgb.predict(client, booster, test_Xy, pred_interactions=True).compute()

        assert len(shap.shape) == 3
        assert shap.shape[0] == rows
        assert shap.shape[1] == cols + 1
        assert shap.shape[2] == cols + 1

        margin = dxgb.predict(client, booster, test_Xy, output_margin=True).compute()
        assert np.allclose(
            np.sum(shap, axis=(len(shap.shape) - 1, len(shap.shape) - 2)),
            margin,
            1e-5,
            1e-5,
        )

    def test_shap_interactions(self, client: "Client") -> None:
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True)
        params = {"objective": "reg:squarederror"}
        self.run_shap_interactions(X, y, params, client)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_sklearn_io(self, client: "Client") -> None:
        from sklearn.datasets import load_digits

        X_, y_ = load_digits(return_X_y=True)
        X, y = da.from_array(X_), da.from_array(y_)
        cls = dxgb.DaskXGBClassifier(n_estimators=10)
        cls.client = client
        cls.fit(X, y)
        predt_0 = cls.predict(X)
        proba_0 = cls.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            with open(path, "wb") as fd:
                pickle.dump(cls, fd)

            with open(path, "rb") as fd:
                cls = pickle.load(fd)
            predt_1 = cls.predict(X)
            proba_1 = cls.predict_proba(X)
            np.testing.assert_allclose(predt_0.compute(), predt_1.compute())
            np.testing.assert_allclose(proba_0.compute(), proba_1.compute())

            path = os.path.join(tmpdir, "cls.json")
            cls.save_model(path)

            cls = dxgb.DaskXGBClassifier()
            cls.load_model(path)
            assert cls.n_classes_ == 10
            predt_2 = cls.predict(X)
            proba_2 = cls.predict_proba(X)

            np.testing.assert_allclose(predt_0.compute(), predt_2.compute())
            np.testing.assert_allclose(proba_0.compute(), proba_2.compute())

            # Use single node to load
            cls = xgb.XGBClassifier()
            cls.load_model(path)
            assert cls.n_classes_ == 10
            predt_3 = cls.predict(X_)
            proba_3 = cls.predict_proba(X_)

            np.testing.assert_allclose(predt_0.compute(), predt_3)
            np.testing.assert_allclose(proba_0.compute(), proba_3)


def test_dask_unsupported_features(client: "Client") -> None:
    X, y, _ = generate_array()
    # gblinear doesn't support distributed training.
    with pytest.raises(NotImplementedError, match="gblinear"):
        dxgb.train(client, {"booster": "gblinear"}, dxgb.DaskDMatrix(client, X, y))


def test_parallel_submits(client: "Client") -> None:
    """Test for running multiple train simultaneously from single clients."""
    try:
        from distributed import MultiLock  # NOQA
    except ImportError:
        pytest.skip("`distributed.MultiLock' is not available")

    from sklearn.datasets import load_digits

    futures = []
    workers = tm.dask.get_client_workers(client)
    n_submits = len(workers)
    for i in range(n_submits):
        X_, y_ = load_digits(return_X_y=True)
        X = dd.from_array(X_, chunksize=32)
        y = dd.from_array(y_, chunksize=32)
        cls = dxgb.DaskXGBClassifier(
            verbosity=1,
            n_estimators=i + 1,
            eval_metric="merror",
        )
        f = client.submit(cls.fit, X, y, pure=False)
        futures.append(f)

    classifiers = client.gather(futures)
    assert len(classifiers) == n_submits
    for i, cls in enumerate(classifiers):
        assert cls.get_booster().num_boosted_rounds() == i + 1


def run_tree_stats(client: Client, tree_method: str, device: str) -> str:
    """assert that different workers count dosn't affect summ statistic's on root"""

    def dask_train(X, y, num_obs, num_features):
        chunk_size = 100
        X = da.from_array(X, chunks=(chunk_size, num_features))
        y = da.from_array(y.reshape(num_obs, 1), chunks=(chunk_size, 1))
        dtrain = dxgb.DaskDMatrix(client, X, y)

        output = dxgb.train(
            client,
            {
                "verbosity": 0,
                "tree_method": tree_method,
                "device": device,
                "objective": "reg:squarederror",
                "max_depth": 3,
            },
            dtrain,
            num_boost_round=1,
        )
        dump_model = output["booster"].get_dump(with_stats=True, dump_format="json")[0]
        return json.loads(dump_model)

    num_obs = 1000
    num_features = 10
    X, y = make_regression(num_obs, num_features, random_state=777)
    model = dask_train(X, y, num_obs, num_features)

    # asserts children have correct cover.
    stack = [model]
    while stack:
        node: dict = stack.pop()
        if "leaf" in node.keys():
            continue
        cover = 0
        for c in node["children"]:
            cover += c["cover"]
            stack.append(c)
        assert cover == node["cover"]

    return model["cover"]


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_tree_stats(tree_method: str) -> None:
    with LocalCluster(n_workers=1, dashboard_address=":0") as cluster:
        with Client(cluster) as client:
            local = run_tree_stats(client, tree_method, "cpu")
    with LocalCluster(n_workers=2, dashboard_address=":0") as cluster:
        with Client(cluster) as client:
            distributed = run_tree_stats(client, tree_method, "cpu")

    assert local == distributed


def test_parallel_submit_multi_clients() -> None:
    """Test for running multiple train simultaneously from multiple clients."""
    try:
        from distributed import MultiLock  # NOQA
    except ImportError:
        pytest.skip("`distributed.MultiLock' is not available")

    from sklearn.datasets import load_digits

    with LocalCluster(n_workers=4, dashboard_address=":0") as cluster:
        with Client(cluster) as client:
            workers = tm.dask.get_client_workers(client)

        n_submits = len(workers)
        assert n_submits == 4
        futures = []

        for i in range(n_submits):
            client = Client(cluster)
            X_, y_ = load_digits(return_X_y=True)
            X_ += 1.0
            X = dd.from_array(X_, chunksize=32)
            y = dd.from_array(y_, chunksize=32)
            cls = dxgb.DaskXGBClassifier(
                verbosity=1,
                n_estimators=i + 1,
                eval_metric="merror",
            )
            f = client.submit(cls.fit, X, y, pure=False)
            futures.append((client, f))

        t_futures = []
        with ThreadPoolExecutor(max_workers=16) as e:
            for i in range(n_submits):

                def _() -> dxgb.DaskXGBClassifier:
                    return futures[i][0].compute(futures[i][1]).result()

                f = e.submit(_)
                t_futures.append(f)

        for i, f in enumerate(t_futures):
            assert f.result().get_booster().num_boosted_rounds() == i + 1


def test_init_estimation(client: Client) -> None:
    check_init_estimation("hist", "cpu", client)


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_uneven_nan(tree_method) -> None:
    n_workers = 2
    with LocalCluster(n_workers=n_workers) as cluster:
        with Client(cluster) as client:
            check_uneven_nan(client, tree_method, "cpu", n_workers)


class TestDaskCallbacks:
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_early_stopping(self, client: "Client") -> None:
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)
        m = dxgb.DaskDMatrix(client, X, y)

        valid = dxgb.DaskDMatrix(client, X, y)
        early_stopping_rounds = 5
        booster = dxgb.train(
            client,
            {
                "objective": "binary:logistic",
                "eval_metric": "error",
                "tree_method": "hist",
            },
            m,
            evals=[(valid, "Valid")],
            num_boost_round=1000,
            early_stopping_rounds=early_stopping_rounds,
        )["booster"]
        assert hasattr(booster, "best_score")
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

        valid_X, valid_y = load_breast_cancer(return_X_y=True)
        valid_X, valid_y = da.from_array(valid_X), da.from_array(valid_y)
        cls = dxgb.DaskXGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            n_estimators=1000,
            early_stopping_rounds=early_stopping_rounds,
        )
        cls.client = client
        cls.fit(
            X,
            y,
            eval_set=[(valid_X, valid_y)],
        )
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

        # Specify the metric
        cls = dxgb.DaskXGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            n_estimators=1000,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="error",
        )
        cls.client = client
        cls.fit(
            X,
            y,
            eval_set=[(valid_X, valid_y)],
        )
        assert tm.non_increasing(cls.evals_result()["validation_0"]["error"])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format="json")
        assert len(cls.evals_result()["validation_0"]["error"]) < 20
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_early_stopping_custom_eval(self, client: "Client") -> None:
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)
        m = dxgb.DaskDMatrix(client, X, y)

        def eval_error_metric(predt: np.ndarray, dtrain: xgb.DMatrix):
            return tm.eval_error_metric(predt, dtrain, rev_link=False)

        valid = dxgb.DaskDMatrix(client, X, y)
        early_stopping_rounds = 5
        booster = dxgb.train(
            client,
            {
                "objective": "binary:logistic",
                "eval_metric": "error",
                "tree_method": "hist",
            },
            m,
            evals=[(m, "Train"), (valid, "Valid")],
            custom_metric=eval_error_metric,
            num_boost_round=1000,
            early_stopping_rounds=early_stopping_rounds,
        )["booster"]
        assert hasattr(booster, "best_score")
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

        valid_X, valid_y = load_breast_cancer(return_X_y=True)
        valid_X, valid_y = da.from_array(valid_X), da.from_array(valid_y)
        cls = dxgb.DaskXGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            n_estimators=1000,
            eval_metric=tm.eval_error_metric_skl,
            early_stopping_rounds=early_stopping_rounds,
        )
        cls.client = client
        cls.fit(
            X,
            y,
            eval_set=[(valid_X, valid_y)],
        )
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_callback(self, client: "Client") -> None:
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            cls = dxgb.DaskXGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                n_estimators=10,
                callbacks=[
                    xgb.callback.TrainingCheckPoint(
                        directory=Path(tmpdir), interval=1, name="model"
                    )
                ],
            )
            cls.client = client
            cls.fit(
                X,
                y,
            )
            for i in range(1, 10):
                assert os.path.exists(
                    os.path.join(
                        tmpdir,
                        f"model_{i}.{xgb.callback.TrainingCheckPoint.default_format}",
                    )
                )


@gen_cluster(
    client=True,
    clean_kwargs={"processes": False, "threads": False},
    allow_unclosed=True,
)
async def test_worker_left(c: Client, s: Scheduler, a: Worker, b: Worker):
    async with Worker(s.address):
        dx = da.random.random((1000, 10)).rechunk(chunks=(10, None))
        dy = da.random.random((1000,)).rechunk(chunks=(10,))
        d_train = await dxgb.DaskDMatrix(  # type: ignore
            c,
            dx,
            dy,
        )
    await async_poll_for(lambda: len(s.workers) == 2, timeout=5)
    with pytest.raises(RuntimeError, match="Missing"):
        await dxgb.train(
            c,
            {},
            d_train,
            evals=[(d_train, "train")],
        )


@gen_cluster(
    client=True,
    Worker=Nanny,
    clean_kwargs={"processes": False, "threads": False},
    allow_unclosed=True,
)
async def test_worker_restarted(c, s, a, b):
    dx = da.random.random((1000, 10)).rechunk(chunks=(10, None))
    dy = da.random.random((1000,)).rechunk(chunks=(10,))
    d_train = await dxgb.DaskDMatrix(
        c,
        dx,
        dy,
    )
    await c.restart_workers([a.worker_address])
    with pytest.raises(RuntimeError, match="Missing"):
        await dxgb.train(
            c,
            {},
            d_train,
            evals=[(d_train, "train")],
        )


def test_doc_link() -> None:
    for est in [
        dxgb.DaskXGBRegressor(),
        dxgb.DaskXGBClassifier(),
        dxgb.DaskXGBRanker(),
        dxgb.DaskXGBRFRegressor(),
        dxgb.DaskXGBRFClassifier(),
    ]:
        name = est.__class__.__name__
        link = est._get_doc_link()
        assert f"xgboost.dask.{name}" in link
