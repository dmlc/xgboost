"""Tests for dask shared by different test modules."""

import json
from typing import Any, List, Literal, Tuple, Type, Union, cast, overload

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client, get_worker
from packaging.version import parse as parse_version
from sklearn.datasets import make_classification, make_regression

import xgboost as xgb
import xgboost.testing as tm
from xgboost.compat import concat
from xgboost.testing.updater import get_basescore

from .. import dask as dxgb
from .._typing import EvalsLog
from ..dask import TrainReturnT, _get_rabit_args
from ..dask.utils import _DASK_VERSION
from .data import make_batches
from .data import make_categorical as make_cat_local
from .multi_target import LsObj1, LsObj2
from .ordinal import make_recoded
from .utils import Device, assert_allclose


def check_init_estimation_clf(tree_method: str, device: Device, client: Client) -> None:
    """Test init estimation for classsifier."""
    X, y = make_classification(n_samples=4096 * 2, n_features=32, random_state=1994)
    clf = xgb.XGBClassifier(
        n_estimators=1, max_depth=1, tree_method=tree_method, device=device
    )
    clf.fit(X, y)
    base_score = get_basescore(clf)

    dx = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dclf = dxgb.DaskXGBClassifier(
        n_estimators=1,
        max_depth=1,
        tree_method=tree_method,
        device=device,
    )
    dclf.client = client
    dclf.fit(dx, dy)
    dbase_score = get_basescore(dclf)
    np.testing.assert_allclose(base_score, dbase_score)


def check_init_estimation_reg(tree_method: str, device: Device, client: Client) -> None:
    """Test init estimation for regressor."""
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(n_samples=4096 * 2, n_features=32, random_state=1994)
    reg = xgb.XGBRegressor(
        n_estimators=1, max_depth=1, tree_method=tree_method, device=device
    )
    reg.fit(X, y)
    base_score = get_basescore(reg)

    dx = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dreg = dxgb.DaskXGBRegressor(
        n_estimators=1, max_depth=1, tree_method=tree_method, device=device
    )
    dreg.client = client
    dreg.fit(dx, dy)
    dbase_score = get_basescore(dreg)
    np.testing.assert_allclose(base_score, dbase_score)


def check_init_estimation(tree_method: str, device: Device, client: Client) -> None:
    """Test init estimation."""
    check_init_estimation_reg(tree_method, device, client)
    check_init_estimation_clf(tree_method, device, client)


def check_uneven_nan(
    client: Client, tree_method: str, device: Device, n_workers: int
) -> None:
    """Issue #9271, not every worker has missing value."""
    assert n_workers >= 2

    with client.as_current():
        clf = dxgb.DaskXGBClassifier(tree_method=tree_method, device=device)
        X = pd.DataFrame({"a": range(10000), "b": range(10000, 0, -1)})
        y = pd.Series([*[0] * 5000, *[1] * 5000])

        X.loc[:3000:1000, "a"] = np.nan

        client.wait_for_workers(n_workers=n_workers)

        clf.fit(
            dd.from_pandas(X, npartitions=n_workers),
            dd.from_pandas(y, npartitions=n_workers),
        )


def make_multi_output_regression(
    device: Device, *, n_samples: int = 512, n_features: int = 8, n_targets: int = 3
) -> Tuple[da.Array, da.Array]:
    """Make a Dask array multi-output regression dataset for CPU or CUDA tests."""
    chunksize = 64

    X, y = make_regression(
        n_samples, n_features, n_targets=n_targets, random_state=2026
    )
    dX, dy = (
        da.from_array(X, chunks=(chunksize, n_features)),
        da.from_array(y, chunks=(chunksize, n_targets)),
    )
    if device == "cuda":
        dX, dy = dX.to_backend("cupy"), dy.to_backend("cupy")
    return dX, dy


def _as_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array)


def _train_multi_output_tree(
    client: Client, device: Device
) -> Tuple[da.Array, da.Array, dxgb.DaskDMatrix, TrainReturnT]:
    n_targets = 3
    X, y = make_multi_output_regression(device, n_targets=n_targets)
    Xy = dxgb.DaskDMatrix(client, X, y)
    result = dxgb.train(
        client,
        {
            "device": device,
            "tree_method": "hist",
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "multi_strategy": "multi_output_tree",
            "num_target": n_targets,
            "max_depth": 4,
            "max_bin": 64,
            "debug_synchronize": True,
        },
        Xy,
        num_boost_round=4,
        evals=[(Xy, "train")],
    )
    return X, y, Xy, result


def check_multi_output_tree_regressor(client: Client, device: Device) -> None:
    """Test Dask vector-leaf regression with train and sklearn-style APIs."""
    tolerance = 1e-3
    X, y, Xy, result = _train_multi_output_tree(client, device)
    n_targets = y.shape[1]
    assert isinstance(n_targets, int)

    history = result["history"]["train"]["mae"]
    assert np.isfinite(np.asarray(history)).all()
    assert tm.non_increasing(history, tolerance=tolerance)

    predt = _as_numpy(dxgb.predict(client, result["booster"], Xy).compute())
    assert predt.shape == (X.shape[0], n_targets)
    assert np.isfinite(predt).all()

    reg = dxgb.DaskXGBRegressor(
        n_estimators=4,
        device=device,
        tree_method="hist",
        objective="reg:absoluteerror",
        multi_strategy="multi_output_tree",
        max_depth=4,
        max_bin=64,
    )
    reg.client = client
    reg.fit(X, y, eval_set=[(X, y)])

    predt = _as_numpy(reg.predict(X).compute())
    assert predt.shape == (X.shape[0], n_targets)
    assert np.isfinite(predt).all()

    config = json.loads(reg.get_booster().save_config())
    assert config["learner"]["learner_train_param"]["multi_strategy"] == (
        "multi_output_tree"
    )

    for objective in (LsObj1(device), LsObj2(device, False)):
        reg = dxgb.DaskXGBRegressor(
            n_estimators=2,
            device=device,
            tree_method="hist",
            objective=objective,
            multi_strategy="multi_output_tree",
            max_depth=2,
            max_bin=64,
        )
        reg.client = client
        reg.fit(X, y)
        predt = _as_numpy(reg.predict(X).compute())
        assert predt.shape == (X.shape[0], n_targets)
        assert np.isfinite(predt).all()


def check_multi_output_tree_classifier(client: Client, device: Device) -> None:
    """Test Dask vector-leaf classification with array and dataframe labels."""
    n_targets = 3
    X, y = make_multi_output_regression(device, n_targets=n_targets)

    def check_classifier(labels: Union[da.Array, dd.DataFrame]) -> None:
        clf = dxgb.DaskXGBClassifier(
            n_estimators=2,
            device=device,
            tree_method="hist",
            objective=LsObj2(device, False),
            multi_strategy="multi_output_tree",
            max_depth=2,
            max_bin=64,
        )
        clf.client = client
        clf.fit(X, labels)

        assert isinstance(clf.classes_, np.ndarray)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))
        assert clf.n_classes_ == 2

        predt = _as_numpy(clf.predict(X).compute())
        proba = _as_numpy(clf.predict_proba(X).compute())
        assert predt.shape == (X.shape[0], n_targets)
        assert proba.shape == predt.shape
        np.testing.assert_array_equal(predt, (proba > 0.5).astype(predt.dtype))

        config = json.loads(clf.get_booster().save_config())["learner"]
        assert config["objective"]["name"] == "binary:logistic"
        assert int(config["learner_model_param"]["num_class"]) == 0

    y_ind = (y > 0.0).astype(np.int32)
    check_classifier(y_ind)
    y_df = dd.from_dask_array(y_ind)
    if device == "cuda":
        y_df = y_df.to_backend("cudf")
    check_classifier(y_df)


def check_multi_output_tree_shap(client: Client, device: Device) -> None:
    """Test SHAP output shapes for Dask vector-leaf models."""
    X, y, _, result = _train_multi_output_tree(client, device)
    n_targets = y.shape[1]
    assert isinstance(n_targets, int)
    booster = result["booster"]

    margin = _as_numpy(dxgb.predict(client, booster, X, output_margin=True).compute())
    n_features = X.shape[1]
    assert isinstance(n_features, int)
    contributions = dxgb.predict(client, booster, X, pred_contribs=True)
    contributions_shape = (X.shape[0], n_targets, n_features + 1)
    assert contributions.shape == contributions_shape
    assert contributions.chunks == (
        X.chunks[0],
        (n_targets,),
        (n_features + 1,),
    )
    contributions = _as_numpy(contributions.compute())
    assert contributions.shape == contributions_shape
    np.testing.assert_allclose(
        contributions.sum(axis=-1),
        margin,
        rtol=1e-4,
        atol=1e-4,
    )

    interactions = dxgb.predict(client, booster, X, pred_interactions=True)
    interactions_shape = (X.shape[0], n_targets, n_features + 1, n_features + 1)
    assert interactions.shape == interactions_shape
    assert interactions.chunks == (
        X.chunks[0],
        (n_targets,),
        (n_features + 1,),
        (n_features + 1,),
    )
    interactions = _as_numpy(interactions.compute())
    assert interactions.shape == interactions_shape
    np.testing.assert_allclose(
        interactions.sum(axis=(-2, -1)),
        margin,
        rtol=1e-4,
        atol=1e-4,
    )


def check_external_memory(  # pylint: disable=too-many-locals
    worker_id: int,
    n_workers: int,
    device: str,
    comm_args: dict,
    is_qdm: bool,
) -> None:
    """Basic checks for distributed external memory."""
    n_samples_per_batch = 32
    n_features = 4
    n_batches = 16
    use_cupy = device != "cpu"

    n_threads = get_worker().state.nthreads
    with xgb.collective.CommunicatorContext(dmlc_communicator="rabit", **comm_args):
        it = tm.IteratorForTest(
            *make_batches(
                n_samples_per_batch,
                n_features,
                n_batches,
                use_cupy=use_cupy,
                random_state=worker_id,
            ),
            cache="cache",
        )
        if is_qdm:
            Xy: xgb.DMatrix = xgb.ExtMemQuantileDMatrix(it, nthread=n_threads)
        else:
            Xy = xgb.DMatrix(it, nthread=n_threads)
        results: EvalsLog = {}
        xgb.train(
            {"tree_method": "hist", "nthread": n_threads, "device": device},
            Xy,
            evals=[(Xy, "Train")],
            num_boost_round=32,
            evals_result=results,
        )
        assert tm.non_increasing(cast(List[float], results["Train"]["rmse"]))

    lx, ly, lw = [], [], []
    for i in range(n_workers):
        x, y, w = make_batches(
            n_samples_per_batch,
            n_features,
            n_batches,
            use_cupy=use_cupy,
            random_state=i,
        )
        lx.extend(x)
        ly.extend(y)
        lw.extend(w)

    X = concat(lx)
    yconcat = concat(ly)
    wconcat = concat(lw)
    if is_qdm:
        Xy = xgb.QuantileDMatrix(X, yconcat, weight=wconcat, nthread=n_threads)
    else:
        Xy = xgb.DMatrix(X, yconcat, weight=wconcat, nthread=n_threads)

    results_local: EvalsLog = {}
    xgb.train(
        {"tree_method": "hist", "nthread": n_threads, "device": device},
        Xy,
        evals=[(Xy, "Train")],
        num_boost_round=32,
        evals_result=results_local,
    )
    np.testing.assert_allclose(
        results["Train"]["rmse"], results_local["Train"]["rmse"], rtol=1e-4
    )


def get_rabit_args(client: Client, n_workers: int) -> Any:
    """Get RABIT collective communicator arguments for tests."""
    return client.sync(_get_rabit_args, client, n_workers)


def get_client_workers(client: Client) -> List[str]:
    "Get workers from a dask client."
    kwargs = {"n_workers": -1} if _DASK_VERSION() >= parse_version("2025.4.0") else {}
    workers = client.scheduler_info(**kwargs)["workers"]
    return list(workers.keys())


def make_ltr(  # pylint: disable=too-many-locals,too-many-arguments
    client: Client,
    n_samples: int,
    n_features: int,
    *,
    n_query_groups: int,
    max_rel: int,
    device: str,
) -> Tuple[dd.DataFrame, dd.Series, dd.Series]:
    """Synthetic dataset for learning to rank."""
    workers = get_client_workers(client)
    n_samples_per_worker = n_samples // len(workers)

    if device == "cpu":
        from pandas import DataFrame as DF
    else:
        from cudf import DataFrame as DF

    def make(n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        X, y = make_classification(
            n,
            n_features,
            n_informative=n_features,
            n_redundant=0,
            n_classes=max_rel,
            random_state=seed,
        )
        qid = rng.integers(size=(n,), low=0, high=n_query_groups)
        df = DF(X, columns=[f"f{i}" for i in range(n_features)])
        df["qid"] = qid
        df["y"] = y
        return df

    futures = []
    i = 0
    for k in range(0, n_samples, n_samples_per_worker):
        fut = client.submit(
            make, n=n_samples_per_worker, seed=k, workers=[workers[i % len(workers)]]
        )
        futures.append(fut)
        i += 1

    last = n_samples - (n_samples_per_worker * len(workers))
    if last != 0:
        fut = client.submit(make, n=last, seed=n_samples_per_worker * len(workers))
        futures.append(fut)

    meta = make(1, 0)
    df = dd.from_delayed(futures, meta=meta)
    assert isinstance(df, dd.DataFrame)
    return df.drop(["qid", "y"], axis=1), df.y, df.qid


def check_no_group_split(client: Client, device: str) -> None:
    """Test for the allow_group_split parameter."""
    X_tr, q_tr, y_tr = make_ltr(
        client, 4096, 128, n_query_groups=4, max_rel=5, device=device
    )
    X_va, q_va, y_va = make_ltr(
        client, 1024, 128, n_query_groups=4, max_rel=5, device=device
    )

    ltr = dxgb.DaskXGBRanker(
        allow_group_split=False,
        n_estimators=36,
        device=device,
        objective="rank:pairwise",
    )
    ltr.fit(
        X_tr,
        y_tr,
        qid=q_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        eval_qid=[q_tr, q_va],
        verbose=True,
    )

    assert ltr.n_features_in_ == 128
    assert X_tr.shape[1] == ltr.n_features_in_  # no change
    ndcg = ltr.evals_result()["validation_0"]["ndcg@32"]
    assert tm.non_decreasing(ndcg[:16], tolerance=1e-2), ndcg
    np.testing.assert_allclose(ndcg[-1], 1.0, rtol=1e-2)


@overload
def make_categorical(  # pylint: disable=too-many-locals, too-many-arguments
    client: Client,
    n_samples: int,
    n_features: int,
    n_categories: int,
    *,
    onehot: bool = ...,
    n_targets: Literal[1] = ...,
    cat_dtype: np.typing.DTypeLike = ...,
) -> Tuple[dd.DataFrame, dd.Series]: ...


@overload
def make_categorical(  # pylint: disable=too-many-locals, too-many-arguments
    client: Client,
    n_samples: int,
    n_features: int,
    n_categories: int,
    *,
    onehot: bool = ...,
    n_targets: int,
    cat_dtype: np.typing.DTypeLike = ...,
) -> Tuple[dd.DataFrame, Union[dd.Series, dd.DataFrame]]: ...


def make_categorical(  # pylint: disable=too-many-locals, too-many-arguments
    client: Client,
    n_samples: int,
    n_features: int,
    n_categories: int,
    *,
    onehot: bool = False,
    n_targets: int = 1,
    cat_dtype: np.typing.DTypeLike = np.int64,
) -> Tuple[dd.DataFrame, Union[dd.Series, dd.DataFrame]]:
    """Synthesize categorical data with dask."""
    workers = get_client_workers(client)
    n_workers = len(workers)
    dfs = []

    label_cols = (
        [f"label_{i}" for i in range(n_targets)] if n_targets > 1 else ["label"]
    )

    def pack(**kwargs: Any) -> dd.DataFrame:
        X, y = make_cat_local(**kwargs)
        if y.ndim == 2:
            for i in range(y.shape[1]):
                X[f"label_{i}"] = y[:, i]
        else:
            X["label"] = y
        return X

    meta = pack(
        n_samples=1,
        n_features=n_features,
        n_categories=n_categories,
        onehot=False,
        n_targets=n_targets,
        cat_dtype=cat_dtype,
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
            n_targets=n_targets,
            cat_dtype=cat_dtype,
            onehot=False,
            workers=[worker],
        )
        dfs.append(future)

    df: dd.DataFrame = cast(dd.DataFrame, dd.from_delayed(dfs, meta=meta))
    y = df[label_cols]
    if n_targets == 1:
        y = y[label_cols[0]]
    X = df[df.columns.difference(label_cols)]

    if onehot:
        return dd.get_dummies(X), y
    return X, y


# pylint: disable=too-many-locals
def run_recode(client: Client, device: Device) -> None:
    """Run re-coding test with the Dask interface."""

    def create_dmatrix(
        DMatrixT: Type[dxgb.DaskDMatrix], *args: Any, **kwargs: Any
    ) -> dxgb.DaskDMatrix:
        if DMatrixT is dxgb.DaskQuantileDMatrix:
            ref = kwargs.pop("ref", None)
            return DMatrixT(*args, ref=ref, **kwargs)

        kwargs.pop("ref", None)
        return DMatrixT(*args, **kwargs)

    def run(DMatrixT: Type[dxgb.DaskDMatrix]) -> None:
        enc, reenc, y, _, _ = make_recoded(device, n_features=96)
        to = get_client_workers(client)

        denc, dreenc, dy = (
            dd.from_pandas(enc, npartitions=8).persist(workers=to),
            dd.from_pandas(reenc, npartitions=8).persist(workers=to),
            da.from_array(y, chunks=(y.shape[0] // 8,)).persist(workers=to),
        )

        Xy = create_dmatrix(DMatrixT, client, denc, dy, enable_categorical=True)
        Xy_valid = create_dmatrix(
            DMatrixT, client, dreenc, dy, enable_categorical=True, ref=Xy
        )
        # Base model
        results = dxgb.train(
            client, {"device": device}, Xy, evals=[(Xy_valid, "Valid")]
        )

        # Training continuation
        Xy = create_dmatrix(DMatrixT, client, denc, dy, enable_categorical=True)
        Xy_valid = create_dmatrix(
            DMatrixT, client, dreenc, dy, enable_categorical=True, ref=Xy
        )
        results_1 = dxgb.train(
            client,
            {"device": device},
            Xy,
            evals=[(Xy_valid, "Valid")],
            xgb_model=results["booster"],
        )

        # Reversed training continuation
        Xy = create_dmatrix(DMatrixT, client, dreenc, dy, enable_categorical=True)
        Xy_valid = create_dmatrix(
            DMatrixT, client, denc, dy, enable_categorical=True, ref=Xy
        )
        results_2 = dxgb.train(
            client,
            {"device": device},
            Xy,
            evals=[(Xy_valid, "Valid")],
            xgb_model=results["booster"],
        )
        assert np.isfinite(results_1["history"]["Valid"]["rmse"]).all()
        assert np.isfinite(results_2["history"]["Valid"]["rmse"]).all()

        predt_0 = dxgb.inplace_predict(client, results, denc).compute()
        predt_1 = dxgb.inplace_predict(client, results, dreenc).compute()
        assert_allclose(device, predt_0, predt_1)

        predt_0 = dxgb.predict(client, results, Xy).compute()
        predt_1 = dxgb.predict(client, results, Xy_valid).compute()
        assert_allclose(device, predt_0, predt_1)

    for DMatrixT in [dxgb.DaskDMatrix, dxgb.DaskQuantileDMatrix]:
        run(DMatrixT)
