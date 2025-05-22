"""Tests for dask shared by different test modules."""

from typing import Any, List, Literal, Tuple, cast

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client, get_worker
from packaging.version import parse as parse_version
from sklearn.datasets import make_classification

import xgboost as xgb
import xgboost.testing as tm
from xgboost.compat import concat
from xgboost.testing.updater import get_basescore

from .. import dask as dxgb
from ..dask import _DASK_VERSION, _get_rabit_args


def check_init_estimation_clf(
    tree_method: str, device: Literal["cpu", "cuda"], client: Client
) -> None:
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


def check_init_estimation_reg(
    tree_method: str, device: Literal["cpu", "cuda"], client: Client
) -> None:
    """Test init estimation for regressor."""
    from sklearn.datasets import make_regression

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


def check_init_estimation(
    tree_method: str, device: Literal["cpu", "cuda"], client: Client
) -> None:
    """Test init estimation."""
    check_init_estimation_reg(tree_method, device, client)
    check_init_estimation_clf(tree_method, device, client)


def check_uneven_nan(
    client: Client, tree_method: str, device: Literal["cpu", "cuda"], n_workers: int
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
            *tm.make_batches(
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
        results: xgb.callback.TrainingCallback.EvalsLog = {}
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
        x, y, w = tm.make_batches(
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

    results_local: xgb.callback.TrainingCallback.EvalsLog = {}
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
            n, n_features, n_informative=n_features, n_redundant=0, n_classes=max_rel
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

    ltr = dxgb.DaskXGBRanker(allow_group_split=False, n_estimators=36, device=device)
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
