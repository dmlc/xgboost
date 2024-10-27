"""Tests for dask shared by different test modules."""

from typing import List, Literal, cast

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client, get_worker

import xgboost as xgb
import xgboost.testing as tm
from xgboost.compat import concat
from xgboost.testing.updater import get_basescore

from .. import dask as dxgb


def check_init_estimation_clf(
    tree_method: str, device: Literal["cpu", "cuda"], client: Client
) -> None:
    """Test init estimation for classsifier."""
    from sklearn.datasets import make_classification

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
