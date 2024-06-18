"""Tests for dask shared by different test modules."""

import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client

import xgboost as xgb
from xgboost.testing.updater import get_basescore


def check_init_estimation_clf(tree_method: str, client: Client) -> None:
    """Test init estimation for classsifier."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=4096 * 2, n_features=32, random_state=1994)
    clf = xgb.XGBClassifier(n_estimators=1, max_depth=1, tree_method=tree_method)
    clf.fit(X, y)
    base_score = get_basescore(clf)

    dx = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dclf = xgb.dask.DaskXGBClassifier(
        n_estimators=1, max_depth=1, tree_method=tree_method
    )
    dclf.client = client
    dclf.fit(dx, dy)
    dbase_score = get_basescore(dclf)
    np.testing.assert_allclose(base_score, dbase_score)


def check_init_estimation_reg(tree_method: str, client: Client) -> None:
    """Test init estimation for regressor."""
    from sklearn.datasets import make_regression

    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(n_samples=4096 * 2, n_features=32, random_state=1994)
    reg = xgb.XGBRegressor(n_estimators=1, max_depth=1, tree_method=tree_method)
    reg.fit(X, y)
    base_score = get_basescore(reg)

    dx = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dreg = xgb.dask.DaskXGBRegressor(
        n_estimators=1, max_depth=1, tree_method=tree_method
    )
    dreg.client = client
    dreg.fit(dx, dy)
    dbase_score = get_basescore(dreg)
    np.testing.assert_allclose(base_score, dbase_score)


def check_init_estimation(tree_method: str, client: Client) -> None:
    """Test init estimation."""
    check_init_estimation_reg(tree_method, client)
    check_init_estimation_clf(tree_method, client)


def check_uneven_nan(client: Client, tree_method: str, n_workers: int) -> None:
    """Issue #9271, not every worker has missing value."""
    assert n_workers >= 2

    with client.as_current():
        clf = xgb.dask.DaskXGBClassifier(tree_method=tree_method)
        X = pd.DataFrame({"a": range(10000), "b": range(10000, 0, -1)})
        y = pd.Series([*[0] * 5000, *[1] * 5000])

        X["a"][:3000:1000] = np.nan

        client.wait_for_workers(n_workers=n_workers)

        clf.fit(
            dd.from_pandas(X, npartitions=n_workers),
            dd.from_pandas(y, npartitions=n_workers),
        )
