"""Tests for dask shared by different test modules."""
import numpy as np
from dask import array as da
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
