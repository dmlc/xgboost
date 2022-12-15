import numpy as np
from dask import array as da
from distributed import Client

import xgboost as xgb


def check_init_estimation_clf(tree_method: str, client: Client) -> None:
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=4096 * 2, n_features=32, random_state=1994)
    clf = xgb.XGBClassifier(n_estimators=1, max_depth=1, tree_method=tree_method)
    clf.fit(X, y)
    base_score = clf.get_params()["base_score"]

    dX = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dclf = xgb.dask.DaskXGBClassifier(
        n_estimators=1, max_depth=1, tree_method=tree_method
    )
    dclf.client = client
    dclf.fit(dX, dy)
    dbase_score = dclf.get_params()["base_score"]
    np.testing.assert_allclose(base_score, dbase_score)
    print(base_score, dbase_score)


def check_init_estimation_reg(tree_method: str, client: Client) -> None:
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=4096 * 2, n_features=32, random_state=1994)
    reg = xgb.XGBRegressor(n_estimators=1, max_depth=1, tree_method=tree_method)
    reg.fit(X, y)
    base_score = reg.get_params()["base_score"]

    dX = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dreg = xgb.dask.DaskXGBRegressor(
        n_estimators=1, max_depth=1, tree_method=tree_method
    )
    dreg.client = client
    dreg.fit(dX, dy)
    dbase_score = dreg.get_params()["base_score"]
    np.testing.assert_allclose(base_score, dbase_score)
    print(base_score, dbase_score)


def check_init_estimation(tree_method: str, client: Client) -> None:
    check_init_estimation_reg(tree_method, client)
    check_init_estimation_clf(tree_method, client)
