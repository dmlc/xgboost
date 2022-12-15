import numpy as np
from dask import array as da
from distributed import Client

import xgboost as xgb


def check_init_estimation(tree_method: str, client: Client) -> None:
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=4096, n_features=32)
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
