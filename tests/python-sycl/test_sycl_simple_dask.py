from xgboost import dask as dxgb
from xgboost import testing as tm

from hypothesis import given, strategies, assume, settings, note

import dask.array as da
import dask.distributed


def train_result(client, param, dtrain, num_rounds):
    result = dxgb.train(
        client,
        param,
        dtrain,
        num_rounds,
        verbose_eval=False,
        evals=[(dtrain, "train")],
    )
    return result


class TestSYCLDask:
    # The simplest test verify only one node training.
    def test_simple(self):
        cluster = dask.distributed.LocalCluster(n_workers=1)
        client = dask.distributed.Client(cluster)

        param = {}
        param["tree_method"] = "hist"
        param["device"] = "sycl"
        param["verbosity"] = 0
        param["objective"] = "reg:squarederror"

        # X and y must be Dask dataframes or arrays
        num_obs = 1e4
        num_features = 20
        X = da.random.random(size=(num_obs, num_features), chunks=(1000, num_features))
        y = da.random.random(size=(num_obs, 1), chunks=(1000, 1))
        dtrain = dxgb.DaskDMatrix(client, X, y)

        result = train_result(client, param, dtrain, 10)
        assert tm.non_increasing(result["history"]["train"]["rmse"])
