"""
Example of training with Dask on CPU
====================================

"""

from dask import array as da
from dask.distributed import Client, LocalCluster

from xgboost import dask as dxgb
from xgboost.dask.callback import EvaluationMonitor


def approx_train(client: Client, X: da.Array, y: da.Array) -> da.Array:
    # DaskDMatrix acts like normal DMatrix, works as a proxy for local DMatrix scatter
    # around workers.
    dtrain = dxgb.DaskDMatrix(client, X, y)

    # Use train method from xgboost.dask instead of xgboost.  This distributed version
    # of train returns a dictionary containing the resulting booster and evaluation
    # history obtained from evaluation metrics.
    output = dxgb.train(
        client,
        {"tree_method": "approx", "device": "cpu"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )
    bst = output["booster"]
    history = output["history"]

    # you can pass output directly into `predict` too.
    prediction = dxgb.predict(client, bst, dtrain)
    print("Evaluation history:", history)
    return prediction


def hist_train(client: Client, X: da.Array, y: da.Array) -> da.Array:
    """`DaskQuantileDMatrix` is a data type specialized for `hist` tree methods for
     reducing memory usage.

    .. versionadded:: 1.2.0

    """
    # `DaskQuantileDMatrix` is used instead of `DaskDMatrix`, be careful that it can not
    # be used for anything else other than as a training DMatrix, unless a reference is
    # specified. See the `ref` argument of `DaskQuantileDMatrix`.
    dtrain = dxgb.DaskQuantileDMatrix(client, X, y)
    output = dxgb.train(
        client,
        {"tree_method": "hist", "device": "cpu"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
        # See the document of `EvaluationMonitor` for why it's used this way.
        callbacks=[EvaluationMonitor(client, period=1)],
        # Disable the internal logging and prefer the client-side `EvaluationMonitor`.
        verbose_eval=False,
    )
    bst = output["booster"]
    history = output["history"]

    prediction = dxgb.predict(client, bst, X)
    print("Evaluation history:", history)
    return prediction


if __name__ == "__main__":
    with LocalCluster(n_workers=7, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            m = 100000
            n = 100
            rng = da.random.default_rng(1)
            X = rng.normal(size=(m, n), chunks=(10000, -1))
            y = X.sum(axis=1)

            print("Using DaskQuantileDMatrix")
            from_ddqdm = hist_train(client, X, y).compute()
            print("Using DMatrix")
            from_dmatrix = approx_train(client, X, y).compute()
