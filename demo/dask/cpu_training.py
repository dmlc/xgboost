"""
Example of training with Dask on CPU
====================================

"""

from dask import array as da
from dask.distributed import Client, LocalCluster

from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix


def main(client: Client) -> None:
    # generate some random data for demonstration
    m = 100000
    n = 100
    rng = da.random.default_rng(1)
    X = rng.normal(size=(m, n), chunks=(10000, -1))
    y = X.sum(axis=1)

    # DaskDMatrix acts like normal DMatrix, works as a proxy for local
    # DMatrix scatter around workers.
    dtrain = DaskDMatrix(client, X, y)

    # Use train method from xgboost.dask instead of xgboost.  This
    # distributed version of train returns a dictionary containing the
    # resulting booster and evaluation history obtained from
    # evaluation metrics.
    output = dxgb.train(
        client,
        {"verbosity": 1, "tree_method": "hist"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )
    bst = output["booster"]
    history = output["history"]

    # you can pass output directly into `predict` too.
    prediction = dxgb.predict(client, bst, dtrain)
    print("Evaluation history:", history)
    print("Error:", da.sqrt((prediction - y) ** 2).mean().compute())


if __name__ == "__main__":
    # or use other clusters for scaling
    with LocalCluster(n_workers=7, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            main(client)
