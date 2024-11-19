"""
Example of training with Dask on GPU
====================================
"""

import dask
import dask_cudf
from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix


def using_dask_matrix(client: Client, X: da.Array, y: da.Array) -> da.Array:
    # DaskDMatrix acts like normal DMatrix, works as a proxy for local DMatrix scatter
    # around workers.
    dtrain = DaskDMatrix(client, X, y)

    # Use train method from xgboost.dask instead of xgboost.  This distributed version
    # of train returns a dictionary containing the resulting booster and evaluation
    # history obtained from evaluation metrics.
    output = dxgb.train(
        client,
        # Make sure the device is set to CUDA.
        {"tree_method": "hist", "device": "cuda"},
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


def using_quantile_device_dmatrix(client: Client, X: da.Array, y: da.Array) -> da.Array:
    """`DaskQuantileDMatrix` is a data type specialized for `hist` tree methods for
     reducing memory usage.

    .. versionadded:: 1.2.0

    """
    # `DaskQuantileDMatrix` is used instead of `DaskDMatrix`, be careful that it can not
    # be used for anything else other than training unless a reference is specified. See
    # the `ref` argument of `DaskQuantileDMatrix`.
    dtrain = dxgb.DaskQuantileDMatrix(client, X, y)
    output = dxgb.train(
        client,
        # Make sure the device is set to CUDA.
        {"tree_method": "hist", "device": "cuda"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )

    prediction = dxgb.predict(client, output, X)
    return prediction


if __name__ == "__main__":
    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker process.
    with LocalCUDACluster(n_workers=2, threads_per_worker=4) as cluster:
        # Create client from cluster, set the backend to GPU array (cupy).
        with Client(cluster) as client, dask.config.set({"array.backend": "cupy"}):
            # Generate some random data for demonstration
            rng = da.random.default_rng(1)

            m = 2**18
            n = 100
            X = rng.uniform(size=(m, n), chunks=(128**2, -1))
            y = X.sum(axis=1)

            X = dd.from_dask_array(X)
            y = dd.from_dask_array(y)
            # XGBoost can take arrays. This is to show that DataFrame uses the GPU
            # backend as well.
            assert isinstance(X, dask_cudf.DataFrame)
            assert isinstance(y, dask_cudf.Series)

            print("Using DaskQuantileDMatrix")
            from_ddqdm = using_quantile_device_dmatrix(client, X, y).compute()
            print("Using DMatrix")
            from_dmatrix = using_dask_matrix(client, X, y).compute()
