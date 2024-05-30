"""
Use scikit-learn regressor interface with GPU histogram tree method
===================================================================
"""

import dask
from dask import array as da
from dask.distributed import Client

# It's recommended to use dask_cuda for GPU assignment
from dask_cuda import LocalCUDACluster

from xgboost import dask as dxgb


def main(client: Client) -> dxgb.Booster:
    # Generate some random data for demonstration
    rng = da.random.default_rng(1)

    m = 2**18
    n = 100
    X = rng.uniform(size=(m, n), chunks=(128**2, -1))
    y = X.sum(axis=1)

    regressor = dxgb.DaskXGBRegressor(verbosity=1)
    # Set the device to CUDA
    regressor.set_params(tree_method="hist", device="cuda")
    # Assigning client here is optional
    regressor.client = client

    regressor.fit(X, y, eval_set=[(X, y)])
    prediction = regressor.predict(X)

    bst = regressor.get_booster()
    history = regressor.evals_result()

    print("Evaluation history:", history)
    # returned prediction is always a dask array.
    assert isinstance(prediction, da.Array)
    return bst  # returning the trained model


if __name__ == "__main__":
    # With dask cuda, one can scale up XGBoost to arbitrary GPU clusters.
    # `LocalCUDACluster` used here is only for demonstration purpose.
    with LocalCUDACluster() as cluster:
        # Create client from cluster, set the backend to GPU array (cupy).
        with Client(cluster) as client, dask.config.set({"array.backend": "cupy"}):
            main(client)
