'''Dask interface demo:

Use scikit-learn regressor interface with GPU histogram tree method.'''

from dask.distributed import Client
# It's recommended to use dask_cuda for GPU assignment
from dask_cuda import LocalCUDACluster
from dask import array as da
import xgboost


def main(client):
    # generate some random data for demonstration
    n = 100
    m = 1000000
    partition_size = 10000
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)

    regressor = xgboost.dask.DaskXGBRegressor(verbosity=1)
    regressor.set_params(tree_method='gpu_hist')
    # assigning client here is optional
    regressor.client = client

    regressor.fit(X, y, eval_set=[(X, y)])
    prediction = regressor.predict(X)

    bst = regressor.get_booster()
    history = regressor.evals_result()

    print('Evaluation history:', history)
    # returned prediction is always a dask array.
    assert isinstance(prediction, da.Array)
    return bst                  # returning the trained model


if __name__ == '__main__':
    # With dask cuda, one can scale up XGBoost to arbitrary GPU clusters.
    # `LocalCUDACluster` used here is only for demonstration purpose.
    with LocalCUDACluster() as cluster:
        with Client(cluster) as client:
            main(client)
