'''Dask interface demo:

Use scikit-learn regressor interface with GPU histogram tree method.'''

from dask.distributed import Client
# It's recommended to use dask_cuda for GPU assignment
from dask_cuda import LocalCUDACluster
from dask import array as da
import xgboost

if __name__ == '__main__':
    cluster = LocalCUDACluster()
    client = Client(cluster)

    n = 100
    m = 1000000
    partition_size = 10000
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)

    regressor = xgboost.dask.DaskXGBRegressor(verbosity=2)
    regressor.set_params(tree_method='gpu_hist')
    regressor.client = client

    regressor.fit(X, y, eval_set=[(X, y)])
    prediction = regressor.predict(X)

    bst = regressor.get_booster()
    history = regressor.evals_result()

    print('Evaluation history:', history)
