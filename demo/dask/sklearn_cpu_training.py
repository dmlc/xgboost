'''Dask interface demo:

Use scikit-learn regressor interface with CPU histogram tree method.'''
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask import array as da
import xgboost

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=2, silence_logs=False)  # or use any other clusters
    client = Client(cluster)

    n = 100
    m = 10000
    partition_size = 100
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)

    regressor = xgboost.dask.DaskXGBRegressor(verbosity=2, n_estimators=2)
    regressor.set_params(tree_method='hist')
    regressor.client = client

    regressor.fit(X, y, eval_set=[(X, y)])
    prediction = regressor.predict(X)

    bst = regressor.get_booster()
    history = regressor.evals_result()

    print('Evaluation history:', history)
    assert isinstance(prediction, da.Array)
