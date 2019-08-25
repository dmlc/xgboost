import xgboost as xgb
from xgboost.dask import DaskDMatrix
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask import array as da


if __name__ == '__main__':
    cluster = LocalCluster(n_workers=4)  # or use any other clusters
    client = Client(cluster)

    n = 100
    m = 100000
    partition_size = 1000
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)

    dtrain = DaskDMatrix(X, y)

    output = xgb.dask.train(client,
                            {'verbosity': 3, 'nthread': 4,
                             'tree_method': 'hist'},
                            dtrain,
                            num_boost_round=4)
    bst = output['booster']
    history = output['history']

    prediction = xgb.dask.predict(bst, dtrain)
    print('Evaluation history:', history)
