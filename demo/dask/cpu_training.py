import xgboost as xgb
from xgboost.dask import DaskDMatrix
from dask.distributed import Client
from dask.distributed import LocalCluster
from dask import array as da


def main(client):
    n = 100
    m = 100000
    partition_size = 1000
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)

    dtrain = DaskDMatrix(X, y)

    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             'nthread': 1,
                             'tree_method': 'hist'},
                            dtrain,
                            num_boost_round=4, evals=[(dtrain, 'train')])
    bst = output['booster']
    history = output['history']

    prediction = xgb.dask.predict(client, bst, dtrain)
    print('Evaluation history:', history)
    return prediction


if __name__ == '__main__':
    # or use any other clusters
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    main(client)
