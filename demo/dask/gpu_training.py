from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import array as da
import xgboost as xgb
from xgboost.dask import DaskDMatrix


def main(client):
    n = 100
    m = 100000
    partition_size = 1000
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)

    # DaskDMatrix acts like normal DMatrix, works as a proxy for local
    # DMatrix scatter around workers.
    dtrain = DaskDMatrix(X, y)

    # Use train method from xgboost.dask instead of xgboost.  This
    # distributed version of train returns a dictionary containing the
    # resulting booster and evaluation history obtained from
    # evaluation metrics.
    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             'nthread': 1,
                             'tree_method': 'gpu_hist'},
                            dtrain,
                            num_boost_round=4, evals=[(dtrain, 'train')])
    bst = output['booster']
    history = output['history']

    prediction = xgb.dask.predict(client, bst, dtrain)
    print('Evaluation history:', history)
    return prediction


if __name__ == '__main__':
    # or use any other clusters
    cluster = LocalCUDACluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    main(client)
