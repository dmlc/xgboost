from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import array as da
import xgboost as xgb
from xgboost.dask import DaskDMatrix


def main(client):
    # generate some random data for demonstration
    m = 100000
    n = 100
    X = da.random.random(size=(m, n), chunks=100)
    y = da.random.random(size=(m, ), chunks=100)

    # DaskDMatrix acts like normal DMatrix, works as a proxy for local
    # DMatrix scatter around workers.
    dtrain = DaskDMatrix(client, X, y)

    # Use train method from xgboost.dask instead of xgboost.  This
    # distributed version of train returns a dictionary containing the
    # resulting booster and evaluation history obtained from
    # evaluation metrics.
    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             # Golden line for GPU training
                             'tree_method': 'gpu_hist'},
                            dtrain,
                            num_boost_round=4, evals=[(dtrain, 'train')])
    bst = output['booster']
    history = output['history']

    # you can pass output directly into `predict` too.
    prediction = xgb.dask.predict(client, bst, dtrain)
    prediction = prediction.compute()
    print('Evaluation history:', history)
    return prediction


if __name__ == '__main__':
    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker
    # process.
    with LocalCUDACluster(n_workers=2, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            main(client)
