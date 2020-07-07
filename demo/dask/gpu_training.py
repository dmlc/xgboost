from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import array as da
import xgboost as xgb
import numpy as np
from xgboost.dask import DaskDMatrix, DaskDeviceQuantileDMatrix
# We use sklearn instead of dask-ml for demo, but for real world use dask-ml
# should be preferred.
from sklearn.metrics import mean_squared_error
from time import time


ROUNDS = 100


def generate_random(chunks=100):
    m = 100000
    n = 100
    X = da.random.random(size=(m, n), chunks=chunks)
    y = da.random.random(size=(m, ), chunks=chunks)
    return X, y


def assert_non_decreasing(L, tolerance=1e-4):
    assert all((y - x) < tolerance for x, y in zip(L, L[1:]))


def train_with_dask_dmatrix(client):
    # generate some random data for demonstration
    X, y = generate_random()

    # DaskDMatrix acts like normal DMatrix, works as a proxy for local
    # DMatrix scatter around workers.
    start = time()
    dtrain = DaskDMatrix(client, X, y)
    end = time()
    print('Constructing DMatrix:', end - start)

    # Use train method from xgboost.dask instead of xgboost.  This
    # distributed version of train returns a dictionary containing the
    # resulting booster and evaluation history obtained from
    # evaluation metrics.
    start = time()
    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             # Golden line for GPU training
                             'tree_method': 'gpu_hist'},
                            dtrain,
                            num_boost_round=ROUNDS, evals=[(dtrain, 'train')])
    end = time()
    print('Training:', end - start)

    bst = output['booster']
    history = output['history']

    # you can pass output directly into `predict` too.
    prediction = xgb.dask.predict(client, bst, dtrain)
    mse = mean_squared_error(y_pred=prediction.compute(), y_true=y.compute())
    print('Evaluation history:', history)
    return mse


def train_with_dask_device_dmatrix(client):
    import cupy
    # generate some random data for demonstration
    X, y = generate_random(10000)

    X = X.map_blocks(cupy.array)
    y = y.map_blocks(cupy.array)

    # DaskDeviceQuantileDMatrix helps reducing memory when input is from device
    # diectly.
    start = time()
    dtrain = DaskDeviceQuantileDMatrix(client, X, y)
    end = time()
    print('Constructing DaskDeviceQuantileDMatrix:', end - start)

    # Use train method from xgboost.dask instead of xgboost.  This
    # distributed version of train returns a dictionary containing the
    # resulting booster and evaluation history obtained from
    # evaluation metrics.
    start = time()
    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             # Golden line for GPU training
                             'tree_method': 'gpu_hist'},
                            dtrain,
                            num_boost_round=ROUNDS, evals=[(dtrain, 'train')])
    end = time()
    print('Training:', end - start)
    bst = output['booster']
    history = output['history']
    assert_non_decreasing(history['train']['rmse'])

    # you can pass output directly into `predict` too.
    prediction = xgb.dask.predict(client, bst, dtrain)
    mse = mean_squared_error(y_pred=prediction.compute(),
                             y_true=y.map_blocks(cupy.asnumpy).compute())
    print('Evaluation history:', history)
    return mse


if __name__ == '__main__':
    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker
    # process.
    with LocalCUDACluster(n_workers=2, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            mse_dmatrix = train_with_dask_dmatrix(client)
            mse_iter = train_with_dask_device_dmatrix(client)
            assert np.isclose(mse_iter, mse_dmatrix, atol=1e-3)
