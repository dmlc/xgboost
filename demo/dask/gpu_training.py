"""
Example of training with Dask on GPU
====================================
"""
from dask_cuda import LocalCUDACluster
import dask_cudf
from dask.distributed import Client, wait
from dask import array as da
from dask import dataframe as dd
import xgboost as xgb
from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix
import argparse


def using_dask_matrix(client: Client, X, y):
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
    print('Evaluation history:', history)
    return prediction


def using_quantile_device_dmatrix(client: Client, X, y):
    '''`DaskDeviceQuantileDMatrix` is a data type specialized for `gpu_hist`, tree
     method that reduces memory overhead.  When training on GPU pipeline, it's
     preferred over `DaskDMatrix`.

    .. versionadded:: 1.2.0

    '''
    # Input must be on GPU for `DaskDeviceQuantileDMatrix`.
    X = dask_cudf.from_dask_dataframe(dd.from_dask_array(X))
    y = dask_cudf.from_dask_dataframe(dd.from_dask_array(y))

    # `DaskDeviceQuantileDMatrix` is used instead of `DaskDMatrix`, be careful
    # that it can not be used for anything else other than training.
    dtrain = dxgb.DaskDeviceQuantileDMatrix(client, X, y)
    output = xgb.dask.train(client,
                            {'verbosity': 2,
                             'tree_method': 'gpu_hist'},
                            dtrain,
                            num_boost_round=4)

    prediction = xgb.dask.predict(client, output, X)
    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ddqdm', choices=[0, 1], type=int, default=1,
        help='''Whether should we use `DaskDeviceQuantileDMatrix`''')
    args = parser.parse_args()

    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker
    # process.
    with LocalCUDACluster(n_workers=2, threads_per_worker=4) as cluster:
        with Client(cluster) as client:
            # generate some random data for demonstration
            m = 100000
            n = 100
            X = da.random.random(size=(m, n), chunks=100)
            y = da.random.random(size=(m, ), chunks=100)

            if args.ddqdm == 1:
                print('Using DaskDeviceQuantileDMatrix')
                from_ddqdm = using_quantile_device_dmatrix(client, X, y)
            else:
                print('Using DMatrix')
                from_dmatrix = using_dask_matrix(client, X, y)
