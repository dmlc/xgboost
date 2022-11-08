import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from sklearn.datasets import make_classification

import xgboost as xgb


def main(client):
    # Optionally force XGBoost to use RMM for all GPU memory allocation, see ./README.md
    # xgb.set_config(use_rmm=True)

    X, y = make_classification(n_samples=10000, n_informative=5, n_classes=3)
    # In pratice one should prefer loading the data with dask collections instead of using
    # `from_array`.
    X = dask.array.from_array(X)
    y = dask.array.from_array(y)
    dtrain = xgb.dask.DaskDMatrix(client, X, label=y)

    params = {'max_depth': 8, 'eta': 0.01, 'objective': 'multi:softprob', 'num_class': 3,
              'tree_method': 'gpu_hist', 'eval_metric': 'merror'}
    output = xgb.dask.train(client, params, dtrain, num_boost_round=100,
                            evals=[(dtrain, 'train')])
    bst = output['booster']
    history = output['history']
    for i, e in enumerate(history['train']['merror']):
        print(f'[{i}] train-merror: {e}')


if __name__ == '__main__':
    # To use RMM pool allocator with a GPU Dask cluster, just add rmm_pool_size option to
    # LocalCUDACluster constructor.
    with LocalCUDACluster(rmm_pool_size='2GB') as cluster:
        with Client(cluster) as client:
            main(client)
