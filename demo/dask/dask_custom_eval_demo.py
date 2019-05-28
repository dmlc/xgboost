from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as da
import numpy as np
import xgboost as xgb


# Define rmse function using Python that mimics the internal xgboost rmse function
def custom_rmse(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()

    mse_local_sum = sum(np.square(labels - preds) * weights)
    weights_local_sum = sum(weights)
    # Pass the sum mse and weights to an allreduce call as a numpy array
    result = xgb.rabit.allreduce(np.asarray([mse_local_sum, weights_local_sum]), 2)
    # Result now contains the sum over all workers
    return 'my-error', np.sqrt(result[0] / result[1])


def train(X, y, weights):
    dtrain = xgb.dask.create_worker_dmatrix(X, y, weight=weights)

    evals_result = {}
    bst = xgb.train({}, dtrain, evals=[(dtrain, 'train')], evals_result=evals_result,
                    feval=custom_rmse)
    # Check our custom function outputs roughly the same result as the built-in version
    assert np.allclose(evals_result['train']['rmse'], evals_result['train']['my-error'])


def main():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)

    n = 10
    m = 100
    partition_size = 20
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)
    weights = da.random.random(m, partition_size)
    xgb.dask.run(client, train, X, y, weights)


if __name__ == '__main__':
    main()
