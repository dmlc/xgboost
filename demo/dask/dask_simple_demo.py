from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as da
import numpy as np
import xgboost as xgb


# Define the function to be executed on each worker
def train(X, y):
    print("Start training with worker #{}".format(xgb.rabit.get_rank()))
    # X,y are dask objects distributed across the cluster.
    # We must obtain the data local to this worker and convert it to DMatrix for training.
    # xgb.dask.create_worker_dmatrix follows the API exactly of the standard DMatrix constructor
    # (xgb.DMatrix()), except that it 'unpacks' dask distributed objects to obtain data local to
    # this worker
    dtrain = xgb.dask.create_worker_dmatrix(X, y)

    # Train on the data. Each worker will communicate and synchronise during training. The output
    #  model is expected to be identical on each worker.
    bst = xgb.train({}, dtrain)
    # Make predictions on local data
    pred = bst.predict(dtrain)
    print("Finished training with worker #{}".format(xgb.rabit.get_rank()))
    # Get text representation of the model
    return bst.get_dump()


def train_with_sklearn(X, y):
    print("Training with worker #{} using the sklearn API".format(xgb.rabit.get_rank()))
    X_local = xgb.dask.get_local_data(X)
    y_local = xgb.dask.get_local_data(y)
    model = xgb.XGBRegressor(n_estimators=10)
    model.fit(X_local, y_local)
    print("Finished training with worker #{} using the sklearn API".format(xgb.rabit.get_rank()))
    return model.predict(X_local)


def main():
    # Launch a very simple local cluster using two distributed workers with two CPU threads each
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)

    # Generate some small test data as a dask array
    # These data frames are internally split into partitions of 20 rows each and then distributed
    #  among workers, so we will have 5 partitions distributed among 2 workers
    # Note that the partition size MUST be consistent across different dask dataframes/arrays
    n = 10
    m = 100
    partition_size = 20
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)

    # xgb.dask.run launches an arbitrary function and its arguments on the cluster
    # Here train(X, y) will be called on each worker
    # This function blocks until all work is complete
    models = xgb.dask.run(client, train, X, y)

    # models contains a dictionary mapping workers to results
    # We expect that the models are the same over all workers
    first_model = next(iter(models.values()))
    assert all(model == first_model for worker, model in models.items())

    # We can also train using the sklearn API
    results = xgb.dask.run(client, train_with_sklearn, X, y)


if __name__ == '__main__':
    main()
