import testing as tm
from distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as da
import xgboost as xgb
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(**tm.no_sklearn())


def run_train():
    # Contains one label equal to rank
    dmat = xgb.DMatrix([[0]], label=[xgb.rabit.get_rank()])
    bst = xgb.train({"eta": 1.0}, dmat, 1)
    return bst.predict(dmat)


def test_train():
    # Train two workers, the first has label 0, the second has label 1
    # If they build the model together the output should be 0.5
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    preds = xgb.dask.run(client, run_train)
    assert all(v[0] == 0.5 for v in preds.values())


def run_create_dmatrix(X, y, weights):
    dmat = xgb.dask.create_worker_dmatrix(X, y, weight=weights)
    # Expect this worker to get two partitions and concatenate them
    assert dmat.num_row() == 50


def test_dask_dataframe():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    n = 10
    m = 100
    partition_size = 25
    X = dd.from_array(np.random.random((m, n)), partition_size)
    y = dd.from_array(np.random.random(m), partition_size)
    weights = dd.from_array(np.random.random(m), partition_size)
    xgb.dask.run(client, run_create_dmatrix, X, y, weights)


def test_dask_array():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    n = 10
    m = 100
    partition_size = 25
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)
    weights = da.random.random(m, partition_size)
    xgb.dask.run(client, run_create_dmatrix, X, y, weights)


def run_inconsistent_partitions(X, y):
    with pytest.raises(ValueError) as e_info:
        xgb.dask.create_worker_dmatrix(X, y)


def test_inconsistent_partitions():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    n = 10
    m = 100
    X = dd.from_array(np.random.random((m, n)), 10)
    y = dd.from_array(np.random.random(m), 20)
    xgb.dask.run(client, run_inconsistent_partitions, X, y)


def run_sklearn():
    # Contains one label equal to rank
    X = [[0]]
    y = [xgb.rabit.get_rank()]
    model = xgb.XGBRegressor(learning_rate=1.0)
    model.fit(X, y)
    return model.predict(X)


def test_sklearn():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    preds = xgb.dask.run(client, run_sklearn)
    assert all(v[0] == 0.5 for v in preds.values())
