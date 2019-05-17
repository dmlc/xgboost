import testing as tm
import pytest
import xgboost as xgb
import numpy as np

try:
    from dask.distributed.utils_test import client, loop, cluster_fixture
    import dask.dataframe as dd
    import dask.array as da
except ImportError:
    pass

pytestmark = pytest.mark.skipif(**tm.no_sklearn())


def run_train():
    # Contains one label equal to rank
    dmat = xgb.DMatrix([[0]], label=[xgb.rabit.get_rank()])
    bst = xgb.train({"eta": 1.0, "lambda": 0.0}, dmat, 1)
    pred = bst.predict(dmat)
    expected_result = np.average(range(xgb.rabit.get_world_size()))
    assert all(p == expected_result for p in pred)


def test_train(client):
    # Train two workers, the first has label 0, the second has label 1
    # If they build the model together the output should be 0.5
    xgb.dask.run(client, run_train)


def run_create_dmatrix(X, y, weights):
    dmat = xgb.dask.create_worker_dmatrix(X, y, weight=weights)
    # Expect this worker to get two partitions and concatenate them
    assert dmat.num_row() == 50


def test_dask_dataframe(client):
    n = 10
    m = 100
    partition_size = 25
    X = dd.from_array(np.random.random((m, n)), partition_size)
    y = dd.from_array(np.random.random(m), partition_size)
    weights = dd.from_array(np.random.random(m), partition_size)
    xgb.dask.run(client, run_create_dmatrix, X, y, weights)


def test_dask_array(client):
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


def test_inconsistent_partitions(client):
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
    pred = model.predict(X)
    expected_result = np.average(range(xgb.rabit.get_world_size()))
    assert all(p == expected_result for p in pred)


def test_sklearn(client):
    xgb.dask.run(client, run_sklearn)
