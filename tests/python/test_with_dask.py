import testing as tm
import pytest
import xgboost as xgb
import sys
import numpy as np

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)

pytestmark = pytest.mark.skipif(**tm.no_dask())

try:
    from distributed.utils_test import client, loop, cluster_fixture
    import dask.dataframe as dd
    import dask.array as da
    from xgboost.dask import DaskDMatrix
except ImportError:
    client = None
    loop = None
    cluster_fixture = None
    pass

kRows = 1000


def generate_array():
    n = 10
    partition_size = 20
    X = da.random.random((kRows, n), partition_size)
    y = da.random.random(kRows, partition_size)
    return X, y


def test_from_dask_dataframe(client):
    X, y = generate_array()

    X = dd.from_dask_array(X)
    y = dd.from_dask_array(y)

    dtrain = DaskDMatrix(client, X, y)
    booster = xgb.dask.train(
        client, {}, dtrain, num_boost_round=2)['booster']

    prediction = xgb.dask.predict(client, model=booster, data=dtrain)

    assert isinstance(prediction, da.Array)
    assert prediction.shape[0] == kRows, prediction

    with pytest.raises(ValueError):
        # evals_result is not supported in dask interface.
        xgb.dask.train(
            client, {}, dtrain, num_boost_round=2, evals_result={})


def test_from_dask_array(client):
    X, y = generate_array()
    dtrain = DaskDMatrix(client, X, y)
    # results is {'booster': Booster, 'history': {...}}
    result = xgb.dask.train(client, {}, dtrain)

    prediction = xgb.dask.predict(client, result, dtrain)

    assert isinstance(prediction, da.Array)


def test_regressor(client):
    X, y = generate_array()
    regressor = xgb.dask.DaskXGBRegressor(verbosity=1, n_estimators=2)
    regressor.set_params(tree_method='hist')
    regressor.client = client
    regressor.fit(X, y, eval_set=[(X, y)])
    prediction = regressor.predict(X)

    history = regressor.evals_result()

    assert isinstance(prediction, da.Array)
    assert isinstance(history, dict)

    assert list(history['validation_0'].keys())[0] == 'rmse'
    assert len(history['validation_0']['rmse']) == 2


def test_classifier(client):
    X, y = generate_array()
    y = (y * 10).astype(np.int32)
    classifier = xgb.dask.DaskXGBClassifier(verbosity=1, n_estimators=2)
    classifier.client = client
    classifier.fit(X, y,  eval_set=[(X, y)])
    prediction = classifier.predict(X)

    history = classifier.evals_result()

    assert isinstance(prediction, da.Array)
    assert isinstance(history, dict)

    assert list(history.keys())[0] == 'validation_0'
    assert list(history['validation_0'].keys())[0] == 'merror'
    assert len(list(history['validation_0'])) == 1
    assert len(history['validation_0']['merror']) == 2
