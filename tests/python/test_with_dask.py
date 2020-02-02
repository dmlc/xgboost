import testing as tm
import pytest
import xgboost as xgb
import sys
import numpy as np

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)

pytestmark = pytest.mark.skipif(**tm.no_dask())

try:
    from distributed import LocalCluster, Client
    import dask.dataframe as dd
    import dask.array as da
    from xgboost.dask import DaskDMatrix
except ImportError:
    LocalCluster = None
    Client = None
    dd = None
    da = None
    DaskDMatrix = None

kRows = 1000
kCols = 10
kWorkers = 5


def generate_array():
    partition_size = 20
    X = da.random.random((kRows, kCols), partition_size)
    y = da.random.random(kRows, partition_size)
    return X, y


def test_from_dask_dataframe():
    with LocalCluster(n_workers=5) as cluster:
        with Client(cluster) as client:
            X, y = generate_array()

            X = dd.from_dask_array(X)
            y = dd.from_dask_array(y)

            dtrain = DaskDMatrix(client, X, y)
            booster = xgb.dask.train(
                client, {}, dtrain, num_boost_round=2)['booster']

            prediction = xgb.dask.predict(client, model=booster, data=dtrain)

            assert prediction.ndim == 1
            assert isinstance(prediction, da.Array)
            assert prediction.shape[0] == kRows

            with pytest.raises(ValueError):
                # evals_result is not supported in dask interface.
                xgb.dask.train(
                    client, {}, dtrain, num_boost_round=2, evals_result={})
            # force prediction to be computed
            prediction = prediction.compute()


def test_from_dask_array():
    with LocalCluster(n_workers=5) as cluster:
        with Client(cluster) as client:
            X, y = generate_array()
            dtrain = DaskDMatrix(client, X, y)
            # results is {'booster': Booster, 'history': {...}}
            result = xgb.dask.train(client, {}, dtrain)

            prediction = xgb.dask.predict(client, result, dtrain)
            assert prediction.shape[0] == kRows

            assert isinstance(prediction, da.Array)
            # force prediction to be computed
            prediction = prediction.compute()


def test_dask_regressor():
    with LocalCluster(n_workers=5) as cluster:
        with Client(cluster) as client:
            X, y = generate_array()
            regressor = xgb.dask.DaskXGBRegressor(verbosity=1, n_estimators=2)
            regressor.set_params(tree_method='hist')
            regressor.client = client
            regressor.fit(X, y, eval_set=[(X, y)])
            prediction = regressor.predict(X)

            assert prediction.ndim == 1
            assert prediction.shape[0] == kRows

            history = regressor.evals_result()

            assert isinstance(prediction, da.Array)
            assert isinstance(history, dict)

            assert list(history['validation_0'].keys())[0] == 'rmse'
            assert len(history['validation_0']['rmse']) == 2


def test_dask_classifier(client):
    with LocalCluster(n_workers=5) as cluster:
        with Client(cluster) as client:
            X, y = generate_array()
            y = (y * 10).astype(np.int32)
            classifier = xgb.dask.DaskXGBClassifier(
                verbosity=1, n_estimators=2)
            classifier.client = client
            classifier.fit(X, y,  eval_set=[(X, y)])
            prediction = classifier.predict(X)

            assert prediction.ndim == 1
            assert prediction.shape[0] == kRows

            history = classifier.evals_result()

            assert isinstance(prediction, da.Array)
            assert isinstance(history, dict)

            assert list(history.keys())[0] == 'validation_0'
            assert list(history['validation_0'].keys())[0] == 'merror'
            assert len(list(history['validation_0'])) == 1
            assert len(history['validation_0']['merror']) == 2

            assert classifier.n_classes_ == 10

            # Test with dataframe.
            X_d = dd.from_dask_array(X)
            y_d = dd.from_dask_array(y)
            classifier.fit(X_d, y_d)

            assert classifier.n_classes_ == 10
            prediction = classifier.predict(X_d)

            assert prediction.ndim == 1
            assert prediction.shape[0] == kRows


def run_empty_dmatrix(client, parameters):

    def _check_outputs(out, predictions):
        assert isinstance(out['booster'], xgb.dask.Booster)
        assert len(out['history']['validation']['rmse']) == 2
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 1

    kRows, kCols = 1, 97
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.rand(kRows))
    dtrain = xgb.dask.DaskDMatrix(client, X, y)

    out = xgb.dask.train(client, parameters,
                         dtrain=dtrain,
                         evals=[(dtrain, 'validation')],
                         num_boost_round=2)
    predictions = xgb.dask.predict(client=client, model=out,
                                   data=dtrain).compute()
    _check_outputs(out, predictions)

    # train has more rows than evals
    valid = dtrain
    kRows += 1
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.rand(kRows))
    dtrain = xgb.dask.DaskDMatrix(client, X, y)

    out = xgb.dask.train(client, parameters,
                         dtrain=dtrain,
                         evals=[(valid, 'validation')],
                         num_boost_round=2)
    predictions = xgb.dask.predict(client=client, model=out,
                                   data=valid).compute()
    _check_outputs(out, predictions)


# No test for Exact, as empty DMatrix handling are mostly for distributed
# environment and Exact doesn't support it.

def test_empty_dmatrix_hist():
    with LocalCluster(n_workers=5) as cluster:
        with Client(cluster) as client:
            parameters = {'tree_method': 'hist'}
            run_empty_dmatrix(client, parameters)


def test_empty_dmatrix_approx():
    with LocalCluster(n_workers=5) as cluster:
        with Client(cluster) as client:
            parameters = {'tree_method': 'approx'}
            run_empty_dmatrix(client, parameters)
