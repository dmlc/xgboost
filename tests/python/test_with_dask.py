import testing as tm
import pytest
import unittest
import xgboost as xgb
import sys
import numpy as np
import json
import asyncio
from sklearn.datasets import make_classification
import os
import subprocess
from hypothesis import given, strategies, settings, note
from test_updaters import hist_parameter_strategy, exact_parameter_strategy

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)

pytestmark = pytest.mark.skipif(**tm.no_dask())

try:
    from distributed import LocalCluster, Client
    from distributed.utils_test import client, loop, cluster_fixture
    import dask.dataframe as dd
    import dask.array as da
    from xgboost.dask import DaskDMatrix
except ImportError:
    LocalCluster = None
    Client = None
    client = None
    loop = None
    cluster_fixture = None
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
    with LocalCluster(n_workers=kWorkers) as cluster:
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
            from_dmatrix = prediction.compute()

            prediction = xgb.dask.predict(client, model=booster, data=X)
            from_df = prediction.compute()

            assert isinstance(prediction, dd.Series)
            assert np.all(prediction.compute().values == from_dmatrix)
            assert np.all(from_dmatrix == from_df.to_numpy())

            series_predictions = xgb.dask.inplace_predict(client, booster, X)
            assert isinstance(series_predictions, dd.Series)
            np.testing.assert_allclose(series_predictions.compute().values,
                                       from_dmatrix)


def test_from_dask_array():
    with LocalCluster(n_workers=kWorkers, threads_per_worker=5) as cluster:
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

            booster = result['booster']
            single_node_predt = booster.predict(
                xgb.DMatrix(X.compute())
            )
            np.testing.assert_allclose(prediction, single_node_predt)

            config = json.loads(booster.save_config())
            assert int(config['learner']['generic_param']['nthread']) == 5

            from_arr = xgb.dask.predict(
                client, model=booster, data=X)

            assert isinstance(from_arr, da.Array)
            assert np.all(single_node_predt == from_arr.compute())


def test_dask_predict_shape_infer():
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            X, y = make_classification(n_samples=1000, n_informative=5,
                                       n_classes=3)
            X_ = dd.from_array(X, chunksize=100)
            y_ = dd.from_array(y, chunksize=100)
            dtrain = xgb.dask.DaskDMatrix(client, data=X_, label=y_)

            model = xgb.dask.train(
                client,
                {"objective": "multi:softprob", "num_class": 3},
                dtrain=dtrain
            )

            preds = xgb.dask.predict(client, model, dtrain)
            assert preds.shape[0] == preds.compute().shape[0]
            assert preds.shape[1] == preds.compute().shape[1]


def test_dask_missing_value_reg():
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            X_0 = np.ones((20 // 2, kCols))
            X_1 = np.zeros((20 // 2, kCols))
            X = np.concatenate([X_0, X_1], axis=0)
            np.random.shuffle(X)
            X = da.from_array(X)
            X = X.rechunk(20, 1)
            y = da.random.randint(0, 3, size=20)
            y.rechunk(20)
            regressor = xgb.dask.DaskXGBRegressor(verbosity=1, n_estimators=2,
                                                  missing=0.0)
            regressor.client = client
            regressor.set_params(tree_method='hist')
            regressor.fit(X, y, eval_set=[(X, y)])
            dd_predt = regressor.predict(X).compute()

            np_X = X.compute()
            np_predt = regressor.get_booster().predict(
                xgb.DMatrix(np_X, missing=0.0))
            np.testing.assert_allclose(np_predt, dd_predt)


def test_dask_missing_value_cls():
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            X_0 = np.ones((kRows // 2, kCols))
            X_1 = np.zeros((kRows // 2, kCols))
            X = np.concatenate([X_0, X_1], axis=0)
            np.random.shuffle(X)
            X = da.from_array(X)
            X = X.rechunk(20, None)
            y = da.random.randint(0, 3, size=kRows)
            y = y.rechunk(20, 1)
            cls = xgb.dask.DaskXGBClassifier(verbosity=1, n_estimators=2,
                                             tree_method='hist',
                                             missing=0.0)
            cls.client = client
            cls.fit(X, y, eval_set=[(X, y)])
            dd_predt = cls.predict(X).compute()

            np_X = X.compute()
            np_predt = cls.get_booster().predict(
                xgb.DMatrix(np_X, missing=0.0))
            np.testing.assert_allclose(np_predt, dd_predt)

            cls = xgb.dask.DaskXGBClassifier()
            assert hasattr(cls, 'missing')


def test_dask_regressor():
    with LocalCluster(n_workers=kWorkers) as cluster:
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


def test_dask_classifier():
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            X, y = generate_array()
            y = (y * 10).astype(np.int32)
            classifier = xgb.dask.DaskXGBClassifier(
                verbosity=1, n_estimators=2)
            classifier.client = client
            classifier.fit(X, y,  eval_set=[(X, y)])
            prediction = classifier.predict(X)

            assert prediction.ndim == 2
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

            assert prediction.ndim == 2
            assert prediction.shape[0] == kRows


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_grid_search():
    from sklearn.model_selection import GridSearchCV
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            X, y = generate_array()
            reg = xgb.dask.DaskXGBRegressor(learning_rate=0.1,
                                            tree_method='hist')
            reg.client = client
            model = GridSearchCV(reg, {'max_depth': [2, 4],
                                       'n_estimators': [5, 10]},
                                 cv=2, verbose=1, iid=True)
            model.fit(X, y)
            # Expect unique results for each parameter value This confirms
            # sklearn is able to successfully update the parameter
            means = model.cv_results_['mean_test_score']
            assert len(means) == len(set(means))


def run_empty_dmatrix_reg(client, parameters):

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


def run_empty_dmatrix_cls(client, parameters):
    n_classes = 4

    def _check_outputs(out, predictions):
        assert isinstance(out['booster'], xgb.dask.Booster)
        assert len(out['history']['validation']['merror']) == 2
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[1] == n_classes, predictions.shape

    kRows, kCols = 1, 97
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.randint(low=0, high=n_classes, size=kRows))
    dtrain = xgb.dask.DaskDMatrix(client, X, y)
    parameters['objective'] = 'multi:softprob'
    parameters['num_class'] = n_classes

    out = xgb.dask.train(client, parameters,
                         dtrain=dtrain,
                         evals=[(dtrain, 'validation')],
                         num_boost_round=2)
    predictions = xgb.dask.predict(client=client, model=out,
                                   data=dtrain)
    assert predictions.shape[1] == n_classes
    predictions = predictions.compute()
    _check_outputs(out, predictions)

    # train has more rows than evals
    valid = dtrain
    kRows += 1
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.randint(low=0, high=n_classes, size=kRows))
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
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            parameters = {'tree_method': 'hist'}
            run_empty_dmatrix_reg(client, parameters)
            run_empty_dmatrix_cls(client, parameters)


def test_empty_dmatrix_approx():
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            parameters = {'tree_method': 'approx'}
            run_empty_dmatrix_reg(client, parameters)
            run_empty_dmatrix_cls(client, parameters)


async def run_from_dask_array_asyncio(scheduler_address):
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y = generate_array()
        m = await DaskDMatrix(client, X, y)
        output = await xgb.dask.train(client, {}, dtrain=m)

        with_m = await xgb.dask.predict(client, output, m)
        with_X = await xgb.dask.predict(client, output, X)
        inplace = await xgb.dask.inplace_predict(client, output, X)
        assert isinstance(with_m, da.Array)
        assert isinstance(with_X, da.Array)
        assert isinstance(inplace, da.Array)

        np.testing.assert_allclose(await client.compute(with_m),
                                   await client.compute(with_X))
        np.testing.assert_allclose(await client.compute(with_m),
                                   await client.compute(inplace))

        client.shutdown()
        return output


async def run_dask_regressor_asyncio(scheduler_address):
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y = generate_array()
        regressor = await xgb.dask.DaskXGBRegressor(verbosity=1,
                                                    n_estimators=2)
        regressor.set_params(tree_method='hist')
        regressor.client = client
        await regressor.fit(X, y, eval_set=[(X, y)])
        prediction = await regressor.predict(X)

        assert prediction.ndim == 1
        assert prediction.shape[0] == kRows

        history = regressor.evals_result()

        assert isinstance(prediction, da.Array)
        assert isinstance(history, dict)

        assert list(history['validation_0'].keys())[0] == 'rmse'
        assert len(history['validation_0']['rmse']) == 2


async def run_dask_classifier_asyncio(scheduler_address):
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y = generate_array()
        y = (y * 10).astype(np.int32)
        classifier = await xgb.dask.DaskXGBClassifier(
            verbosity=1, n_estimators=2)
        classifier.client = client
        await classifier.fit(X, y,  eval_set=[(X, y)])
        prediction = await classifier.predict(X)

        assert prediction.ndim == 2
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
        await classifier.fit(X_d, y_d)

        assert classifier.n_classes_ == 10
        prediction = await classifier.predict(X_d)

        assert prediction.ndim == 2
        assert prediction.shape[0] == kRows
        assert prediction.shape[1] == 10


def test_with_asyncio():
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            address = client.scheduler.address
            output = asyncio.run(run_from_dask_array_asyncio(address))
            assert isinstance(output['booster'], xgb.Booster)
            assert isinstance(output['history'], dict)

            asyncio.run(run_dask_regressor_asyncio(address))
            asyncio.run(run_dask_classifier_asyncio(address))


class TestWithDask:
    def run_updater_test(self, client, params, num_rounds, dataset,
                         tree_method):
        params['tree_method'] = tree_method
        params = dataset.set_params(params)
        # multi class doesn't handle empty dataset well (empty
        # means at least 1 worker has data).
        if params['objective'] == "multi:softmax":
            return
        # It doesn't make sense to distribute a completely
        # empty dataset.
        if dataset.X.shape[0] == 0:
            return

        chunk = 128
        X = da.from_array(dataset.X,
                          chunks=(chunk, dataset.X.shape[1]))
        y = da.from_array(dataset.y, chunks=(chunk, ))
        if dataset.w is not None:
            w = da.from_array(dataset.w, chunks=(chunk, ))
        else:
            w = None

        m = xgb.dask.DaskDMatrix(
            client, data=X, label=y, weight=w)
        history = xgb.dask.train(client, params=params, dtrain=m,
                                 num_boost_round=num_rounds,
                                 evals=[(m, 'train')])['history']
        note(history)
        assert tm.non_increasing(history['train'][dataset.metric])

    @given(params=hist_parameter_strategy,
           num_rounds=strategies.integers(10, 20),
           dataset=tm.dataset_strategy)
    @settings(deadline=None)
    def test_hist(self, params, num_rounds, dataset, client):
        self.run_updater_test(client, params, num_rounds, dataset, 'hist')

    @given(params=exact_parameter_strategy,
           num_rounds=strategies.integers(10, 20),
           dataset=tm.dataset_strategy)
    @settings(deadline=None)
    def test_approx(self, client, params, num_rounds, dataset):
        self.run_updater_test(client, params, num_rounds, dataset, 'approx')

    def run_quantile(self, name):
        if sys.platform.startswith("win"):
            pytest.skip("Skipping dask tests on Windows")

        exe = None
        for possible_path in {'./testxgboost', './build/testxgboost',
                              '../build/testxgboost',
                              '../cpu-build/testxgboost',
                              '../gpu-build/testxgboost'}:
            if os.path.exists(possible_path):
                exe = possible_path
        if exe is None:
            return

        test = "--gtest_filter=Quantile." + name

        def runit(worker_addr, rabit_args):
            port = None
            # setup environment for running the c++ part.
            for arg in rabit_args:
                if arg.decode('utf-8').startswith('DMLC_TRACKER_PORT'):
                    port = arg.decode('utf-8')
            port = port.split('=')
            env = os.environ.copy()
            env[port[0]] = port[1]
            return subprocess.run([exe, test], env=env, stdout=subprocess.PIPE)

        with LocalCluster(n_workers=4) as cluster:
            with Client(cluster) as client:
                workers = list(xgb.dask._get_client_workers(client).keys())
                rabit_args = client.sync(
                    xgb.dask._get_rabit_args, workers, client)
                futures = client.map(runit,
                                     workers,
                                     pure=False,
                                     workers=workers,
                                     rabit_args=rabit_args)
                results = client.gather(futures)
                for ret in results:
                    msg = ret.stdout.decode('utf-8')
                    assert msg.find('1 test from Quantile') != -1, msg
                    assert ret.returncode == 0, msg

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.gtest
    def test_quantile_basic(self):
        self.run_quantile('SameOnAllWorkers')
