import testing as tm
import pytest
import xgboost as xgb
import sys
import numpy as np
import json
import asyncio
from sklearn.datasets import make_classification
import os
import subprocess
from hypothesis import given, settings, note
from test_updaters import hist_parameter_strategy, exact_parameter_strategy

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)
if tm.no_dask()['condition']:
    pytest.skip(msg=tm.no_dask()['reason'], allow_module_level=True)


try:
    from distributed import LocalCluster, Client, get_client
    from distributed.utils_test import client, loop, cluster_fixture
    import dask.dataframe as dd
    import dask.array as da
    from xgboost.dask import DaskDMatrix
    import dask
except ImportError:
    LocalCluster = None
    Client = None
    get_client = None
    client = None
    loop = None
    cluster_fixture = None
    dd = None
    da = None
    DaskDMatrix = None
    dask = None

kRows = 1000
kCols = 10
kWorkers = 5


def generate_array(with_weights=False):
    partition_size = 20
    X = da.random.random((kRows, kCols), partition_size)
    y = da.random.random(kRows, partition_size)
    if with_weights:
        w = da.random.random(kRows, partition_size)
        return X, y, w
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


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_boost_from_prediction(tree_method):
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)

    X_ = dd.from_array(X, chunksize=100)
    y_ = dd.from_array(y, chunksize=100)

    with LocalCluster(n_workers=4) as cluster:
        with Client(cluster) as _:
            model_0 = xgb.dask.DaskXGBClassifier(
                learning_rate=0.3,
                random_state=123,
                n_estimators=4,
                tree_method=tree_method,
            )
            model_0.fit(X=X_, y=y_)
            margin = model_0.predict_proba(X_, output_margin=True)

            model_1 = xgb.dask.DaskXGBClassifier(
                learning_rate=0.3,
                random_state=123,
                n_estimators=4,
                tree_method=tree_method,
            )
            model_1.fit(X=X_, y=y_, base_margin=margin)
            predictions_1 = model_1.predict(X_, base_margin=margin)
            proba_1 = model_1.predict_proba(X_, base_margin=margin)

            cls_2 = xgb.dask.DaskXGBClassifier(
                learning_rate=0.3,
                random_state=123,
                n_estimators=8,
                tree_method=tree_method,
            )
            cls_2.fit(X=X_, y=y_)
            predictions_2 = cls_2.predict(X_)
            proba_2 = cls_2.predict_proba(X_)

            cls_3 = xgb.dask.DaskXGBClassifier(
                learning_rate=0.3,
                random_state=123,
                n_estimators=8,
                tree_method=tree_method,
            )
            cls_3.fit(X=X_, y=y_)
            proba_3 = cls_3.predict_proba(X_)

            # compute variance of probability percentages between two of the
            # same model, use this to check to make sure approx is functioning
            # within normal parameters
            expected_variance = np.max(np.abs(proba_3 - proba_2)).compute()

            if expected_variance > 0:
                margin_variance = np.max(np.abs(proba_1 - proba_2)).compute()
                # Ensure the margin variance is less than the expected variance + 10%
                assert np.all(margin_variance <= expected_variance + .1)
            else:
                np.testing.assert_equal(predictions_1.compute(), predictions_2.compute())
                np.testing.assert_almost_equal(proba_1.compute(), proba_2.compute())


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
            dd_pred_proba = cls.predict_proba(X).compute()

            np_X = X.compute()
            np_pred_proba = cls.get_booster().predict(
                xgb.DMatrix(np_X, missing=0.0))
            np.testing.assert_allclose(np_pred_proba, dd_pred_proba)

            cls = xgb.dask.DaskXGBClassifier()
            assert hasattr(cls, 'missing')


def test_dask_regressor():
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            X, y, w = generate_array(with_weights=True)
            regressor = xgb.dask.DaskXGBRegressor(verbosity=1, n_estimators=2)
            regressor.set_params(tree_method='hist')
            regressor.client = client
            regressor.fit(X, y, sample_weight=w, eval_set=[(X, y)])
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
            X, y, w = generate_array(with_weights=True)
            y = (y * 10).astype(np.int32)
            classifier = xgb.dask.DaskXGBClassifier(
                verbosity=1, n_estimators=2, eval_metric='merror')
            classifier.client = client
            classifier.fit(X, y, sample_weight=w, eval_set=[(X, y)])
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

            # Test .predict_proba()
            probas = classifier.predict_proba(X)
            assert classifier.n_classes_ == 10
            assert probas.ndim == 2
            assert probas.shape[0] == kRows
            assert probas.shape[1] == 10

            cls_booster = classifier.get_booster()
            single_node_proba = cls_booster.inplace_predict(X.compute())

            np.testing.assert_allclose(single_node_proba,
                                       probas.compute())

            # Test with dataframe.
            X_d = dd.from_dask_array(X)
            y_d = dd.from_dask_array(y)
            classifier.fit(X_d, y_d)

            assert classifier.n_classes_ == 10
            prediction = classifier.predict(X_d)

            assert prediction.ndim == 1
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
                                 cv=2, verbose=1)
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
    parameters['eval_metric'] = 'merror'
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
            verbosity=1, n_estimators=2, eval_metric='merror')
        classifier.client = client
        await classifier.fit(X, y, eval_set=[(X, y)])
        prediction = await classifier.predict(X)

        assert prediction.ndim == 1
        assert prediction.shape[0] == kRows

        history = classifier.evals_result()

        assert isinstance(prediction, da.Array)
        assert isinstance(history, dict)

        assert list(history.keys())[0] == 'validation_0'
        assert list(history['validation_0'].keys())[0] == 'merror'
        assert len(list(history['validation_0'])) == 1
        assert len(history['validation_0']['merror']) == 2

        # Test .predict_proba()
        probas = await classifier.predict_proba(X)
        assert classifier.n_classes_ == 10
        assert probas.ndim == 2
        assert probas.shape[0] == kRows
        assert probas.shape[1] == 10

        # Test with dataframe.
        X_d = dd.from_dask_array(X)
        y_d = dd.from_dask_array(y)
        await classifier.fit(X_d, y_d)

        assert classifier.n_classes_ == 10
        prediction = await classifier.predict(X_d)

        assert prediction.ndim == 1
        assert prediction.shape[0] == kRows


def test_with_asyncio():
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            address = client.scheduler.address
            output = asyncio.run(run_from_dask_array_asyncio(address))
            assert isinstance(output['booster'], xgb.Booster)
            assert isinstance(output['history'], dict)

            asyncio.run(run_dask_regressor_asyncio(address))
            asyncio.run(run_dask_classifier_asyncio(address))


def test_predict():
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            X, y = generate_array()
            dtrain = DaskDMatrix(client, X, y)
            booster = xgb.dask.train(
                client, {}, dtrain, num_boost_round=2)['booster']

            pred = xgb.dask.predict(client, model=booster, data=dtrain)
            assert pred.ndim == 1
            assert pred.shape[0] == kRows

            margin = xgb.dask.predict(client, model=booster, data=dtrain,
                                      output_margin=True)
            assert margin.ndim == 1
            assert margin.shape[0] == kRows

            shap = xgb.dask.predict(client, model=booster, data=dtrain,
                                    pred_contribs=True)
            assert shap.ndim == 2
            assert shap.shape[0] == kRows
            assert shap.shape[1] == kCols + 1


def run_aft_survival(client, dmatrix_t):
    # survival doesn't handle empty dataset well.
    df = dd.read_csv(os.path.join(tm.PROJECT_ROOT, 'demo', 'data',
                                  'veterans_lung_cancer.csv'))
    y_lower_bound = df['Survival_label_lower_bound']
    y_upper_bound = df['Survival_label_upper_bound']
    X = df.drop(['Survival_label_lower_bound',
                 'Survival_label_upper_bound'], axis=1)
    m = dmatrix_t(client, X, label_lower_bound=y_lower_bound,
                  label_upper_bound=y_upper_bound)
    base_params = {'verbosity': 0,
                   'objective': 'survival:aft',
                   'eval_metric': 'aft-nloglik',
                   'learning_rate': 0.05,
                   'aft_loss_distribution_scale': 1.20,
                   'max_depth': 6,
                   'lambda': 0.01,
                   'alpha': 0.02}

    nloglik_rec = {}
    dists = ['normal', 'logistic', 'extreme']
    for dist in dists:
        params = base_params
        params.update({'aft_loss_distribution': dist})
        evals_result = {}
        out = xgb.dask.train(client, params, m, num_boost_round=100,
                             evals=[(m, 'train')])
        evals_result = out['history']
        nloglik_rec[dist] = evals_result['train']['aft-nloglik']
        # AFT metric (negative log likelihood) improve monotonically
        assert all(p >= q for p, q in zip(nloglik_rec[dist],
                                          nloglik_rec[dist][:1]))
    # For this data, normal distribution works the best
    assert nloglik_rec['normal'][-1] < 4.9
    assert nloglik_rec['logistic'][-1] > 4.9
    assert nloglik_rec['extreme'][-1] > 4.9


def test_aft_survival():
    with LocalCluster(n_workers=1) as cluster:
        with Client(cluster) as client:
            run_aft_survival(client, DaskDMatrix)


class TestWithDask:
    def run_updater_test(self, client, params, num_rounds, dataset,
                         tree_method):
        params['tree_method'] = tree_method
        params = dataset.set_params(params)
        # It doesn't make sense to distribute a completely
        # empty dataset.
        if dataset.X.shape[0] == 0:
            return

        chunk = 128
        X = da.from_array(dataset.X,
                          chunks=(chunk, dataset.X.shape[1]))
        y = da.from_array(dataset.y, chunks=(chunk,))
        if dataset.w is not None:
            w = da.from_array(dataset.w, chunks=(chunk,))
        else:
            w = None

        m = xgb.dask.DaskDMatrix(
            client, data=X, label=y, weight=w)
        history = xgb.dask.train(client, params=params, dtrain=m,
                                 num_boost_round=num_rounds,
                                 evals=[(m, 'train')])['history']
        note(history)
        history = history['train'][dataset.metric]
        assert tm.non_increasing(history)
        # Make sure that it's decreasing
        assert history[-1] < history[0]

    @given(params=hist_parameter_strategy,
           dataset=tm.dataset_strategy)
    @settings(deadline=None)
    def test_hist(self, params, dataset, client):
        num_rounds = 30
        self.run_updater_test(client, params, num_rounds, dataset, 'hist')

    @given(params=exact_parameter_strategy,
           dataset=tm.dataset_strategy)
    @settings(deadline=None)
    def test_approx(self, client, params, dataset):
        num_rounds = 30
        self.run_updater_test(client, params, num_rounds, dataset, 'approx')

    def run_quantile(self, name):
        if sys.platform.startswith("win"):
            pytest.skip("Skipping dask tests on Windows")

        exe = None
        for possible_path in {'./testxgboost', './build/testxgboost',
                              '../build/testxgboost',
                              '../cpu-build/testxgboost'}:
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
            return subprocess.run([exe, test], env=env, capture_output=True)

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
        self.run_quantile('DistributedBasic')

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.gtest
    def test_quantile(self):
        self.run_quantile('Distributed')

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.gtest
    def test_quantile_same_on_all_workers(self):
        self.run_quantile('SameOnAllWorkers')


class TestDaskCallbacks:
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_early_stopping(self, client):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)
        m = xgb.dask.DaskDMatrix(client, X, y)
        early_stopping_rounds = 5
        booster = xgb.dask.train(client, {'objective': 'binary:logistic',
                                          'eval_metric': 'error',
                                          'tree_method': 'hist'}, m,
                                 evals=[(m, 'Train')],
                                 num_boost_round=1000,
                                 early_stopping_rounds=early_stopping_rounds)['booster']
        assert hasattr(booster, 'best_score')
        assert booster.best_iteration == 10
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_early_stopping_custom_eval(self, client):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)
        m = xgb.dask.DaskDMatrix(client, X, y)
        early_stopping_rounds = 5
        booster = xgb.dask.train(
            client, {'objective': 'binary:logistic',
                     'eval_metric': 'error',
                     'tree_method': 'hist'}, m,
            evals=[(m, 'Train')],
            feval=tm.eval_error_metric,
            num_boost_round=1000,
            early_stopping_rounds=early_stopping_rounds)['booster']
        assert hasattr(booster, 'best_score')
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_data_initialization(self):
        '''Assert each worker has the correct amount of data, and DMatrix initialization doesn't
        generate unnecessary copies of data.

        '''
        with LocalCluster(n_workers=2) as cluster:
            with Client(cluster) as client:
                X, y = generate_array()
                n_partitions = X.npartitions
                m = xgb.dask.DaskDMatrix(client, X, y)
                workers = list(xgb.dask._get_client_workers(client).keys())
                rabit_args = client.sync(xgb.dask._get_rabit_args, workers, client)
                n_workers = len(workers)

                def worker_fn(worker_addr, data_ref):
                    with xgb.dask.RabitContext(rabit_args):
                        local_dtrain = xgb.dask._dmatrix_from_worker_map(**data_ref)
                        assert local_dtrain.num_row() == kRows / n_workers

                futures = client.map(
                    worker_fn, workers, [m.create_fn_args()] * len(workers),
                    pure=False, workers=workers)
                client.gather(futures)

                has_what = client.has_what()
                cnt = 0
                data = set()
                for k, v in has_what.items():
                    for d in v:
                        cnt += 1
                        data.add(d)

                assert len(data) == cnt
                # Subtract the on disk resource from each worker
                assert cnt - n_workers == n_partitions
