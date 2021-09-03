from pathlib import Path
import pickle
import testing as tm
import pytest
import xgboost as xgb
import sys
import numpy as np
import scipy
import json
from typing import List, Tuple, Dict, Optional, Type, Any
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import tempfile
from sklearn.datasets import make_classification
import sklearn
import os
import subprocess
import hypothesis
from hypothesis import given, settings, note, HealthCheck
from test_updaters import hist_parameter_strategy, exact_parameter_strategy
from test_with_sklearn import run_feature_weights, run_data_initialization
from test_predict import verify_leaf_output
from sklearn.datasets import make_regression

if sys.platform.startswith("win"):
    pytest.skip("Skipping dask tests on Windows", allow_module_level=True)
if tm.no_dask()['condition']:
    pytest.skip(msg=tm.no_dask()['reason'], allow_module_level=True)

from distributed import LocalCluster, Client
import dask.dataframe as dd
import dask.array as da
from xgboost.dask import DaskDMatrix


if hasattr(HealthCheck, 'function_scoped_fixture'):
    suppress = [HealthCheck.function_scoped_fixture]
else:
    suppress = hypothesis.utils.conventions.not_set  # type:ignore


@pytest.fixture(scope='module')
def cluster():
    with LocalCluster(
        n_workers=2, threads_per_worker=2, dashboard_address=None
    ) as dask_cluster:
        yield dask_cluster


@pytest.fixture
def client(cluster):
    with Client(cluster) as dask_client:
        yield dask_client


kRows = 1000
kCols = 10
kWorkers = 5


def _get_client_workers(client: "Client") -> List[str]:
    workers = client.scheduler_info()['workers']
    return list(workers.keys())


def generate_array(
        with_weights: bool = False
) -> Tuple[xgb.dask._DaskCollection, xgb.dask._DaskCollection,
           Optional[xgb.dask._DaskCollection]]:
    chunk_size = 20
    rng = da.random.RandomState(1994)
    X = rng.random_sample((kRows, kCols), chunks=(chunk_size, -1))
    y = rng.random_sample(kRows, chunks=chunk_size)
    if with_weights:
        w = rng.random_sample(kRows, chunks=chunk_size)
        return X, y, w
    return X, y, None


def test_from_dask_dataframe() -> None:
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            X, y, _ = generate_array()

            X = dd.from_dask_array(X)
            y = dd.from_dask_array(y)

            dtrain = DaskDMatrix(client, X, y)
            booster = xgb.dask.train(client, {}, dtrain, num_boost_round=2)['booster']

            prediction = xgb.dask.predict(client, model=booster, data=dtrain)

            assert prediction.ndim == 1
            assert isinstance(prediction, da.Array)
            assert prediction.shape[0] == kRows

            with pytest.raises(TypeError):
                # evals_result is not supported in dask interface.
                xgb.dask.train(  # type:ignore
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

            # Make sure the output can be integrated back to original dataframe
            X["predict"] = prediction
            X["inplace_predict"] = series_predictions

            assert bool(X.isnull().values.any().compute()) is False


def test_from_dask_array() -> None:
    with LocalCluster(n_workers=kWorkers, threads_per_worker=5) as cluster:
        with Client(cluster) as client:
            X, y, _ = generate_array()
            dtrain = DaskDMatrix(client, X, y)
            # results is {'booster': Booster, 'history': {...}}
            result = xgb.dask.train(client, {}, dtrain)

            prediction = xgb.dask.predict(client, result, dtrain)
            assert prediction.shape[0] == kRows

            assert isinstance(prediction, da.Array)
            # force prediction to be computed
            prediction = prediction.compute()

            booster: xgb.Booster = result['booster']
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


def test_dask_predict_shape_infer(client: "Client") -> None:
    X, y = make_classification(n_samples=1000, n_informative=5, n_classes=3)
    X_ = dd.from_array(X, chunksize=100)
    y_ = dd.from_array(y, chunksize=100)
    dtrain = xgb.dask.DaskDMatrix(client, data=X_, label=y_)

    model = xgb.dask.train(
        client, {"objective": "multi:softprob", "num_class": 3}, dtrain=dtrain
    )

    preds = xgb.dask.predict(client, model, dtrain)
    assert preds.shape[0] == preds.compute().shape[0]
    assert preds.shape[1] == preds.compute().shape[1]

    prediction = xgb.dask.predict(client, model, X_, output_margin=True)
    assert isinstance(prediction, dd.DataFrame)

    prediction = prediction.compute()
    assert prediction.ndim == 2
    assert prediction.shape[0] == kRows
    assert prediction.shape[1] == 3

    prediction = xgb.dask.inplace_predict(client, model, X_, predict_type="margin")
    assert isinstance(prediction, dd.DataFrame)
    prediction = prediction.compute()
    assert prediction.ndim == 2
    assert prediction.shape[0] == kRows
    assert prediction.shape[1] == 3


def run_boost_from_prediction(
    X: xgb.dask._DaskCollection, y: xgb.dask._DaskCollection, tree_method: str, client: "Client"
) -> None:
    X = client.persist(X)
    y = client.persist(y)

    model_0 = xgb.dask.DaskXGBClassifier(
        learning_rate=0.3, random_state=0, n_estimators=4,
        tree_method=tree_method)
    model_0.fit(X=X, y=y)
    margin = model_0.predict(X, output_margin=True)

    model_1 = xgb.dask.DaskXGBClassifier(
        learning_rate=0.3, random_state=0, n_estimators=4,
        tree_method=tree_method)
    model_1.fit(X=X, y=y, base_margin=margin)
    predictions_1 = model_1.predict(X, base_margin=margin)

    cls_2 = xgb.dask.DaskXGBClassifier(
        learning_rate=0.3, random_state=0, n_estimators=8,
        tree_method=tree_method)
    cls_2.fit(X=X, y=y)
    predictions_2 = cls_2.predict(X)

    assert np.all(predictions_1.compute() == predictions_2.compute())

    margined = xgb.dask.DaskXGBClassifier(n_estimators=4)
    margined.fit(
        X=X, y=y, base_margin=margin, eval_set=[(X, y)], base_margin_eval_set=[margin]
    )

    unmargined = xgb.dask.DaskXGBClassifier(n_estimators=4)
    unmargined.fit(X=X, y=y, eval_set=[(X, y)], base_margin=margin)

    margined_res = margined.evals_result()['validation_0']['logloss']
    unmargined_res = unmargined.evals_result()['validation_0']['logloss']

    assert len(margined_res) == len(unmargined_res)
    for i in range(len(margined_res)):
        # margined is correct one, so smaller error.
        assert margined_res[i] < unmargined_res[i]


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_boost_from_prediction(tree_method: str, client: "Client") -> None:
    from sklearn.datasets import load_breast_cancer
    X_, y_ = load_breast_cancer(return_X_y=True)
    X, y = dd.from_array(X_, chunksize=100), dd.from_array(y_, chunksize=100)
    run_boost_from_prediction(X, y, tree_method, client)


def test_inplace_predict(client: "Client") -> None:
    from sklearn.datasets import load_boston
    X_, y_ = load_boston(return_X_y=True)
    X, y = dd.from_array(X_, chunksize=32), dd.from_array(y_, chunksize=32)
    reg = xgb.dask.DaskXGBRegressor(n_estimators=4).fit(X, y)
    booster = reg.get_booster()
    base_margin = y

    inplace = xgb.dask.inplace_predict(
        client, booster, X, base_margin=base_margin
    ).compute()
    Xy = xgb.dask.DaskDMatrix(client, X, base_margin=base_margin)
    copied = xgb.dask.predict(client, booster, Xy).compute()
    np.testing.assert_allclose(inplace, copied)


def test_dask_missing_value_reg(client: "Client") -> None:
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


def test_dask_missing_value_cls(client: "Client") -> None:
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


@pytest.mark.parametrize("model", ["boosting", "rf"])
def test_dask_regressor(model: str, client: "Client") -> None:
    X, y, w = generate_array(with_weights=True)
    if model == "boosting":
        regressor = xgb.dask.DaskXGBRegressor(verbosity=1, n_estimators=2)
    else:
        regressor = xgb.dask.DaskXGBRFRegressor(verbosity=1, n_estimators=2)

    assert regressor._estimator_type == "regressor"
    assert sklearn.base.is_regressor(regressor)

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
    forest = int(
        json.loads(regressor.get_booster().save_config())["learner"][
            "gradient_booster"
        ]["gbtree_train_param"]["num_parallel_tree"]
    )

    if model == "boosting":
        assert len(history['validation_0']['rmse']) == 2
        assert forest == 1
    else:
        assert len(history['validation_0']['rmse']) == 1
        assert forest == 2


def run_dask_classifier(
    X: xgb.dask._DaskCollection,
    y: xgb.dask._DaskCollection,
    w: xgb.dask._DaskCollection,
    model: str,
    tree_method: Optional[str],
    client: "Client",
    n_classes,
) -> None:
    metric = "merror" if n_classes > 2 else "logloss"

    if model == "boosting":
        classifier = xgb.dask.DaskXGBClassifier(
            verbosity=1, n_estimators=2, eval_metric=metric, tree_method=tree_method
        )
    else:
        classifier = xgb.dask.DaskXGBRFClassifier(
            verbosity=1, n_estimators=2, eval_metric=metric, tree_method=tree_method
        )

    assert classifier._estimator_type == "classifier"
    assert sklearn.base.is_classifier(classifier)

    classifier.client = client
    classifier.fit(X, y, sample_weight=w, eval_set=[(X, y)])
    prediction = classifier.predict(X).compute()

    assert prediction.ndim == 1
    assert prediction.shape[0] == kRows

    history = classifier.evals_result()

    assert isinstance(history, dict)

    assert list(history.keys())[0] == "validation_0"
    assert list(history["validation_0"].keys())[0] == metric
    assert len(list(history["validation_0"])) == 1
    forest = int(
        json.loads(classifier.get_booster().save_config())["learner"][
            "gradient_booster"
        ]["gbtree_train_param"]["num_parallel_tree"]
    )
    if model == "boosting":
        assert len(history["validation_0"][metric]) == 2
        assert forest == 1
    else:
        assert len(history["validation_0"][metric]) == 1
        assert forest == 2

    # Test .predict_proba()
    probas = classifier.predict_proba(X).compute()
    assert classifier.n_classes_ == n_classes
    assert probas.ndim == 2
    assert probas.shape[0] == kRows
    assert probas.shape[1] == n_classes

    if n_classes > 2:
        cls_booster = classifier.get_booster()
        single_node_proba = cls_booster.inplace_predict(X.compute())

        # test shared by CPU and GPU
        if isinstance(single_node_proba, np.ndarray):
            np.testing.assert_allclose(single_node_proba, probas)
        else:
            import cupy
            cupy.testing.assert_allclose(single_node_proba, probas)

    # Test with dataframe, not shared with GPU as cupy doesn't work well with da.unique.
    if isinstance(X, da.Array) and n_classes > 2:
        X_d: dd.DataFrame = X.to_dask_dataframe()

        assert classifier.n_classes_ == n_classes
        prediction_df = classifier.predict(X_d).compute()

        assert prediction_df.ndim == 1
        assert prediction_df.shape[0] == kRows
        np.testing.assert_allclose(prediction_df, prediction)

        probas = classifier.predict_proba(X).compute()
        np.testing.assert_allclose(single_node_proba, probas)


@pytest.mark.parametrize("model", ["boosting", "rf"])
def test_dask_classifier(model: str, client: "Client") -> None:
    X, y, w = generate_array(with_weights=True)
    y = (y * 10).astype(np.int32)
    run_dask_classifier(X, y, w, model, None, client, 10)

    y_bin = y.copy()
    y_bin[y > 5] = 1.0
    y_bin[y <= 5] = 0.0
    run_dask_classifier(X, y_bin, w, model, None, client, 2)


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_grid_search(client: "Client") -> None:
    from sklearn.model_selection import GridSearchCV
    X, y, _ = generate_array()
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


def test_empty_dmatrix_training_continuation(client: "Client") -> None:
    kRows, kCols = 1, 97
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.rand(kRows))
    X.columns = ['X' + str(i) for i in range(0, 97)]
    dtrain = xgb.dask.DaskDMatrix(client, X, y)

    kRows += 1000
    X = dd.from_array(np.random.randn(kRows, kCols), chunksize=10)
    X.columns = ['X' + str(i) for i in range(0, 97)]
    y = dd.from_array(np.random.rand(kRows), chunksize=10)
    valid = xgb.dask.DaskDMatrix(client, X, y)

    out = xgb.dask.train(client, {'tree_method': 'hist'},
                         dtrain=dtrain, num_boost_round=2,
                         evals=[(valid, 'validation')])

    out = xgb.dask.train(client, {'tree_method': 'hist'},
                         dtrain=dtrain, xgb_model=out['booster'],
                         num_boost_round=2,
                         evals=[(valid, 'validation')])
    assert xgb.dask.predict(client, out, dtrain).compute().shape[0] == 1


def run_empty_dmatrix_reg(client: "Client", parameters: dict) -> None:
    def _check_outputs(out: xgb.dask.TrainReturnT, predictions: np.ndarray) -> None:
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

    # valid has more rows than train
    kRows += 1
    X = dd.from_array(np.random.randn(kRows, kCols))
    y = dd.from_array(np.random.rand(kRows))
    valid = xgb.dask.DaskDMatrix(client, X, y)
    out = xgb.dask.train(client, parameters,
                         dtrain=dtrain,
                         evals=[(valid, 'validation')],
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


def run_empty_dmatrix_cls(client: "Client", parameters: dict) -> None:
    n_classes = 4

    def _check_outputs(out: xgb.dask.TrainReturnT, predictions: np.ndarray) -> None:
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


def run_empty_dmatrix_auc(client: "Client", tree_method: str, n_workers: int) -> None:
    from sklearn import datasets
    n_samples = 100
    n_features = 97
    rng = np.random.RandomState(1994)

    make_classification = partial(
        datasets.make_classification,
        n_features=n_features,
        random_state=rng
    )

    # binary
    X_, y_ = make_classification(n_samples=n_samples, random_state=rng)
    X = dd.from_array(X_, chunksize=10)
    y = dd.from_array(y_, chunksize=10)

    n_samples = n_workers - 1
    valid_X_, valid_y_ = make_classification(n_samples=n_samples, random_state=rng)
    valid_X = dd.from_array(valid_X_, chunksize=n_samples)
    valid_y = dd.from_array(valid_y_, chunksize=n_samples)

    cls = xgb.dask.DaskXGBClassifier(
        tree_method=tree_method, n_estimators=2, use_label_encoder=False
    )
    cls.fit(X, y, eval_metric="auc", eval_set=[(valid_X, valid_y)])

    # multiclass
    X_, y_ = make_classification(
        n_samples=n_samples,
        n_classes=n_workers,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0
    )
    for i in range(y_.shape[0]):
        y_[i] = i % n_workers
    X = dd.from_array(X_, chunksize=10)
    y = dd.from_array(y_, chunksize=10)

    n_samples = n_workers - 1
    valid_X_, valid_y_ = make_classification(
        n_samples=n_samples,
        n_classes=n_workers,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0
    )
    for i in range(valid_y_.shape[0]):
        valid_y_[i] = i % n_workers
    valid_X = dd.from_array(valid_X_, chunksize=n_samples)
    valid_y = dd.from_array(valid_y_, chunksize=n_samples)

    cls = xgb.dask.DaskXGBClassifier(
        tree_method=tree_method, n_estimators=2, use_label_encoder=False
    )
    cls.fit(X, y, eval_metric="auc", eval_set=[(valid_X, valid_y)])


def test_empty_dmatrix_auc() -> None:
    with LocalCluster(n_workers=8) as cluster:
        with Client(cluster) as client:
            run_empty_dmatrix_auc(client, "hist", 8)


def run_auc(client: "Client", tree_method: str) -> None:
    from sklearn import datasets
    n_samples = 100
    n_features = 97
    rng = np.random.RandomState(1994)
    X_, y_ = datasets.make_classification(
        n_samples=n_samples, n_features=n_features, random_state=rng
    )
    X = dd.from_array(X_, chunksize=10)
    y = dd.from_array(y_, chunksize=10)

    valid_X_, valid_y_ = datasets.make_classification(
        n_samples=n_samples, n_features=n_features, random_state=rng
    )
    valid_X = dd.from_array(valid_X_, chunksize=10)
    valid_y = dd.from_array(valid_y_, chunksize=10)

    cls = xgb.XGBClassifier(
        tree_method=tree_method, n_estimators=2, use_label_encoder=False
    )
    cls.fit(X_, y_, eval_metric="auc", eval_set=[(valid_X_, valid_y_)])

    dcls = xgb.dask.DaskXGBClassifier(
        tree_method=tree_method, n_estimators=2, use_label_encoder=False
    )
    dcls.fit(X, y, eval_metric="auc", eval_set=[(valid_X, valid_y)])

    approx = dcls.evals_result()["validation_0"]["auc"]
    exact = cls.evals_result()["validation_0"]["auc"]
    for i in range(2):
        # approximated test.
        assert np.abs(approx[i] - exact[i]) <= 0.06


def test_auc(client: "Client") -> None:
    run_auc(client, "hist")

# No test for Exact, as empty DMatrix handling are mostly for distributed
# environment and Exact doesn't support it.
def test_empty_dmatrix_hist() -> None:
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            parameters = {'tree_method': 'hist'}
            run_empty_dmatrix_reg(client, parameters)
            run_empty_dmatrix_cls(client, parameters)


def test_empty_dmatrix_approx() -> None:
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            parameters = {'tree_method': 'approx'}
            run_empty_dmatrix_reg(client, parameters)
            run_empty_dmatrix_cls(client, parameters)


async def run_from_dask_array_asyncio(scheduler_address: str) -> xgb.dask.TrainReturnT:
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y, _ = generate_array()
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
    return output


async def run_dask_regressor_asyncio(scheduler_address: str) -> None:
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y, _ = generate_array()
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

        awaited = await client.compute(prediction)
        assert awaited.shape[0] == kRows


async def run_dask_classifier_asyncio(scheduler_address: str) -> None:
    async with Client(scheduler_address, asynchronous=True) as client:
        X, y, _ = generate_array()
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
        prediction = await client.compute(await classifier.predict(X_d))

        assert prediction.ndim == 1
        assert prediction.shape[0] == kRows


def test_with_asyncio() -> None:
    with LocalCluster() as cluster:
        with Client(cluster) as client:
            address = client.scheduler.address
            output = asyncio.run(run_from_dask_array_asyncio(address))
            assert isinstance(output['booster'], xgb.Booster)
            assert isinstance(output['history'], dict)

            asyncio.run(run_dask_regressor_asyncio(address))
            asyncio.run(run_dask_classifier_asyncio(address))


async def generate_concurrent_trainings() -> None:
    async def train() -> None:
        async with LocalCluster(n_workers=2,
                                threads_per_worker=1,
                                asynchronous=True,
                                dashboard_address=0) as cluster:
            async with Client(cluster, asynchronous=True) as client:
                X, y, w = generate_array(with_weights=True)
                dtrain = await DaskDMatrix(client, X, y, weight=w)
                dvalid = await DaskDMatrix(client, X, y, weight=w)
                output = await xgb.dask.train(client, {}, dtrain=dtrain)
                await xgb.dask.predict(client, output, data=dvalid)
    await asyncio.gather(train(), train())


def test_concurrent_trainings() -> None:
    asyncio.run(generate_concurrent_trainings())


def test_predict(client: "Client") -> None:
    X, y, _ = generate_array()
    dtrain = DaskDMatrix(client, X, y)
    booster = xgb.dask.train(client, {}, dtrain, num_boost_round=2)["booster"]

    predt_0 = xgb.dask.predict(client, model=booster, data=dtrain)
    assert predt_0.ndim == 1
    assert predt_0.shape[0] == kRows

    margin = xgb.dask.predict(client, model=booster, data=dtrain, output_margin=True)
    assert margin.ndim == 1
    assert margin.shape[0] == kRows

    shap = xgb.dask.predict(client, model=booster, data=dtrain, pred_contribs=True)
    assert shap.ndim == 2
    assert shap.shape[0] == kRows
    assert shap.shape[1] == kCols + 1

    booster_f = client.scatter(booster, broadcast=True)

    predt_1 = xgb.dask.predict(client, booster_f, X).compute()
    predt_2 = xgb.dask.inplace_predict(client, booster_f, X).compute()
    np.testing.assert_allclose(predt_0, predt_1)
    np.testing.assert_allclose(predt_0, predt_2)


def test_predict_with_meta(client: "Client") -> None:
    X, y, w = generate_array(with_weights=True)
    assert w is not None
    partition_size = 20
    margin = da.random.random(kRows, partition_size) + 1e4

    dtrain = DaskDMatrix(client, X, y, weight=w, base_margin=margin)
    booster: xgb.Booster = xgb.dask.train(
        client, {}, dtrain, num_boost_round=4)['booster']

    prediction = xgb.dask.predict(client, model=booster, data=dtrain)
    assert prediction.ndim == 1
    assert prediction.shape[0] == kRows

    prediction = client.compute(prediction).result()
    assert np.all(prediction > 1e3)

    m = xgb.DMatrix(X.compute())
    m.set_info(label=y.compute(), weight=w.compute(), base_margin=margin.compute())
    single = booster.predict(m)  # Make sure the ordering is correct.
    assert np.all(prediction == single)


def run_aft_survival(client: "Client", dmatrix_t: Type) -> None:
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


def test_dask_aft_survival() -> None:
    with LocalCluster(n_workers=kWorkers) as cluster:
        with Client(cluster) as client:
            run_aft_survival(client, DaskDMatrix)


def test_dask_ranking(client: "Client") -> None:
    dpath = "demo/rank/"
    mq2008 = tm.get_mq2008(dpath)
    data = []
    for d in mq2008:
        if isinstance(d, scipy.sparse.csr_matrix):
            d[d == 0] = np.inf
            d = d.toarray()
            d[d == 0] = np.nan
            d[np.isinf(d)] = 0
            data.append(dd.from_array(d, chunksize=32))
        else:
            data.append(dd.from_array(d, chunksize=32))

    (
        x_train,
        y_train,
        qid_train,
        x_test,
        y_test,
        qid_test,
        x_valid,
        y_valid,
        qid_valid,
    ) = data
    qid_train = qid_train.astype(np.uint32)
    qid_valid = qid_valid.astype(np.uint32)
    qid_test = qid_test.astype(np.uint32)

    rank = xgb.dask.DaskXGBRanker(n_estimators=2500)
    rank.fit(
        x_train,
        y_train,
        qid=qid_train,
        eval_set=[(x_test, y_test), (x_train, y_train)],
        eval_qid=[qid_test, qid_train],
        eval_metric=["ndcg"],
        verbose=True,
        early_stopping_rounds=10,
    )
    assert rank.n_features_in_ == 46
    assert rank.best_score > 0.98


@pytest.mark.parametrize("booster", ["dart", "gbtree"])
def test_dask_predict_leaf(booster: str, client: "Client") -> None:
    from sklearn.datasets import load_digits

    X_, y_ = load_digits(return_X_y=True)
    num_parallel_tree = 4
    X, y = dd.from_array(X_, chunksize=32), dd.from_array(y_, chunksize=32)
    rounds = 4
    cls = xgb.dask.DaskXGBClassifier(
        n_estimators=rounds, num_parallel_tree=num_parallel_tree, booster=booster
    )
    cls.client = client
    cls.fit(X, y)
    leaf = xgb.dask.predict(
        client,
        cls.get_booster(),
        X.to_dask_array(),      # we can't map_blocks on dataframe when output is 4-dim.
        pred_leaf=True,
        strict_shape=True,
        validate_features=False,
    ).compute()

    assert leaf.shape[0] == X_.shape[0]
    assert leaf.shape[1] == rounds
    assert leaf.shape[2] == cls.n_classes_
    assert leaf.shape[3] == num_parallel_tree

    leaf_from_apply = cls.apply(X).reshape(leaf.shape).compute()
    np.testing.assert_allclose(leaf_from_apply, leaf)

    verify_leaf_output(leaf, num_parallel_tree)


def test_dask_iteration_range(client: "Client"):
    X, y, _ = generate_array()
    n_rounds = 10

    Xy = xgb.DMatrix(X.compute(), y.compute())

    dXy = xgb.dask.DaskDMatrix(client, X, y)
    booster = xgb.dask.train(
        client, {"tree_method": "hist"}, dXy, num_boost_round=n_rounds
    )["booster"]

    for i in range(0, n_rounds):
        iter_range = (0, i)
        native_predt = booster.predict(Xy, iteration_range=iter_range)

        with_dask_dmatrix = xgb.dask.predict(
            client, booster, dXy, iteration_range=iter_range
        )
        with_dask_collection = xgb.dask.predict(
            client, booster, X, iteration_range=iter_range
        )
        with_inplace = xgb.dask.inplace_predict(
            client, booster, X, iteration_range=iter_range
        )
        np.testing.assert_allclose(native_predt, with_dask_dmatrix.compute())
        np.testing.assert_allclose(native_predt, with_dask_collection.compute())
        np.testing.assert_allclose(native_predt, with_inplace.compute())

    full_predt = xgb.dask.predict(client, booster, X, iteration_range=(0, n_rounds))
    default = xgb.dask.predict(client, booster, X)
    np.testing.assert_allclose(full_predt.compute(), default.compute())


class TestWithDask:
    @pytest.mark.parametrize('config_key,config_value', [('verbosity', 0), ('use_rmm', True)])
    def test_global_config(
            self,
            client: "Client",
            config_key: str,
            config_value: Any
    ) -> None:
        X, y, _ = generate_array()
        xgb.config.set_config(**{config_key: config_value})
        dtrain = DaskDMatrix(client, X, y)
        before_fname = './before_training-test_global_config'
        after_fname = './after_training-test_global_config'

        class TestCallback(xgb.callback.TrainingCallback):
            def write_file(self, fname: str) -> None:
                with open(fname, 'w') as fd:
                    fd.write(str(xgb.config.get_config()[config_key]))

            def before_training(self, model: xgb.Booster) -> xgb.Booster:
                self.write_file(before_fname)
                assert xgb.config.get_config()[config_key] == config_value
                return model

            def after_training(self, model: xgb.Booster) -> xgb.Booster:
                assert xgb.config.get_config()[config_key] == config_value
                return model

            def before_iteration(
                    self, model: xgb.Booster, epoch: int, evals_log: Dict
            ) -> bool:
                assert xgb.config.get_config()[config_key] == config_value
                return False

            def after_iteration(
                    self, model: xgb.Booster, epoch: int, evals_log: Dict
            ) -> bool:
                self.write_file(after_fname)
                assert xgb.config.get_config()[config_key] == config_value
                return False

        xgb.dask.train(client, {}, dtrain, num_boost_round=4, callbacks=[TestCallback()])[
            'booster']

        with open(before_fname, 'r') as before, open(after_fname, 'r') as after:
            assert before.read() == str(config_value)
            assert after.read() == str(config_value)

        os.remove(before_fname)
        os.remove(after_fname)

    def run_updater_test(
            self,
            client: "Client",
            params: Dict,
            num_rounds: int,
            dataset: tm.TestDataset,
            tree_method: str
    ) -> None:
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

        def is_stump():
            return params["max_depth"] == 1 or params["max_leaves"] == 1

        def minimum_bin():
            return "max_bin" in params and params["max_bin"] == 2

        if minimum_bin() and is_stump():
            assert tm.non_increasing(history, tolerance=1e-3)
        else:
            assert tm.non_increasing(history)
        # Make sure that it's decreasing
        assert history[-1] < history[0]

    @given(params=hist_parameter_strategy,
           dataset=tm.dataset_strategy)
    @settings(deadline=None, suppress_health_check=suppress)
    def test_hist(
            self, params: Dict, dataset: tm.TestDataset, client: "Client"
    ) -> None:
        num_rounds = 30
        self.run_updater_test(client, params, num_rounds, dataset, 'hist')

    @given(params=exact_parameter_strategy,
           dataset=tm.dataset_strategy)
    @settings(deadline=None, suppress_health_check=suppress)
    def test_approx(
            self, client: "Client", params: Dict, dataset: tm.TestDataset
    ) -> None:
        num_rounds = 30
        self.run_updater_test(client, params, num_rounds, dataset, 'approx')

    def run_quantile(self, name: str) -> None:
        if sys.platform.startswith("win"):
            pytest.skip("Skipping dask tests on Windows")

        exe: Optional[str] = None
        for possible_path in {'./testxgboost', './build/testxgboost',
                              '../build/testxgboost',
                              '../cpu-build/testxgboost'}:
            if os.path.exists(possible_path):
                exe = possible_path
        if exe is None:
            return

        test = "--gtest_filter=Quantile." + name

        def runit(
            worker_addr: str, rabit_args: List[bytes]
        ) -> subprocess.CompletedProcess:
            port_env = ''
            # setup environment for running the c++ part.
            for arg in rabit_args:
                if arg.decode('utf-8').startswith('DMLC_TRACKER_PORT'):
                    port_env = arg.decode('utf-8')
            port = port_env.split('=')
            env = os.environ.copy()
            env[port[0]] = port[1]
            return subprocess.run([str(exe), test], env=env, capture_output=True)

        with LocalCluster(n_workers=4) as cluster:
            with Client(cluster) as client:
                workers = _get_client_workers(client)
                rabit_args = client.sync(
                    xgb.dask._get_rabit_args, len(workers), client)
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
    def test_quantile_basic(self) -> None:
        self.run_quantile('DistributedBasic')

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.gtest
    def test_quantile(self) -> None:
        self.run_quantile('Distributed')

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.gtest
    def test_quantile_same_on_all_workers(self) -> None:
        self.run_quantile('SameOnAllWorkers')

    def test_n_workers(self) -> None:
        with LocalCluster(n_workers=2) as cluster:
            with Client(cluster) as client:
                workers = _get_client_workers(client)
                from sklearn.datasets import load_breast_cancer
                X, y = load_breast_cancer(return_X_y=True)
                dX = client.submit(da.from_array, X, workers=[workers[0]]).result()
                dy = client.submit(da.from_array, y, workers=[workers[0]]).result()
                train = xgb.dask.DaskDMatrix(client, dX, dy)

                dX = dd.from_array(X)
                dX = client.persist(dX, workers=workers[1])
                dy = dd.from_array(y)
                dy = client.persist(dy, workers=workers[1])
                valid = xgb.dask.DaskDMatrix(client, dX, dy)

                merged = xgb.dask._get_workers_from_data(train, evals=[(valid, 'Valid')])
                assert len(merged) == 2

    @pytest.mark.skipif(**tm.no_dask())
    def test_feature_weights(self, client: "Client") -> None:
        kRows = 1024
        kCols = 64
        rng = da.random.RandomState(1994)
        X = rng.random_sample((kRows, kCols), chunks=(32, -1))
        y = rng.random_sample(kRows, chunks=32)

        fw = np.ones(shape=(kCols,))
        for i in range(kCols):
            fw[i] *= float(i)
        fw = da.from_array(fw)
        poly_increasing = run_feature_weights(X, y, fw, model=xgb.dask.DaskXGBRegressor)

        fw = np.ones(shape=(kCols,))
        for i in range(kCols):
            fw[i] *= float(kCols - i)
        fw = da.from_array(fw)
        poly_decreasing = run_feature_weights(X, y, fw, model=xgb.dask.DaskXGBRegressor)

        # Approxmated test, this is dependent on the implementation of random
        # number generator in std library.
        assert poly_increasing[0] > 0.08
        assert poly_decreasing[0] < -0.08

    @pytest.mark.skipif(**tm.no_dask())
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_custom_objective(self, client: "Client") -> None:
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)
        rounds = 20

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'log')

            def sqr(
                labels: np.ndarray, predts: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
                with open(path, 'a') as fd:
                    print('Running sqr', file=fd)
                grad = predts - labels
                hess = np.ones(shape=labels.shape[0])
                return grad, hess

            reg = xgb.dask.DaskXGBRegressor(n_estimators=rounds, objective=sqr,
                                            tree_method='hist')
            reg.fit(X, y, eval_set=[(X, y)])

            # Check the obj is ran for rounds.
            with open(path, 'r') as fd:
                out = fd.readlines()
                assert len(out) == rounds

            results_custom = reg.evals_result()

            reg = xgb.dask.DaskXGBRegressor(n_estimators=rounds, tree_method='hist')
            reg.fit(X, y, eval_set=[(X, y)])
            results_native = reg.evals_result()

            np.testing.assert_allclose(results_custom['validation_0']['rmse'],
                                       results_native['validation_0']['rmse'])
            tm.non_increasing(results_native['validation_0']['rmse'])

    def test_no_duplicated_partition(self) -> None:
        '''Assert each worker has the correct amount of data, and DMatrix initialization doesn't
        generate unnecessary copies of data.

        '''
        with LocalCluster(n_workers=2) as cluster:
            with Client(cluster) as client:
                X, y, _ = generate_array()
                n_partitions = X.npartitions
                m = xgb.dask.DaskDMatrix(client, X, y)
                workers = _get_client_workers(client)
                rabit_args = client.sync(xgb.dask._get_rabit_args, len(workers), client)
                n_workers = len(workers)

                def worker_fn(worker_addr: str, data_ref: Dict) -> None:
                    with xgb.dask.RabitContext(rabit_args):
                        local_dtrain = xgb.dask._dmatrix_from_list_of_parts(**data_ref)
                        total = np.array([local_dtrain.num_row()])
                        total = xgb.rabit.allreduce(total, xgb.rabit.Op.SUM)
                        assert total[0] == kRows

                futures = []
                for i in range(len(workers)):
                    futures.append(
                        client.submit(
                            worker_fn, workers[i],
                            m._create_fn_args(workers[i]), pure=False,
                            workers=[workers[i]])
                    )
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

    def test_data_initialization(self, client: "Client") -> None:
        """assert that we don't create duplicated DMatrix"""
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        X, y = dd.from_array(X, chunksize=32), dd.from_array(y, chunksize=32)
        run_data_initialization(xgb.dask.DaskDMatrix, xgb.dask.DaskXGBClassifier, X, y)

    def run_shap(self, X: Any, y: Any, params: Dict[str, Any], client: "Client") -> None:
        rows = X.shape[0]
        cols = X.shape[1]

        def assert_shape(shape: Tuple[int, ...]) -> None:
            assert shape[0] == rows
            if "num_class" in params.keys():
                assert shape[1] == params["num_class"]
                assert shape[2] == cols + 1
            else:
                assert shape[1] == cols + 1

        X, y = da.from_array(X, chunks=(32, -1)), da.from_array(y, chunks=32)
        Xy = xgb.dask.DaskDMatrix(client, X, y)
        booster = xgb.dask.train(client, params, Xy, num_boost_round=10)['booster']

        test_Xy = xgb.dask.DaskDMatrix(client, X, y)

        shap = xgb.dask.predict(client, booster, test_Xy, pred_contribs=True).compute()
        margin = xgb.dask.predict(client, booster, test_Xy, output_margin=True).compute()
        assert_shape(shap.shape)
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-5, 1e-5)

        shap = xgb.dask.predict(client, booster, X, pred_contribs=True).compute()
        margin = xgb.dask.predict(client, booster, X, output_margin=True).compute()
        assert_shape(shap.shape)
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-5, 1e-5)

        if "num_class" not in params.keys():
            X = dd.from_dask_array(X).repartition(npartitions=32)
            y = dd.from_dask_array(y).repartition(npartitions=32)
            shap_df = xgb.dask.predict(
                client, booster, X, pred_contribs=True, validate_features=False
            ).compute()
            assert_shape(shap_df.shape)
            assert np.allclose(
                np.sum(shap_df, axis=len(shap_df.shape) - 1), margin, 1e-5, 1e-5
            )

    def run_shap_cls_sklearn(self, X: Any, y: Any, client: "Client") -> None:
        X, y = da.from_array(X, chunks=(32, -1)), da.from_array(y, chunks=32)
        cls = xgb.dask.DaskXGBClassifier(n_estimators=4)
        cls.client = client
        cls.fit(X, y)
        booster = cls.get_booster()

        test_Xy = xgb.dask.DaskDMatrix(client, X, y)

        shap = xgb.dask.predict(client, booster, test_Xy, pred_contribs=True).compute()
        margin = xgb.dask.predict(client, booster, test_Xy, output_margin=True).compute()
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-5, 1e-5)

        shap = xgb.dask.predict(client, booster, X, pred_contribs=True).compute()
        margin = xgb.dask.predict(client, booster, X, output_margin=True).compute()
        assert np.allclose(np.sum(shap, axis=len(shap.shape) - 1), margin, 1e-5, 1e-5)

    def test_shap(self, client: "Client") -> None:
        from sklearn.datasets import load_boston, load_digits
        X, y = load_boston(return_X_y=True)
        params: Dict[str, Any] = {'objective': 'reg:squarederror'}
        self.run_shap(X, y, params, client)

        X, y = load_digits(return_X_y=True)
        params = {'objective': 'multi:softmax', 'num_class': 10}
        self.run_shap(X, y, params, client)

        params = {'objective': 'multi:softprob', 'num_class': 10}
        self.run_shap(X, y, params, client)

        self.run_shap_cls_sklearn(X, y, client)

    def run_shap_interactions(
        self,
        X: Any,
        y: Any,
        params: Dict[str, Any],
        client: "Client"
    ) -> None:
        rows = X.shape[0]
        cols = X.shape[1]
        X, y = da.from_array(X, chunks=(32, -1)), da.from_array(y, chunks=32)

        Xy = xgb.dask.DaskDMatrix(client, X, y)
        booster = xgb.dask.train(client, params, Xy, num_boost_round=10)['booster']

        test_Xy = xgb.dask.DaskDMatrix(client, X, y)

        shap = xgb.dask.predict(
            client, booster, test_Xy, pred_interactions=True
        ).compute()

        assert len(shap.shape) == 3
        assert shap.shape[0] == rows
        assert shap.shape[1] == cols + 1
        assert shap.shape[2] == cols + 1

        margin = xgb.dask.predict(client, booster, test_Xy, output_margin=True).compute()
        assert np.allclose(np.sum(shap, axis=(len(shap.shape) - 1, len(shap.shape) - 2)),
                           margin,
                           1e-5, 1e-5)

    def test_shap_interactions(self, client: "Client") -> None:
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        params = {'objective': 'reg:squarederror'}
        self.run_shap_interactions(X, y, params, client)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_sklearn_io(self, client: 'Client') -> None:
        from sklearn.datasets import load_digits
        X_, y_ = load_digits(return_X_y=True)
        X, y = da.from_array(X_), da.from_array(y_)
        cls = xgb.dask.DaskXGBClassifier(n_estimators=10)
        cls.client = client
        cls.fit(X, y)
        predt_0 = cls.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            with open(path, "wb") as fd:
                pickle.dump(cls, fd)

            with open(path, "rb") as fd:
                cls = pickle.load(fd)
            predt_1 = cls.predict(X)
            np.testing.assert_allclose(predt_0.compute(), predt_1.compute())

            path = os.path.join(tmpdir, 'cls.json')
            cls.save_model(path)

            cls = xgb.dask.DaskXGBClassifier()
            cls.load_model(path)
            assert cls.n_classes_ == 10
            predt_2 = cls.predict(X)

            np.testing.assert_allclose(predt_0.compute(), predt_2.compute())

            # Use single node to load
            cls = xgb.XGBClassifier()
            cls.load_model(path)
            assert cls.n_classes_ == 10
            predt_3 = cls.predict(X_)

            np.testing.assert_allclose(predt_0.compute(), predt_3)


def test_dask_unsupported_features(client: "Client") -> None:
    X, y, _ = generate_array()
    # gblinear doesn't support distributed training.
    with pytest.raises(NotImplementedError, match="gblinear"):
        xgb.dask.train(
            client, {"booster": "gblinear"}, xgb.dask.DaskDMatrix(client, X, y)
        )


def test_parallel_submits(client: "Client") -> None:
    """Test for running multiple train simultaneously from single clients."""
    try:
        from distributed import MultiLock  # NOQA
    except ImportError:
        pytest.skip("`distributed.MultiLock' is not available")

    from sklearn.datasets import load_digits

    futures = []
    workers = _get_client_workers(client)
    n_submits = len(workers)
    for i in range(n_submits):
        X_, y_ = load_digits(return_X_y=True)
        X = dd.from_array(X_, chunksize=32)
        y = dd.from_array(y_, chunksize=32)
        cls = xgb.dask.DaskXGBClassifier(
            verbosity=1,
            n_estimators=i + 1,
            eval_metric="merror",
            use_label_encoder=False,
        )
        f = client.submit(cls.fit, X, y, pure=False)
        futures.append(f)

    classifiers = client.gather(futures)
    assert len(classifiers) == n_submits
    for i, cls in enumerate(classifiers):
        assert cls.get_booster().num_boosted_rounds() == i + 1


def run_tree_stats(client: Client, tree_method: str) -> str:
    """assert that different workers count dosn't affect summ statistic's on root"""

    def dask_train(X, y, num_obs, num_features):
        chunk_size = 100
        X = da.from_array(X, chunks=(chunk_size, num_features))
        y = da.from_array(y.reshape(num_obs, 1), chunks=(chunk_size, 1))
        dtrain = xgb.dask.DaskDMatrix(client, X, y)

        output = xgb.dask.train(
            client,
            {
                "verbosity": 0,
                "tree_method": tree_method,
                "objective": "reg:squarederror",
                "max_depth": 3,
            },
            dtrain,
            num_boost_round=1,
        )
        dump_model = output["booster"].get_dump(with_stats=True, dump_format="json")[0]
        return json.loads(dump_model)

    num_obs = 1000
    num_features = 10
    X, y = make_regression(num_obs, num_features, random_state=777)
    model = dask_train(X, y, num_obs, num_features)

    # asserts children have correct cover.
    stack = [model]
    while stack:
        node: dict = stack.pop()
        if "leaf" in node.keys():
            continue
        cover = 0
        for c in node["children"]:
            cover += c["cover"]
            stack.append(c)
        assert cover == node["cover"]

    return model["cover"]


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_tree_stats(tree_method: str) -> None:
    with LocalCluster(n_workers=1) as cluster:
        with Client(cluster) as client:
            local = run_tree_stats(client, tree_method)
    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster) as client:
            distributed = run_tree_stats(client, tree_method)

    assert local == distributed


def test_parallel_submit_multi_clients() -> None:
    """Test for running multiple train simultaneously from multiple clients."""
    try:
        from distributed import MultiLock  # NOQA
    except ImportError:
        pytest.skip("`distributed.MultiLock' is not available")

    from sklearn.datasets import load_digits

    with LocalCluster(n_workers=4) as cluster:
        with Client(cluster) as client:
            workers = _get_client_workers(client)

        n_submits = len(workers)
        assert n_submits == 4
        futures = []

        for i in range(n_submits):
            client = Client(cluster)
            X_, y_ = load_digits(return_X_y=True)
            X_ += 1.0
            X = dd.from_array(X_, chunksize=32)
            y = dd.from_array(y_, chunksize=32)
            cls = xgb.dask.DaskXGBClassifier(
                verbosity=1,
                n_estimators=i + 1,
                eval_metric="merror",
                use_label_encoder=False,
            )
            f = client.submit(cls.fit, X, y, pure=False)
            futures.append((client, f))

        t_futures = []
        with ThreadPoolExecutor(max_workers=16) as e:
            for i in range(n_submits):
                def _() -> xgb.dask.DaskXGBClassifier:
                    return futures[i][0].compute(futures[i][1]).result()

                f = e.submit(_)
                t_futures.append(f)

        for i, f in enumerate(t_futures):
            assert f.result().get_booster().num_boosted_rounds() == i + 1


class TestDaskCallbacks:
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_early_stopping(self, client: "Client") -> None:
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)
        m = xgb.dask.DaskDMatrix(client, X, y)

        valid = xgb.dask.DaskDMatrix(client, X, y)
        early_stopping_rounds = 5
        booster = xgb.dask.train(client, {'objective': 'binary:logistic',
                                          'eval_metric': 'error',
                                          'tree_method': 'hist'}, m,
                                 evals=[(valid, 'Valid')],
                                 num_boost_round=1000,
                                 early_stopping_rounds=early_stopping_rounds)['booster']
        assert hasattr(booster, 'best_score')
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

        valid_X, valid_y = load_breast_cancer(return_X_y=True)
        valid_X, valid_y = da.from_array(valid_X), da.from_array(valid_y)
        cls = xgb.dask.DaskXGBClassifier(objective='binary:logistic', tree_method='hist',
                                         n_estimators=1000)
        cls.client = client
        cls.fit(X, y, early_stopping_rounds=early_stopping_rounds,
                eval_set=[(valid_X, valid_y)])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

        # Specify the metric
        cls = xgb.dask.DaskXGBClassifier(objective='binary:logistic', tree_method='hist',
                                         n_estimators=1000)
        cls.client = client
        cls.fit(X, y, early_stopping_rounds=early_stopping_rounds,
                eval_set=[(valid_X, valid_y)], eval_metric='error')
        assert tm.non_increasing(cls.evals_result()['validation_0']['error'])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format='json')
        assert len(cls.evals_result()['validation_0']['error']) < 20
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_early_stopping_custom_eval(self, client: "Client") -> None:
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)
        m = xgb.dask.DaskDMatrix(client, X, y)

        valid = xgb.dask.DaskDMatrix(client, X, y)
        early_stopping_rounds = 5
        booster = xgb.dask.train(
            client, {'objective': 'binary:logistic',
                     'eval_metric': 'error',
                     'tree_method': 'hist'}, m,
            evals=[(m, 'Train'), (valid, 'Valid')],
            feval=tm.eval_error_metric,
            num_boost_round=1000,
            early_stopping_rounds=early_stopping_rounds)['booster']
        assert hasattr(booster, 'best_score')
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

        valid_X, valid_y = load_breast_cancer(return_X_y=True)
        valid_X, valid_y = da.from_array(valid_X), da.from_array(valid_y)
        cls = xgb.dask.DaskXGBClassifier(objective='binary:logistic', tree_method='hist',
                                         n_estimators=1000)
        cls.client = client
        cls.fit(X, y, early_stopping_rounds=early_stopping_rounds,
                eval_set=[(valid_X, valid_y)], eval_metric=tm.eval_error_metric)
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_callback(self, client: "Client") -> None:
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        X, y = da.from_array(X), da.from_array(y)

        cls = xgb.dask.DaskXGBClassifier(objective='binary:logistic', tree_method='hist',
                                         n_estimators=10)
        cls.client = client

        with tempfile.TemporaryDirectory() as tmpdir:
            cls.fit(X, y, callbacks=[xgb.callback.TrainingCheckPoint(
                directory=Path(tmpdir),
                iterations=1,
                name='model'
            )])
            for i in range(1, 10):
                assert os.path.exists(
                    os.path.join(tmpdir, 'model_' + str(i) + '.json'))
