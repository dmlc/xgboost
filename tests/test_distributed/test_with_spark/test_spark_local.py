import glob
import logging
import os
import random
import tempfile
from collections import namedtuple
from typing import Generator, Iterable, List, Sequence

import numpy as np
import pytest
import xgboost as xgb
from pyspark import SparkConf
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as spark_sql_func
from xgboost import XGBClassifier, XGBModel, XGBRegressor
from xgboost import testing as tm
from xgboost.callback import LearningRateScheduler
from xgboost.collective import Config
from xgboost.spark import (
    SparkXGBClassifier,
    SparkXGBClassifierModel,
    SparkXGBRanker,
    SparkXGBRegressor,
    SparkXGBRegressorModel,
)
from xgboost.spark.core import _non_booster_params
from xgboost.spark.data import pred_contribs
from xgboost.testing.collective import get_avail_port

from .utils import SparkTestCase

logging.getLogger("py4j").setLevel(logging.INFO)

pytestmark = [tm.timeout(60), pytest.mark.skipif(**tm.no_spark())]

FAST_N_ESTIMATORS = 10

RegData = namedtuple(
    "RegData",
    (
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "weights",
        "base_margin",
        "is_val",
        "X",
        "y",
        "df",
    ),
)


@pytest.fixture(scope="module")
def reg_data(
    spark: SparkSession,
) -> RegData:
    rng = np.random.default_rng(seed=42)
    X = rng.random((100, 10))
    # Make odd rows sparse with some random values to test both dense and sparse paths in the Spark estimator.
    X[1::2, :] = 0.0
    X[1::2, 1] = rng.random(len(X[1::2, 1]))
    X[1::2, 2] = rng.random(len(X[1::2, 2]))
    y = rng.random(100)
    w = rng.random(100)
    base_margin = rng.random(100)
    is_val = rng.random(100) < 0.2
    X_train, X_test = X[~is_val], X[is_val]
    y_train, y_test = y[~is_val], y[is_val]
    rows = []
    for i in range(len(y)):
        vec = (
            Vectors.dense(X[i, :])
            if i % 2 == 0
            else Vectors.sparse(X.shape[1], {1: float(X[i, 1]), 2: float(X[i, 2])})
        )
        rows.append(
            (
                i,
                vec,
                float(y[i]),
                float(w[i]),
                float(base_margin[i]),
                bool(is_val[i]),
            )
        )
    df = spark.createDataFrame(
        rows, ["row_id", "features", "label", "weight", "base_margin", "is_val"]
    )
    return RegData(X_train, X_test, y_train, y_test, w, base_margin, is_val, X, y, df)


def no_sparse_unwrap() -> tm.PytestSkip:
    try:
        from pyspark.sql.functions import unwrap_udt

    except ImportError:
        return {"reason": "PySpark<3.4", "condition": True}

    return {"reason": "PySpark<3.4", "condition": False}


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    os.environ["XGBOOST_PYSPARK_SHARED_SESSION"] = "1"
    config = {
        "spark.master": "local[4]",
        "spark.python.worker.reuse": "true",
        "spark.driver.host": "127.0.0.1",
        "spark.task.maxFailures": "1",
        "spark.sql.shuffle.partitions": "4",
        "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
        "spark.sql.pyspark.jvmStacktrace.enabled": "true",
        "spark.ui.enabled": "false",
    }

    builder = SparkSession.builder.appName("XGBoost PySpark Python API Tests")
    for k, v in config.items():
        builder.config(k, v)
    logging.getLogger("pyspark").setLevel(logging.INFO)
    sess = builder.getOrCreate()
    try:
        yield sess
    finally:
        sess.stop()
        sess.sparkContext.stop()
        os.environ.pop("XGBOOST_PYSPARK_SHARED_SESSION", None)


def test_regressor(
    reg_data: RegData,
) -> None:
    train_rows = np.where(~reg_data.is_val)[0]
    validation_rows = np.where(reg_data.is_val)[0]

    # Setup: dataset with weights and an explicit validation split.
    reg_param = {
        "n_estimators": 10,
        "max_depth": 5,
        "objective": "reg:squarederror",
        "max_bin": 9,
        "eval_metric": "rmse",
        "early_stopping_rounds": 1,
    }
    # Train a reference sklearn model on the explicit train/validation split.
    reg = XGBRegressor(**reg_param).fit(
        reg_data.X_train,
        reg_data.y_train,
        sample_weight=reg_data.weights[train_rows],
        eval_set=[(reg_data.X_test, reg_data.y_test)],
        sample_weight_eval_set=[reg_data.weights[validation_rows]],
    )
    # Train Spark estimator using the same split via validation_indicator_col.
    spark_regressor = SparkXGBRegressor(
        pred_contrib_col="pred_contribs",
        weight_col="weight",
        validation_indicator_col="is_val",
        **reg_param,
    ).fit(reg_data.df)
    pred_result = spark_regressor.transform(reg_data.df)
    preds = (
        pred_result.orderBy("row_id")
        .select("prediction")
        .toPandas()["prediction"]
        .to_numpy()
    )
    pred_contribs = np.array(
        pred_result.orderBy("row_id")
        .select("pred_contribs")
        .toPandas()["pred_contribs"]
        .tolist()
    )
    # Prediction parity with sklearn reference.
    assert np.allclose(preds, reg.predict(reg_data.X), rtol=1e-3)
    # Contribs should sum to prediction; direct equality can be noisy due to
    # precision differences between Spark UDF inference and in-process predict.
    assert np.allclose(pred_contribs.sum(axis=1), preds, rtol=1e-3)
    # Eval result parity on the validation set.
    assert np.allclose(
        reg.evals_result()["validation_0"]["rmse"],
        spark_regressor.training_summary.validation_objective_history["rmse"],
        atol=1e-6,
    )
    # Best score parity (early stopping uses the validation set).
    assert np.allclose(
        reg.best_score, spark_regressor._xgb_sklearn_model.best_score, atol=1e-3
    )


def test_training_continuation(
    reg_data: RegData,
) -> None:
    params = {
        "n_estimators": 2,
        "max_depth": 3,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }

    base = SparkXGBRegressor(n_estimators=2, **params).fit(reg_data.df)
    continued = SparkXGBRegressor(
        n_estimators=4, xgb_model=base.get_booster(), **params
    ).fit(reg_data.df)

    preds_base = (
        base.transform(reg_data.df)
        .select("prediction")
        .toPandas()["prediction"]
        .to_numpy()
    )
    preds_cont = (
        continued.transform(reg_data.df)
        .select("prediction")
        .toPandas()["prediction"]
        .to_numpy()
    )

    ref_base = XGBRegressor(n_estimators=2, **params).fit(reg_data.X, reg_data.y)
    ref_cont = XGBRegressor(n_estimators=4, **params).fit(
        reg_data.X, reg_data.y, xgb_model=ref_base.get_booster()
    )

    assert np.allclose(preds_cont, ref_cont.predict(reg_data.X), rtol=1e-3)
    assert not np.allclose(preds_base, preds_cont, rtol=1e-6)


def test_regressor_with_base_margin(
    reg_data: RegData,
) -> None:
    params = {
        "n_estimators": 5,
        "max_depth": 3,
        "objective": "reg:squarederror",
    }
    spark_model = SparkXGBRegressor(base_margin_col="base_margin", **params).fit(
        reg_data.df
    )
    preds = (
        spark_model.transform(reg_data.df)
        .orderBy("row_id")
        .select("prediction")
        .toPandas()["prediction"]
        .to_numpy()
    )

    ref = XGBRegressor(**params).fit(
        reg_data.X, reg_data.y, base_margin=reg_data.base_margin
    )
    expected = ref.predict(reg_data.X, base_margin=reg_data.base_margin)

    assert np.allclose(preds, expected, rtol=1e-3)


def test_regressor_save_load(reg_data: RegData) -> None:
    train_df = reg_data.df.select("features", "label")
    model = SparkXGBRegressor(n_estimators=5, max_depth=3).fit(train_df)
    preds_before = (
        model.transform(train_df)
        .select("prediction")
        .toPandas()["prediction"]
        .to_numpy()
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "spark-xgb-reg-model")
        model.save(path)
        loaded = SparkXGBRegressorModel.load(path)
        preds_after = (
            loaded.transform(train_df)
            .select("prediction")
            .toPandas()["prediction"]
            .to_numpy()
        )

    assert np.allclose(preds_before, preds_after, rtol=1e-6)


def test_regressor_params() -> None:
    py_reg = SparkXGBRegressor()
    assert hasattr(py_reg, "n_estimators")
    assert py_reg.n_estimators.parent == py_reg.uid
    assert not hasattr(py_reg, "gpu_id")
    assert hasattr(py_reg, "device")
    assert py_reg.getOrDefault(py_reg.n_estimators) == 100
    assert py_reg.getOrDefault(getattr(py_reg, "objective")), "reg:squarederror"
    py_reg2 = SparkXGBRegressor(n_estimators=200)
    assert py_reg2.getOrDefault(getattr(py_reg2, "n_estimators")), 200
    py_reg3 = py_reg2.copy({getattr(py_reg2, "max_depth"): 10})
    assert py_reg3.getOrDefault(getattr(py_reg3, "n_estimators")), 200
    assert py_reg3.getOrDefault(getattr(py_reg3, "max_depth")), 10
    with pytest.raises(ValueError, match="Number of workers"):
        SparkXGBRegressor(num_workers=-1)._validate_params()


def test_valid_type(spark: SparkSession) -> None:
    # Validation indicator must be boolean.
    df_train = spark.createDataFrame(
        [
            (Vectors.dense(1.0, 2.0, 3.0), 0, 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, 0),
            (Vectors.dense(4.0, 5.0, 6.0), 0, 1),
            (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 1),
        ],
        ["features", "label", "isVal"],
    )
    reg = SparkXGBRegressor(
        features_col="features",
        label_col="label",
        validation_indicator_col="isVal",
    )
    with pytest.raises(TypeError, match="The validation indicator must be boolean"):
        reg.fit(df_train)


def test_callbacks(reg_data: RegData) -> None:
    train_df = reg_data.df.select("features", "label")

    def custom_lr(boosting_round: int) -> float:
        return 1.0 / (boosting_round + 1)

    cb = [LearningRateScheduler(custom_lr)]
    reg_params = {
        "n_estimators": FAST_N_ESTIMATORS,
        "max_depth": 3,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "spark-xgb-reg-cb")
        regressor = SparkXGBRegressor(callbacks=cb, **reg_params)
        # Callbacks are estimator-only; ensure they survive save/load.
        regressor.save(path)
        regressor = SparkXGBRegressor.load(path)
        loaded_callbacks = regressor.getOrDefault(regressor.callbacks)
        assert loaded_callbacks is not None
        assert len(loaded_callbacks) == 1

        model = regressor.fit(train_df)
        preds = (
            model.transform(train_df)
            .select("prediction")
            .toPandas()["prediction"]
            .to_numpy()
        )

    assert preds.shape == (len(reg_data.y),)
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_empty_train_data(spark: SparkSession, tree_method: str) -> None:
    df_train = spark.createDataFrame(
        [
            (Vectors.dense(10.1, 11.2, 11.3), 0, True),
            (Vectors.dense(1, 1.2, 1.3), 1, True),
            (Vectors.dense(14.0, 15.0, 16.0), 0, True),
            (Vectors.dense(1.1, 1.2, 1.3), 1, False),
        ],
        ["features", "label", "val_col"],
    )
    classifier = SparkXGBRegressor(
        num_workers=2,
        min_child_weight=0.0,
        reg_alpha=0,
        reg_lambda=0,
        tree_method=tree_method,
        validation_indicator_col="val_col",
    )
    model = classifier.fit(df_train)
    pred_result = model.transform(df_train).collect()
    for row in pred_result:
        assert row.prediction == 1.0


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_empty_validation_data(spark: SparkSession, tree_method: str) -> None:
    df_train = spark.createDataFrame(
        [
            (Vectors.dense(10.1, 11.2, 11.3), 0, False),
            (Vectors.dense(1, 1.2, 1.3), 1, False),
            (Vectors.dense(14.0, 15.0, 16.0), 0, False),
            (Vectors.dense(1.1, 1.2, 1.3), 1, True),
        ],
        ["features", "label", "val_col"],
    )
    classifier = SparkXGBClassifier(
        num_workers=2,
        tree_method=tree_method,
        min_child_weight=0.0,
        reg_alpha=0,
        reg_lambda=0,
        validation_indicator_col="val_col",
        n_estimators=FAST_N_ESTIMATORS,
    )
    model = classifier.fit(df_train)
    pred_result = model.transform(df_train).collect()
    for row in pred_result:
        assert row.prediction == row.label


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
def test_empty_partition(spark: SparkSession, tree_method: str) -> None:
    # raw_df.repartition(4) will result int severe data skew, actually,
    # there is no any data in reducer partition 1, reducer partition 2
    # see https://github.com/dmlc/xgboost/issues/8221
    raw_df = spark.range(0, 40, 1, 50).withColumn(
        "label",
        spark_sql_func.when(spark_sql_func.rand(1) > 0.5, 1).otherwise(0),
    )
    vector_assembler = VectorAssembler().setInputCols(["id"]).setOutputCol("features")
    data_trans = vector_assembler.setHandleInvalid("keep").transform(raw_df)
    classifier = SparkXGBClassifier(
        tree_method=tree_method,
        n_estimators=FAST_N_ESTIMATORS,
    )
    model = classifier.fit(data_trans)
    pred_result = model.transform(data_trans).collect()
    for row in pred_result:
        assert row.prediction in [0.0, 1.0]


MultiClfData = namedtuple("MultiClfData", ("multi_clf_df_train", "multi_clf_df_test"))


@pytest.fixture(scope="module")
def multi_clf_data(spark: SparkSession) -> Generator[MultiClfData, None, None]:
    X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [0.0, 1.0, 5.5], [-1.0, -2.0, 1.0]])
    y = np.array([0, 0, 1, 2])
    cls1 = xgb.XGBClassifier(n_estimators=FAST_N_ESTIMATORS)
    cls1.fit(X, y)
    predt0 = cls1.predict(X)
    proba0: np.ndarray = cls1.predict_proba(X)
    pred_contrib0: np.ndarray = pred_contribs(cls1, X, None, False)

    # convert np array to pyspark dataframe
    multi_cls_df_train_data = [
        (Vectors.dense(X[0, :]), int(y[0])),
        (Vectors.dense(X[1, :]), int(y[1])),
        (Vectors.sparse(3, {1: float(X[2, 1]), 2: float(X[2, 2])}), int(y[2])),
        (Vectors.dense(X[3, :]), int(y[3])),
    ]
    multi_clf_df_train = spark.createDataFrame(
        multi_cls_df_train_data, ["features", "label"]
    )

    multi_clf_df_test = spark.createDataFrame(
        [
            (
                Vectors.dense(X[0, :]),
                float(predt0[0]),
                proba0[0, :].tolist(),
                pred_contrib0[0, :].tolist(),
            ),
            (
                Vectors.dense(X[1, :]),
                float(predt0[1]),
                proba0[1, :].tolist(),
                pred_contrib0[1, :].tolist(),
            ),
            (
                Vectors.sparse(3, {1: 1.0, 2: 5.5}),
                float(predt0[2]),
                proba0[2, :].tolist(),
                pred_contrib0[2, :].tolist(),
            ),
        ],
        [
            "features",
            "expected_prediction",
            "expected_probability",
            "expected_pred_contribs",
        ],
    )
    yield MultiClfData(multi_clf_df_train, multi_clf_df_test)


ClfWithWeight = namedtuple(
    "ClfWithWeight",
    (
        "cls_params_with_eval",
        "cls_df_train_with_eval_weight",
        "cls_df_test_with_eval_weight",
        "cls_with_eval_best_score",
        "cls_with_eval_and_weight_best_score",
        "cls_expected_evals_result_train",
        "cls_expected_evals_result_validation",
    ),
)


@pytest.fixture(scope="module")
def clf_with_weight(
    spark: SparkSession,
) -> Generator[ClfWithWeight, SparkSession, None]:
    """Test classifier with weight and eval set."""

    X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5], [4.0, 5.0, 6.0], [0.0, 6.0, 7.5]])
    w = np.array([1.0, 2.0, 1.0, 2.0])
    y = np.array([0, 1, 0, 1])
    cls1 = XGBClassifier(n_estimators=FAST_N_ESTIMATORS)
    cls1.fit(X, y, sample_weight=w)

    X_train = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
    X_val = np.array([[4.0, 5.0, 6.0], [0.0, 6.0, 7.5]])
    y_train = np.array([0, 1])
    y_val = np.array([0, 1])
    w_train = np.array([1.0, 2.0])
    w_val = np.array([1.0, 2.0])
    cls2 = XGBClassifier(
        eval_metric="logloss",
        early_stopping_rounds=1,
        n_estimators=FAST_N_ESTIMATORS,
    )
    cls2.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )

    cls3 = XGBClassifier(
        eval_metric="logloss",
        early_stopping_rounds=1,
        n_estimators=FAST_N_ESTIMATORS,
    )
    cls3.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
    )

    cls4 = XGBClassifier(eval_metric="logloss", n_estimators=FAST_N_ESTIMATORS)
    cls4.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
    )
    cls4_evals_result = cls4.evals_result()
    cls_expected_evals_result_train = cls4_evals_result["validation_0"]["logloss"]
    cls_expected_evals_result_validation = cls4_evals_result["validation_1"]["logloss"]

    cls_df_train_with_eval_weight = spark.createDataFrame(
        [
            (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
            (Vectors.dense(4.0, 5.0, 6.0), 0, True, 1.0),
            (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, True, 2.0),
        ],
        ["features", "label", "isVal", "weight"],
    )
    cls_params_with_eval = {
        "validation_indicator_col": "isVal",
        "early_stopping_rounds": 1,
        "eval_metric": "logloss",
        "n_estimators": FAST_N_ESTIMATORS,
    }
    cls_df_test_with_eval_weight = spark.createDataFrame(
        [
            (
                Vectors.dense(1.0, 2.0, 3.0),
                [float(p) for p in cls1.predict_proba(X)[0, :]],
                [float(p) for p in cls2.predict_proba(X)[0, :]],
                [float(p) for p in cls3.predict_proba(X)[0, :]],
            ),
        ],
        [
            "features",
            "expected_prob_with_weight",
            "expected_prob_with_eval",
            "expected_prob_with_weight_and_eval",
        ],
    )
    cls_with_eval_best_score = cls2.best_score
    cls_with_eval_and_weight_best_score = cls3.best_score
    yield ClfWithWeight(
        cls_params_with_eval,
        cls_df_train_with_eval_weight,
        cls_df_test_with_eval_weight,
        cls_with_eval_best_score,
        cls_with_eval_and_weight_best_score,
        cls_expected_evals_result_train,
        cls_expected_evals_result_validation,
    )


ClfData = namedtuple(
    "ClfData", ("cls_params", "cls_df_train", "cls_df_train_large", "cls_df_test")
)


@pytest.fixture(scope="module")
def clf_data(spark: SparkSession) -> Generator[ClfData, None, None]:
    cls_params = {"max_depth": 5, "n_estimators": 10, "scale_pos_weight": 4}

    X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
    y = np.array([0, 1])
    cl1 = xgb.XGBClassifier(n_estimators=FAST_N_ESTIMATORS)
    cl1.fit(X, y)
    predt0 = cl1.predict(X)
    proba0: np.ndarray = cl1.predict_proba(X)
    pred_contrib0: np.ndarray = pred_contribs(cl1, X, None, True)
    cl2 = xgb.XGBClassifier(**cls_params)
    cl2.fit(X, y)
    predt1 = cl2.predict(X)
    proba1: np.ndarray = cl2.predict_proba(X)
    pred_contrib1: np.ndarray = pred_contribs(cl2, X, None, True)

    # convert np array to pyspark dataframe
    cls_df_train_data = [
        (Vectors.dense(X[0, :]), int(y[0])),
        (Vectors.sparse(3, {1: float(X[1, 1]), 2: float(X[1, 2])}), int(y[1])),
    ]
    cls_df_train = spark.createDataFrame(cls_df_train_data, ["features", "label"])

    cls_df_train_large = spark.createDataFrame(
        cls_df_train_data * 20, ["features", "label"]
    )

    cls_df_test = spark.createDataFrame(
        [
            (
                Vectors.dense(X[0, :]),
                int(predt0[0]),
                proba0[0, :].tolist(),
                pred_contrib0[0, :].tolist(),
                int(predt1[0]),
                proba1[0, :].tolist(),
                pred_contrib1[0, :].tolist(),
            ),
            (
                Vectors.sparse(3, {1: 1.0, 2: 5.5}),
                int(predt0[1]),
                proba0[1, :].tolist(),
                pred_contrib0[1, :].tolist(),
                int(predt1[1]),
                proba1[1, :].tolist(),
                pred_contrib1[1, :].tolist(),
            ),
        ],
        [
            "features",
            "expected_prediction",
            "expected_probability",
            "expected_pred_contribs",
            "expected_prediction_with_params",
            "expected_probability_with_params",
            "expected_pred_contribs_with_params",
        ],
    )
    yield ClfData(cls_params, cls_df_train, cls_df_train_large, cls_df_test)


def assert_model_compatible(model: XGBModel, model_path: str) -> None:
    bst = xgb.Booster()
    path = glob.glob(f"{model_path}/**/model/part-00000", recursive=True)[0]
    bst.load_model(path)
    np.testing.assert_equal(
        np.array(model.get_booster().save_raw("json")), np.array(bst.save_raw("json"))
    )


def check_sub_dict_match(
    sub_dist: dict, whole_dict: dict, excluding_keys: Sequence[str]
) -> None:
    for k in sub_dist:
        if k not in excluding_keys:
            assert k in whole_dict, f"check on {k} failed"
            assert sub_dist[k] == whole_dict[k], f"check on {k} failed"


def get_params_map(
    params_kv: dict, estimator: xgb.spark.core._SparkXGBEstimator
) -> dict:
    return {getattr(estimator, k): v for k, v in params_kv.items()}


class TestPySparkLocal:
    def test_multi_classifier_basic(self, multi_clf_data: MultiClfData) -> None:
        cls = SparkXGBClassifier(
            pred_contrib_col="pred_contribs", n_estimators=FAST_N_ESTIMATORS
        )
        model = cls.fit(multi_clf_data.multi_clf_df_train)
        pred_result = model.transform(multi_clf_data.multi_clf_df_test).collect()
        for row in pred_result:
            np.testing.assert_equal(row.prediction, row.expected_prediction)
            np.testing.assert_allclose(
                row.probability, row.expected_probability, rtol=1e-3
            )
            np.testing.assert_allclose(
                row.pred_contribs, row.expected_pred_contribs, atol=1e-3
            )

    def test_classifier_with_weight_eval(self, clf_with_weight: ClfWithWeight) -> None:
        # with weight
        classifier_with_weight = SparkXGBClassifier(
            weight_col="weight", n_estimators=FAST_N_ESTIMATORS
        )
        model_with_weight = classifier_with_weight.fit(
            clf_with_weight.cls_df_train_with_eval_weight
        )
        pred_result_with_weight = model_with_weight.transform(
            clf_with_weight.cls_df_test_with_eval_weight
        ).collect()
        for row in pred_result_with_weight:
            assert np.allclose(
                row.probability, row.expected_prob_with_weight, atol=1e-3
            )

        # with eval
        classifier_with_eval = SparkXGBClassifier(
            **clf_with_weight.cls_params_with_eval
        )
        model_with_eval = classifier_with_eval.fit(
            clf_with_weight.cls_df_train_with_eval_weight
        )
        assert np.isclose(
            model_with_eval._xgb_sklearn_model.best_score,
            clf_with_weight.cls_with_eval_best_score,
            atol=1e-3,
        )
        pred_result_with_eval = model_with_eval.transform(
            clf_with_weight.cls_df_test_with_eval_weight
        ).collect()
        for row in pred_result_with_eval:
            assert np.allclose(row.probability, row.expected_prob_with_eval, atol=1e-3)
        # with weight and eval
        classifier_with_weight_eval = SparkXGBClassifier(
            weight_col="weight", **clf_with_weight.cls_params_with_eval
        )
        model_with_weight_eval = classifier_with_weight_eval.fit(
            clf_with_weight.cls_df_train_with_eval_weight
        )
        pred_result_with_weight_eval = model_with_weight_eval.transform(
            clf_with_weight.cls_df_test_with_eval_weight
        ).collect()
        np.testing.assert_allclose(
            model_with_weight_eval._xgb_sklearn_model.best_score,
            clf_with_weight.cls_with_eval_and_weight_best_score,
            atol=1e-3,
        )

        for row in pred_result_with_weight_eval:
            np.testing.assert_allclose(
                row.probability, row.expected_prob_with_weight_and_eval, atol=1e-3
            )

    def test_classifier_model_save_load(self, clf_data: ClfData) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = "file:" + tmpdir
            clf = SparkXGBClassifier(**clf_data.cls_params)
            model = clf.fit(clf_data.cls_df_train)
            model.save(path)
            loaded_model = SparkXGBClassifierModel.load(path)
            assert model.uid == loaded_model.uid
            for k, v in clf_data.cls_params.items():
                assert loaded_model.getOrDefault(k) == v

            pred_result = loaded_model.transform(clf_data.cls_df_test).collect()
            for row in pred_result:
                np.testing.assert_allclose(
                    row.probability, row.expected_probability_with_params, atol=1e-3
                )

            with pytest.raises(AssertionError, match="Expected class name"):
                SparkXGBRegressorModel.load(path)

            assert_model_compatible(model, tmpdir)

    def test_classifier_basic(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(
            **clf_data.cls_params, pred_contrib_col="pred_contrib"
        )
        model = classifier.fit(clf_data.cls_df_train)
        pred_result = model.transform(clf_data.cls_df_test).collect()
        for row in pred_result:
            np.testing.assert_equal(row.prediction, row.expected_prediction_with_params)
            np.testing.assert_allclose(
                row.probability, row.expected_probability_with_params, rtol=1e-3
            )
            np.testing.assert_equal(
                row.pred_contrib, row.expected_pred_contribs_with_params
            )

    def test_classifier_with_params(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(**clf_data.cls_params)
        all_params = dict(
            **(classifier._gen_xgb_params_dict()),
            **(classifier._gen_fit_params_dict()),
            **(classifier._gen_predict_params_dict()),
        )
        check_sub_dict_match(
            clf_data.cls_params, all_params, excluding_keys=_non_booster_params
        )

        model = classifier.fit(clf_data.cls_df_train)
        all_params = dict(
            **(model._gen_xgb_params_dict()),
            **(model._gen_fit_params_dict()),
            **(model._gen_predict_params_dict()),
        )
        check_sub_dict_match(
            clf_data.cls_params, all_params, excluding_keys=_non_booster_params
        )
        pred_result = model.transform(clf_data.cls_df_test).collect()
        for row in pred_result:
            np.testing.assert_equal(row.prediction, row.expected_prediction_with_params)
            np.testing.assert_allclose(
                row.probability, row.expected_probability_with_params, rtol=1e-3
            )

    def test_classifier_model_pipeline_save_load(self, clf_data: ClfData) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = "file:" + tmpdir
            classifier = SparkXGBClassifier()
            pipeline = Pipeline(stages=[classifier])
            pipeline = pipeline.copy(
                extra=get_params_map(clf_data.cls_params, classifier)
            )
            model = pipeline.fit(clf_data.cls_df_train)
            model.save(path)

            loaded_model = PipelineModel.load(path)
            for k, v in clf_data.cls_params.items():
                assert loaded_model.stages[0].getOrDefault(k) == v

            pred_result = loaded_model.transform(clf_data.cls_df_test).collect()
            for row in pred_result:
                np.testing.assert_allclose(
                    row.probability, row.expected_probability_with_params, atol=1e-3
                )
            assert_model_compatible(model.stages[0], tmpdir)

    def test_classifier_with_cross_validator(self, clf_data: ClfData) -> None:
        xgb_classifier = SparkXGBClassifier(n_estimators=1)
        paramMaps = ParamGridBuilder().addGrid(xgb_classifier.max_depth, [1, 2]).build()
        cvBin = CrossValidator(
            estimator=xgb_classifier,
            estimatorParamMaps=paramMaps,
            evaluator=BinaryClassificationEvaluator(),
            seed=1,
            parallelism=4,
            numFolds=2,
        )
        cvBinModel = cvBin.fit(clf_data.cls_df_train_large)
        cvBinModel.transform(clf_data.cls_df_test)

    def test_convert_to_sklearn_model_clf(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(
            n_estimators=FAST_N_ESTIMATORS,
            missing=2.0,
            max_depth=3,
            sketch_eps=0.5,
        )
        clf_model = classifier.fit(clf_data.cls_df_train)

        # Check that regardless of what booster, _convert_to_model converts to the
        # correct class type
        sklearn_classifier = classifier._convert_to_sklearn_model(
            clf_model.get_booster().save_raw("json"),
            clf_model.get_booster().save_config(),
        )
        assert isinstance(sklearn_classifier, XGBClassifier)
        assert sklearn_classifier.n_estimators == FAST_N_ESTIMATORS
        assert sklearn_classifier.missing == 2.0
        assert sklearn_classifier.max_depth == 3
        assert sklearn_classifier.get_params()["sketch_eps"] == 0.5

    def test_classifier_array_col_as_feature(self, clf_data: ClfData) -> None:
        train_dataset = clf_data.cls_df_train.withColumn(
            "features", vector_to_array(spark_sql_func.col("features"))
        )
        test_dataset = clf_data.cls_df_test.withColumn(
            "features", vector_to_array(spark_sql_func.col("features"))
        )
        classifier = SparkXGBClassifier(n_estimators=FAST_N_ESTIMATORS)
        model = classifier.fit(train_dataset)

        pred_result = model.transform(test_dataset).collect()
        for row in pred_result:
            np.testing.assert_equal(row.prediction, row.expected_prediction)
            np.testing.assert_allclose(
                row.probability, row.expected_probability, rtol=1e-3
            )

    def test_classifier_with_feature_names_types_weights(
        self, clf_data: ClfData
    ) -> None:
        classifier = SparkXGBClassifier(
            feature_names=["a1", "a2", "a3"],
            feature_types=["i", "int", "float"],
            feature_weights=[2.0, 5.0, 3.0],
            n_estimators=FAST_N_ESTIMATORS,
        )
        model = classifier.fit(clf_data.cls_df_train)
        model.transform(clf_data.cls_df_test).collect()

    def test_early_stop_param_validation(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(early_stopping_rounds=1)
        with pytest.raises(ValueError, match="early_stopping_rounds"):
            classifier.fit(clf_data.cls_df_train)

    def test_classifier_with_list_eval_metric(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(
            eval_metric=["auc", "rmse"], n_estimators=FAST_N_ESTIMATORS
        )
        model = classifier.fit(clf_data.cls_df_train)
        model.transform(clf_data.cls_df_test).collect()

    def test_classifier_with_string_eval_metric(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(
            eval_metric="auc", n_estimators=FAST_N_ESTIMATORS
        )
        model = classifier.fit(clf_data.cls_df_train)
        model.transform(clf_data.cls_df_test).collect()

    def test_classifier_params_basic(self) -> None:
        py_clf = SparkXGBClassifier()
        assert hasattr(py_clf, "n_estimators")
        assert py_clf.n_estimators.parent == py_clf.uid
        assert not hasattr(py_clf, "gpu_id")
        assert hasattr(py_clf, "device")

        assert py_clf.getOrDefault(py_clf.n_estimators) == 100
        assert py_clf.getOrDefault(getattr(py_clf, "objective")) is None
        py_clf2 = SparkXGBClassifier(n_estimators=200)
        assert py_clf2.getOrDefault(getattr(py_clf2, "n_estimators")) == 200
        py_clf3 = py_clf2.copy({getattr(py_clf2, "max_depth"): 10})
        assert py_clf3.getOrDefault(getattr(py_clf3, "n_estimators")) == 200
        assert py_clf3.getOrDefault(getattr(py_clf3, "max_depth")), 10

    def test_classifier_kwargs_basic(self, clf_data: ClfData) -> None:
        py_clf = SparkXGBClassifier(**clf_data.cls_params)
        assert hasattr(py_clf, "n_estimators")
        assert py_clf.n_estimators.parent == py_clf.uid
        assert not hasattr(py_clf, "gpu_id")
        assert hasattr(py_clf, "device")
        assert hasattr(py_clf, "arbitrary_params_dict")
        assert py_clf.getOrDefault(py_clf.arbitrary_params_dict) == {}

        # Testing overwritten params
        py_clf = SparkXGBClassifier()
        py_clf.setParams(x=1, y=2)
        py_clf.setParams(y=3, z=4)
        xgb_params = py_clf._gen_xgb_params_dict()
        assert xgb_params["x"] == 1
        assert xgb_params["y"] == 3
        assert xgb_params["z"] == 4

    def test_device_param(self, clf_data: ClfData) -> None:
        clf = SparkXGBClassifier(device="cuda", tree_method="exact")
        with pytest.raises(ValueError, match="not supported for distributed"):
            clf.fit(clf_data.cls_df_train)

        clf = SparkXGBClassifier(device="cuda", tree_method="approx")
        clf._validate_params()
        clf = SparkXGBClassifier(device="cuda")
        clf._validate_params()

    def test_gpu_params(self) -> None:
        clf = SparkXGBClassifier()
        assert not clf._run_on_gpu()

        clf = SparkXGBClassifier(device="cuda", tree_method="hist")
        assert clf._run_on_gpu()

        clf = SparkXGBClassifier(device="cuda")
        assert clf._run_on_gpu()

        clf = SparkXGBClassifier(tree_method="hist")
        assert not clf._run_on_gpu()

        clf = SparkXGBClassifier(device="cuda", tree_method="approx")
        assert clf._run_on_gpu()

    def test_gpu_transform(self, clf_data: ClfData) -> None:
        """local mode"""
        classifier = SparkXGBClassifier(device="cpu", n_estimators=FAST_N_ESTIMATORS)
        model: SparkXGBClassifierModel = classifier.fit(clf_data.cls_df_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = "file:" + tmpdir
            model.write().overwrite().save(path)

            # The model trained with CPU, transform defaults to cpu
            assert not model._run_on_gpu()

            # without error
            model.transform(clf_data.cls_df_test).collect()

            model.set_device("cuda")
            assert model._run_on_gpu()

            model_loaded = SparkXGBClassifierModel.load(path)

            # The model trained with CPU, transform defaults to cpu
            assert not model_loaded._run_on_gpu()
            # without error
            model_loaded.transform(clf_data.cls_df_test).collect()

            model_loaded.set_device("cuda")
            assert model_loaded._run_on_gpu()

    def test_validate_gpu_params(self) -> None:
        # Standalone
        standalone_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "12")
            .set("spark.task.cpus", "1")
            .set("spark.executor.resource.gpu.amount", "1")
            .set("spark.task.resource.gpu.amount", "0.08")
        )
        classifier_on_cpu = SparkXGBClassifier(device="cpu")
        classifier_on_gpu = SparkXGBClassifier(device="cuda")

        # No exception for classifier on CPU
        classifier_on_cpu._validate_gpu_params("3.4.0", standalone_conf)

        with pytest.raises(
            ValueError, match="XGBoost doesn't support GPU fractional configurations"
        ):
            classifier_on_gpu._validate_gpu_params("3.3.0", standalone_conf)

        # No issues
        classifier_on_gpu._validate_gpu_params("3.4.0", standalone_conf)
        classifier_on_gpu._validate_gpu_params("3.4.1", standalone_conf)
        classifier_on_gpu._validate_gpu_params("3.5.0", standalone_conf)
        classifier_on_gpu._validate_gpu_params("3.5.1", standalone_conf)

        # no spark.executor.resource.gpu.amount
        standalone_bad_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "12")
            .set("spark.task.cpus", "1")
            .set("spark.task.resource.gpu.amount", "0.08")
        )
        msg_match = (
            "The `spark.executor.resource.gpu.amount` is required for training on GPU"
        )
        with pytest.raises(ValueError, match=msg_match):
            classifier_on_gpu._validate_gpu_params("3.3.0", standalone_bad_conf)
        with pytest.raises(ValueError, match=msg_match):
            classifier_on_gpu._validate_gpu_params("3.4.0", standalone_bad_conf)
        with pytest.raises(ValueError, match=msg_match):
            classifier_on_gpu._validate_gpu_params("3.4.1", standalone_bad_conf)
        with pytest.raises(ValueError, match=msg_match):
            classifier_on_gpu._validate_gpu_params("3.5.0", standalone_bad_conf)
        with pytest.raises(ValueError, match=msg_match):
            classifier_on_gpu._validate_gpu_params("3.5.1", standalone_bad_conf)

        standalone_bad_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "12")
            .set("spark.task.cpus", "1")
            .set("spark.executor.resource.gpu.amount", "1")
        )
        msg_match = (
            "The `spark.task.resource.gpu.amount` is required for training on GPU"
        )
        with pytest.raises(ValueError, match=msg_match):
            classifier_on_gpu._validate_gpu_params("3.3.0", standalone_bad_conf)

        classifier_on_gpu._validate_gpu_params("3.4.0", standalone_bad_conf)
        classifier_on_gpu._validate_gpu_params("3.5.0", standalone_bad_conf)
        classifier_on_gpu._validate_gpu_params("3.5.1", standalone_bad_conf)

        # Yarn and K8s mode
        for mode in ["yarn", "k8s://"]:
            conf = (
                SparkConf()
                .setMaster(mode)
                .set("spark.executor.cores", "12")
                .set("spark.task.cpus", "1")
                .set("spark.executor.resource.gpu.amount", "1")
                .set("spark.task.resource.gpu.amount", "0.08")
            )
            with pytest.raises(
                ValueError,
                match="XGBoost doesn't support GPU fractional configurations",
            ):
                classifier_on_gpu._validate_gpu_params("3.3.0", conf)
            with pytest.raises(
                ValueError,
                match="XGBoost doesn't support GPU fractional configurations",
            ):
                classifier_on_gpu._validate_gpu_params("3.4.0", conf)
            with pytest.raises(
                ValueError,
                match="XGBoost doesn't support GPU fractional configurations",
            ):
                classifier_on_gpu._validate_gpu_params("3.4.1", conf)
            with pytest.raises(
                ValueError,
                match="XGBoost doesn't support GPU fractional configurations",
            ):
                classifier_on_gpu._validate_gpu_params("3.5.0", conf)

            classifier_on_gpu._validate_gpu_params("3.5.1", conf)

        for mode in ["yarn", "k8s://"]:
            bad_conf = (
                SparkConf()
                .setMaster(mode)
                .set("spark.executor.cores", "12")
                .set("spark.task.cpus", "1")
                .set("spark.executor.resource.gpu.amount", "1")
            )
            msg_match = (
                "The `spark.task.resource.gpu.amount` is required for training on GPU"
            )
            with pytest.raises(ValueError, match=msg_match):
                classifier_on_gpu._validate_gpu_params("3.3.0", bad_conf)
            with pytest.raises(ValueError, match=msg_match):
                classifier_on_gpu._validate_gpu_params("3.4.0", bad_conf)
            with pytest.raises(ValueError, match=msg_match):
                classifier_on_gpu._validate_gpu_params("3.5.0", bad_conf)

            classifier_on_gpu._validate_gpu_params("3.5.1", bad_conf)

    def test_skip_stage_level_scheduling(self) -> None:
        standalone_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "12")
            .set("spark.task.cpus", "1")
            .set("spark.executor.resource.gpu.amount", "1")
            .set("spark.task.resource.gpu.amount", "0.08")
        )

        classifier_on_cpu = SparkXGBClassifier(device="cpu")
        classifier_on_gpu = SparkXGBClassifier(device="cuda")

        # the correct configurations should not skip stage-level scheduling
        assert not classifier_on_gpu._skip_stage_level_scheduling(
            "3.4.0", standalone_conf
        )
        assert not classifier_on_gpu._skip_stage_level_scheduling(
            "3.4.1", standalone_conf
        )
        assert not classifier_on_gpu._skip_stage_level_scheduling(
            "3.5.0", standalone_conf
        )
        assert not classifier_on_gpu._skip_stage_level_scheduling(
            "3.5.1", standalone_conf
        )

        # spark version < 3.4.0
        assert classifier_on_gpu._skip_stage_level_scheduling("3.3.0", standalone_conf)
        # not run on GPU
        assert classifier_on_cpu._skip_stage_level_scheduling("3.4.0", standalone_conf)

        # spark.executor.cores is not set
        bad_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.task.cpus", "1")
            .set("spark.executor.resource.gpu.amount", "1")
            .set("spark.task.resource.gpu.amount", "0.08")
        )
        assert classifier_on_gpu._skip_stage_level_scheduling("3.4.0", bad_conf)

        # spark.executor.cores=1
        bad_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "1")
            .set("spark.task.cpus", "1")
            .set("spark.executor.resource.gpu.amount", "1")
            .set("spark.task.resource.gpu.amount", "0.08")
        )
        assert classifier_on_gpu._skip_stage_level_scheduling("3.4.0", bad_conf)

        # spark.executor.resource.gpu.amount is not set
        bad_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "12")
            .set("spark.task.cpus", "1")
            .set("spark.task.resource.gpu.amount", "0.08")
        )
        assert classifier_on_gpu._skip_stage_level_scheduling("3.4.0", bad_conf)

        # spark.executor.resource.gpu.amount>1
        bad_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "12")
            .set("spark.task.cpus", "1")
            .set("spark.executor.resource.gpu.amount", "2")
            .set("spark.task.resource.gpu.amount", "0.08")
        )
        assert classifier_on_gpu._skip_stage_level_scheduling("3.4.0", bad_conf)

        # spark.task.resource.gpu.amount is not set
        bad_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "12")
            .set("spark.task.cpus", "1")
            .set("spark.executor.resource.gpu.amount", "1")
        )
        assert not classifier_on_gpu._skip_stage_level_scheduling("3.4.0", bad_conf)

        # spark.task.resource.gpu.amount=1
        bad_conf = (
            SparkConf()
            .setMaster("spark://foo")
            .set("spark.executor.cores", "12")
            .set("spark.task.cpus", "1")
            .set("spark.executor.resource.gpu.amount", "1")
            .set("spark.task.resource.gpu.amount", "1")
        )
        assert classifier_on_gpu._skip_stage_level_scheduling("3.4.0", bad_conf)

        # For Yarn and K8S
        for mode in ["yarn", "k8s://"]:
            for gpu_amount in ["0.08", "0.2", "1.0"]:
                conf = (
                    SparkConf()
                    .setMaster(mode)
                    .set("spark.executor.cores", "12")
                    .set("spark.task.cpus", "1")
                    .set("spark.executor.resource.gpu.amount", "1")
                    .set("spark.task.resource.gpu.amount", gpu_amount)
                )
                assert classifier_on_gpu._skip_stage_level_scheduling("3.3.0", conf)
                assert classifier_on_gpu._skip_stage_level_scheduling("3.4.0", conf)
                assert classifier_on_gpu._skip_stage_level_scheduling("3.4.1", conf)
                assert classifier_on_gpu._skip_stage_level_scheduling("3.5.0", conf)

                # This will be fixed when spark 4.0.0 is released.
                if gpu_amount == "1.0":
                    assert classifier_on_gpu._skip_stage_level_scheduling("3.5.1", conf)
                else:
                    # Starting from 3.5.1+, stage-level scheduling is working for Yarn and K8s
                    assert not classifier_on_gpu._skip_stage_level_scheduling(
                        "3.5.1", conf
                    )

    def test_classifier_xgb_summary(self, clf_with_weight: ClfWithWeight) -> None:
        clf_df_train = clf_with_weight.cls_df_train_with_eval_weight.filter(
            spark_sql_func.col("isVal") == False
        )
        spark_xgb_model = SparkXGBClassifier(
            eval_metric="logloss", n_estimators=FAST_N_ESTIMATORS
        ).fit(clf_df_train)

        np.testing.assert_allclose(
            clf_with_weight.cls_expected_evals_result_train,
            spark_xgb_model.training_summary.train_objective_history["logloss"],
            atol=1e-3,
        )

        assert spark_xgb_model.training_summary.validation_objective_history == {}

    def test_classifier_xgb_summary_with_validation(
        self, clf_with_weight: ClfWithWeight
    ) -> None:
        spark_xgb_model = SparkXGBClassifier(
            eval_metric="logloss",
            validation_indicator_col="isVal",
            n_estimators=FAST_N_ESTIMATORS,
        ).fit(
            clf_with_weight.cls_df_train_with_eval_weight,
        )

        np.testing.assert_allclose(
            clf_with_weight.cls_expected_evals_result_train,
            spark_xgb_model.training_summary.train_objective_history["logloss"],
            atol=1e-3,
        )

        np.testing.assert_allclose(
            clf_with_weight.cls_expected_evals_result_validation,
            spark_xgb_model.training_summary.validation_objective_history["logloss"],
            atol=1e-3,
        )


class XgboostLocalTest(SparkTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        logging.getLogger().setLevel("INFO")
        random.seed(2020)

        # >>> X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [0.0, 1.0, 5.5], [-1.0, -2.0, 1.0]])
        # >>> y = np.array([0, 0, 1, 2])
        # >>> cl = xgboost.XGBClassifier()
        # >>> cl.fit(X, y)
        # >>> cl.predict_proba(np.array([[1.0, 2.0, 3.0]]))
        # array([[0.5374299 , 0.23128504, 0.23128504]], dtype=float32)

        # Test classifier with both base margin and without
        # >>> import numpy as np
        # >>> import xgboost
        # >>> X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5], [4.0, 5.0, 6.0], [0.0, 6.0, 7.5]])
        # >>> w = np.array([1.0, 2.0, 1.0, 2.0])
        # >>> y = np.array([0, 1, 0, 1])
        # >>> base_margin = np.array([1,0,0,1])
        #
        # This is without the base margin
        # >>> cls1 = xgboost.XGBClassifier()
        # >>> cls1.fit(X, y, sample_weight=w)
        # >>> cls1.predict_proba(np.array([[1.0, 2.0, 3.0]]))
        # array([[0.3333333, 0.6666667]], dtype=float32)
        # >>> cls1.predict(np.array([[1.0, 2.0, 3.0]]))
        # array([1])
        #
        # This is with the same base margin for predict
        # >>> cls2 = xgboost.XGBClassifier()
        # >>> cls2.fit(X, y, sample_weight=w, base_margin=base_margin)
        # >>> cls2.predict_proba(np.array([[1.0, 2.0, 3.0]]), base_margin=[0])
        # array([[0.44142532, 0.5585747 ]], dtype=float32)
        # >>> cls2.predict(np.array([[1.0, 2.0, 3.0]]), base_margin=[0])
        # array([1])
        #
        # This is with a different base margin for predict
        # # >>> cls2 = xgboost.XGBClassifier()
        # >>> cls2.fit(X, y, sample_weight=w, base_margin=base_margin)
        # >>> cls2.predict_proba(np.array([[1.0, 2.0, 3.0]]), base_margin=[1])
        # array([[0.2252, 0.7747 ]], dtype=float32)
        # >>> cls2.predict(np.array([[1.0, 2.0, 3.0]]), base_margin=[0])
        # array([1])
        cls.cls_df_train_without_base_margin = cls.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, 1.0),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, 2.0),
                (Vectors.dense(4.0, 5.0, 6.0), 0, 1.0),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 2.0),
            ],
            ["features", "label", "weight"],
        )
        cls.cls_df_test_without_base_margin = cls.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), [0.3333, 0.6666], 1),
            ],
            [
                "features",
                "expected_prob_without_base_margin",
                "expected_prediction_without_base_margin",
            ],
        )

        cls.cls_df_train_with_same_base_margin = cls.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, 1.0, 1),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, 2.0, 0),
                (Vectors.dense(4.0, 5.0, 6.0), 0, 1.0, 0),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 2.0, 1),
            ],
            ["features", "label", "weight", "base_margin"],
        )
        cls.cls_df_test_with_same_base_margin = cls.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, [0.4415, 0.5585], 1),
            ],
            [
                "features",
                "base_margin",
                "expected_prob_with_base_margin",
                "expected_prediction_with_base_margin",
            ],
        )

        cls.cls_df_train_with_different_base_margin = cls.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, 1.0, 1),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, 2.0, 0),
                (Vectors.dense(4.0, 5.0, 6.0), 0, 1.0, 0),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 2.0, 1),
            ],
            ["features", "label", "weight", "base_margin"],
        )
        cls.cls_df_test_with_different_base_margin = cls.session.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 1, [0.2252, 0.7747], 1),
            ],
            [
                "features",
                "base_margin",
                "expected_prob_with_base_margin",
                "expected_prediction_with_base_margin",
            ],
        )
        cls.cls_df_sparse_train = cls.session.createDataFrame(
            [
                (Vectors.dense(1.0, 0.0, 3.0, 0.0, 0.0), 0),
                (Vectors.sparse(5, {1: 1.0, 3: 5.5}), 1),
                (Vectors.sparse(5, {4: -3.0}), 0),
            ]
            * 5,
            ["features", "label"],
        )

    def test_param_alias(self):
        py_cls = SparkXGBClassifier(features_col="f1", label_col="l1")
        self.assertEqual(py_cls.getOrDefault(py_cls.featuresCol), "f1")
        self.assertEqual(py_cls.getOrDefault(py_cls.labelCol), "l1")
        with pytest.raises(
            ValueError, match="Please use param name features_col instead"
        ):
            SparkXGBClassifier(featuresCol="f1")

    @staticmethod
    def test_param_value_converter():
        py_cls = SparkXGBClassifier(missing=np.float64(1.0), sketch_eps=np.float64(0.3))
        # don't check by isintance(v, float) because for numpy scalar it will also return True
        assert py_cls.getOrDefault(py_cls.missing).__class__.__name__ == "float"
        assert (
            py_cls.getOrDefault(py_cls.arbitrary_params_dict)[
                "sketch_eps"
            ].__class__.__name__
            == "float64"
        )

    def test_classifier_with_base_margin(self):
        cls_without_base_margin = SparkXGBClassifier(weight_col="weight")
        model_without_base_margin = cls_without_base_margin.fit(
            self.cls_df_train_without_base_margin
        )
        pred_result_without_base_margin = model_without_base_margin.transform(
            self.cls_df_test_without_base_margin
        ).collect()
        for row in pred_result_without_base_margin:
            self.assertTrue(
                np.isclose(
                    row.prediction,
                    row.expected_prediction_without_base_margin,
                    atol=1e-3,
                )
            )
            np.testing.assert_allclose(
                row.probability, row.expected_prob_without_base_margin, atol=1e-3
            )

        cls_with_same_base_margin = SparkXGBClassifier(
            weight_col="weight", base_margin_col="base_margin"
        )
        model_with_same_base_margin = cls_with_same_base_margin.fit(
            self.cls_df_train_with_same_base_margin
        )
        pred_result_with_same_base_margin = model_with_same_base_margin.transform(
            self.cls_df_test_with_same_base_margin
        ).collect()
        for row in pred_result_with_same_base_margin:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_base_margin, atol=1e-3
                )
            )
            np.testing.assert_allclose(
                row.probability, row.expected_prob_with_base_margin, atol=1e-3
            )

        cls_with_different_base_margin = SparkXGBClassifier(
            weight_col="weight", base_margin_col="base_margin"
        )
        model_with_different_base_margin = cls_with_different_base_margin.fit(
            self.cls_df_train_with_different_base_margin
        )
        pred_result_with_different_base_margin = (
            model_with_different_base_margin.transform(
                self.cls_df_test_with_different_base_margin
            ).collect()
        )
        for row in pred_result_with_different_base_margin:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_base_margin, atol=1e-3
                )
            )
            np.testing.assert_allclose(
                row.probability, row.expected_prob_with_base_margin, atol=1e-3
            )

    def test_num_workers_param(self):
        classifier = SparkXGBClassifier(num_workers=0)
        self.assertRaises(ValueError, classifier._validate_params)

    @pytest.mark.skipif(**no_sparse_unwrap())
    def test_classifier_with_sparse_optim(self):
        cls = SparkXGBClassifier(missing=0.0, n_estimators=FAST_N_ESTIMATORS)
        model = cls.fit(self.cls_df_sparse_train)
        assert model._xgb_sklearn_model.missing == 0.0
        pred_result = model.transform(self.cls_df_sparse_train).collect()

        # enable sparse optimiaztion
        cls2 = SparkXGBClassifier(
            missing=0.0,
            enable_sparse_data_optim=True,
            n_estimators=FAST_N_ESTIMATORS,
        )
        model2 = cls2.fit(self.cls_df_sparse_train)
        assert model2.getOrDefault(model2.enable_sparse_data_optim)
        assert model2._xgb_sklearn_model.missing == 0.0
        pred_result2 = model2.transform(self.cls_df_sparse_train).collect()

        for row1, row2 in zip(pred_result, pred_result2):
            self.assertTrue(np.allclose(row1.probability, row2.probability, rtol=1e-3))

    def test_unsupported_params(self):
        with pytest.raises(ValueError, match="evals_result"):
            SparkXGBClassifier(evals_result={})

    def test_collective_conf(self):
        classifier = SparkXGBClassifier(
            launch_tracker_on_driver=True,
            coll_cfg=Config(tracker_host_ip="192.168.1.32", tracker_port=59981),
        )
        with pytest.raises(Exception, match="Failed to bind socket"):
            classifier._get_tracker_args()

        classifier = SparkXGBClassifier(
            launch_tracker_on_driver=False,
            coll_cfg=Config(tracker_host_ip="127.0.0.1", tracker_port=58892),
        )
        with pytest.raises(
            ValueError, match="You must enable launch_tracker_on_driver"
        ):
            classifier._get_tracker_args()

        classifier = SparkXGBClassifier(
            launch_tracker_on_driver=True,
            coll_cfg=Config(tracker_host_ip="127.0.0.1", tracker_port=58893),
            num_workers=2,
        )
        launch_tracker_on_driver, rabit_envs = classifier._get_tracker_args()
        assert launch_tracker_on_driver is True
        assert rabit_envs["n_workers"] == 2
        assert rabit_envs["dmlc_tracker_uri"] == "127.0.0.1"
        assert rabit_envs["dmlc_tracker_port"] == 58893

        with tempfile.TemporaryDirectory() as tmpdir:
            path = "file:" + tmpdir
            port = get_avail_port()
            classifier = SparkXGBClassifier(
                launch_tracker_on_driver=True,
                coll_cfg=Config(tracker_host_ip="127.0.0.1", tracker_port=port),
                num_workers=1,
                n_estimators=1,
            )

            def check_conf(conf: Config) -> None:
                assert conf.tracker_host_ip == "127.0.0.1"
                assert conf.tracker_port == port

            check_conf(classifier.getOrDefault(classifier.coll_cfg))
            classifier.write().overwrite().save(path)

            loaded_classifier = SparkXGBClassifier.load(path)
            check_conf(loaded_classifier.getOrDefault(loaded_classifier.coll_cfg))

            model = classifier.fit(self.cls_df_sparse_train)
            check_conf(model.getOrDefault(model.coll_cfg))

            model.write().overwrite().save(path)
            loaded_model = SparkXGBClassifierModel.load(path)
            check_conf(loaded_model.getOrDefault(loaded_model.coll_cfg))

    def test_classifier_with_multi_cols(self):
        df = self.session.createDataFrame(
            [
                (1.0, 2.0, 0),
                (3.1, 4.2, 1),
            ],
            ["a", "b", "label"],
        )
        features = ["a", "b"]
        cls = SparkXGBClassifier(features_col=features, device="cpu", n_estimators=2)
        model = cls.fit(df)
        self.assertEqual(features, model.getOrDefault(model.features_cols))
        self.assertTrue(not model.isSet(model.featuresCol))

        # No exception
        model.transform(df).collect()


LTRData = namedtuple(
    "LTRData",
    (
        "df_train",
        "df_test",
        "df_train_1",
        "ranker_df_merged",
        "expected_evals_result_train",
        "expected_evals_result_validation",
    ),
)


@pytest.fixture(scope="module")
def ltr_data(spark: SparkSession) -> Generator[LTRData, None, None]:
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8")
    ranker_df_train = spark.createDataFrame(
        [
            (Vectors.dense(1.0, 2.0, 3.0), 0, 0),
            (Vectors.dense(4.0, 5.0, 6.0), 1, 0),
            (Vectors.dense(9.0, 4.0, 8.0), 2, 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 0, 1),
            (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 1),
            (Vectors.sparse(3, {1: 8.0, 2: 9.5}), 2, 1),
        ],
        ["features", "label", "qid"],
    )
    X_train = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [9.0, 4.0, 8.0],
            [np.nan, 1.0, 5.5],
            [np.nan, 6.0, 7.5],
            [np.nan, 8.0, 9.5],
        ]
    )
    qid_train = np.array([0, 0, 0, 1, 1, 1])
    y_train = np.array([0, 1, 2, 0, 1, 2])

    X_test = np.array(
        [
            [1.5, 2.0, 3.0],
            [4.5, 5.0, 6.0],
            [9.0, 4.5, 8.0],
            [np.nan, 1.0, 6.0],
            [np.nan, 6.0, 7.0],
            [np.nan, 8.0, 10.5],
        ]
    )
    qid_test = np.array([0, 0, 0, 1, 1, 1])
    y_test = np.array([1, 0, 2, 1, 1, 2])

    ltr = xgb.XGBRanker(
        tree_method="approx",
        objective="rank:pairwise",
        n_estimators=FAST_N_ESTIMATORS,
    )
    ltr.fit(X_train, y_train, qid=qid_train)
    predt = ltr.predict(X_test)

    ltr2 = xgb.XGBRanker(
        tree_method="approx",
        objective="rank:pairwise",
        n_estimators=FAST_N_ESTIMATORS,
    )
    ltr2.fit(
        X_train,
        y_train,
        qid=qid_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_qid=[qid_train, qid_test],
    )
    evals_result = ltr2.evals_result()
    expected_evals_result_train = evals_result["validation_0"]["ndcg@32"]
    expected_evals_result_validation = evals_result["validation_1"]["ndcg@32"]

    ranker_df_test = spark.createDataFrame(
        [
            (Vectors.dense(1.5, 2.0, 3.0), 0, float(predt[0]), 1),
            (Vectors.dense(4.5, 5.0, 6.0), 0, float(predt[1]), 0),
            (Vectors.dense(9.0, 4.5, 8.0), 0, float(predt[2]), 2),
            (Vectors.sparse(3, {1: 1.0, 2: 6.0}), 1, float(predt[3]), 1),
            (Vectors.sparse(3, {1: 6.0, 2: 7.0}), 1, float(predt[4]), 1),
            (Vectors.sparse(3, {1: 8.0, 2: 10.5}), 1, float(predt[5]), 2),
        ],
        ["features", "qid", "expected_prediction", "label"],
    )

    ranker_df_merged = (
        ranker_df_train.select(["features", "label", "qid"])
        .withColumn("isVal", spark_sql_func.lit(False))
        .union(
            ranker_df_test.select(["features", "label", "qid"]).withColumn(
                "isVal", spark_sql_func.lit(True)
            )
        )
    )

    ranker_df_train_1 = spark.createDataFrame(
        [
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 0, 9),
            (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 9),
            (Vectors.sparse(3, {1: 8.0, 2: 9.5}), 2, 9),
            (Vectors.dense(1.0, 2.0, 3.0), 0, 8),
            (Vectors.dense(4.0, 5.0, 6.0), 1, 8),
            (Vectors.dense(9.0, 4.0, 8.0), 2, 8),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 0, 7),
            (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 7),
            (Vectors.sparse(3, {1: 8.0, 2: 9.5}), 2, 7),
            (Vectors.dense(1.0, 2.0, 3.0), 0, 6),
            (Vectors.dense(4.0, 5.0, 6.0), 1, 6),
            (Vectors.dense(9.0, 4.0, 8.0), 2, 6),
        ]
        * 4,
        ["features", "label", "qid"],
    )
    yield LTRData(
        ranker_df_train,
        ranker_df_test,
        ranker_df_train_1,
        ranker_df_merged,
        expected_evals_result_train,
        expected_evals_result_validation,
    )


class TestPySparkLocalLETOR:
    def test_ranker(self, ltr_data: LTRData) -> None:
        ranker = SparkXGBRanker(
            qid_col="qid", objective="rank:pairwise", n_estimators=FAST_N_ESTIMATORS
        )
        assert ranker.getOrDefault(ranker.objective) == "rank:pairwise"
        model = ranker.fit(ltr_data.df_train)
        pred_result = model.transform(ltr_data.df_test).collect()
        for row in pred_result:
            assert np.isclose(row.prediction, row.expected_prediction, rtol=1e-3)

    def test_ranker_qid_sorted(self, ltr_data: LTRData) -> None:
        ranker = SparkXGBRanker(
            qid_col="qid",
            num_workers=4,
            objective="rank:ndcg",
            n_estimators=FAST_N_ESTIMATORS,
        )
        assert ranker.getOrDefault(ranker.objective) == "rank:ndcg"
        model = ranker.fit(ltr_data.df_train_1)
        model.transform(ltr_data.df_test).collect()

    def test_ranker_same_qid_in_same_partition(self, ltr_data: LTRData) -> None:
        ranker = SparkXGBRanker(qid_col="qid", num_workers=4, force_repartition=True)
        df, _ = ranker._prepare_input(ltr_data.df_train_1)

        def f(iterator: Iterable) -> List[int]:
            yield list(set(iterator))

        rows = df.select("qid").rdd.mapPartitions(f).collect()
        assert len(rows) == 4
        for row in rows:
            assert len(row) == 1
            assert row[0].qid in [6, 7, 8, 9]

    def test_ranker_xgb_summary(self, ltr_data: LTRData) -> None:
        spark_xgb_model = SparkXGBRanker(
            tree_method="approx",
            qid_col="qid",
            objective="rank:pairwise",
            n_estimators=FAST_N_ESTIMATORS,
        ).fit(ltr_data.df_train)

        np.testing.assert_allclose(
            ltr_data.expected_evals_result_train,
            spark_xgb_model.training_summary.train_objective_history["ndcg@32"],
            atol=1e-3,
        )

        assert spark_xgb_model.training_summary.validation_objective_history == {}

    def test_ranker_xgb_summary_with_validation(self, ltr_data: LTRData) -> None:
        spark_xgb_model = SparkXGBRanker(
            tree_method="approx",
            qid_col="qid",
            objective="rank:pairwise",
            validation_indicator_col="isVal",
            n_estimators=FAST_N_ESTIMATORS,
        ).fit(ltr_data.ranker_df_merged)

        np.testing.assert_allclose(
            ltr_data.expected_evals_result_train,
            spark_xgb_model.training_summary.train_objective_history["ndcg@32"],
            atol=1e-3,
        )

        np.testing.assert_allclose(
            ltr_data.expected_evals_result_validation,
            spark_xgb_model.training_summary.validation_objective_history["ndcg@32"],
            atol=1e-3,
        )
