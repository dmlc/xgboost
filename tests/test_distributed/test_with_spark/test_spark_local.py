import logging
import os
import tempfile
from collections import namedtuple
from typing import Generator, Iterable, List

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
from xgboost import XGBClassifier, XGBRegressor
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
from xgboost.testing.collective import get_avail_port

logging.getLogger("py4j").setLevel(logging.INFO)

pytestmark = [tm.timeout(60), pytest.mark.skipif(**tm.no_spark())]

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
ClfData = namedtuple(
    "ClfData",
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


class TestRegressor:
    @pytest.fixture(scope="class")
    def reg_data(self, spark: SparkSession) -> RegData:
        rng = np.random.default_rng(seed=42)
        X = rng.random((100, 10))
        # Make odd rows sparse with some random values to test both dense and sparse paths.
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
        return RegData(
            X_train, X_test, y_train, y_test, w, base_margin, is_val, X, y, df
        )

    def test_regressor(self, reg_data: RegData) -> None:
        train_rows = np.where(~reg_data.is_val)[0]
        validation_rows = np.where(reg_data.is_val)[0]

        reg_param = {
            "n_estimators": 10,
            "max_depth": 5,
            "objective": "reg:squarederror",
            "max_bin": 9,
            "eval_metric": "rmse",
            "early_stopping_rounds": 1,
        }
        reg = XGBRegressor(**reg_param).fit(
            reg_data.X_train,
            reg_data.y_train,
            sample_weight=reg_data.weights[train_rows],
            eval_set=[(reg_data.X_test, reg_data.y_test)],
            sample_weight_eval_set=[reg_data.weights[validation_rows]],
        )
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
        assert np.allclose(preds, reg.predict(reg_data.X), rtol=1e-3)
        assert np.allclose(pred_contribs.sum(axis=1), preds, rtol=1e-3)
        assert np.allclose(
            reg.evals_result()["validation_0"]["rmse"],
            spark_regressor.training_summary.validation_objective_history["rmse"],
            atol=1e-6,
        )
        assert np.allclose(
            reg.best_score, spark_regressor._xgb_sklearn_model.best_score, atol=1e-3
        )

    def test_training_continuation(self, reg_data: RegData) -> None:
        params = {
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

    def test_regressor_with_base_margin(self, reg_data: RegData) -> None:
        params = {
            "n_estimators": 5,
            "max_depth": 3,
            "objective": "reg:squarederror",
        }
        spark_model = SparkXGBRegressor(base_margin_col="base_margin", **params).fit(
            reg_data.df
        )
        preds = (
            spark_model.transform(
                reg_data.df.select("row_id", "features", "base_margin")
            )
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

    def test_regressor_save_load(self, reg_data: RegData) -> None:
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

    def test_regressor_params(self) -> None:
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
        with pytest.raises(ValueError, match="Number of workers"):
            SparkXGBRegressor(num_workers=0)._validate_params()

    def test_valid_type(self, spark: SparkSession) -> None:
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

    def test_callbacks(self, reg_data: RegData) -> None:
        train_df = reg_data.df.select("features", "label")

        def custom_lr(boosting_round: int) -> float:
            return 1.0 / (boosting_round + 1)

        cb = [LearningRateScheduler(custom_lr)]
        reg_params = {
            "n_estimators": 10,
            "max_depth": 3,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spark-xgb-reg-cb")
            regressor = SparkXGBRegressor(callbacks=cb, **reg_params)
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
    def test_empty_train_data(self, spark: SparkSession, tree_method: str) -> None:
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


class TestClassifier:
    @pytest.fixture(scope="class")
    def clf_data(self, spark: SparkSession) -> ClfData:
        rng = np.random.default_rng(seed=123)
        X = rng.random((200, 10))
        X[1::2, :] = 0.0
        X[1::2, 1] = rng.random(len(X[1::2, 1]))
        X[1::2, 2] = rng.random(len(X[1::2, 2]))
        y = rng.integers(0, 2, size=200)
        w = rng.random(200)
        base_margin = rng.random(200)
        is_val = rng.random(200) < 0.2
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
                    int(y[i]),
                    float(w[i]),
                    float(base_margin[i]),
                    bool(is_val[i]),
                )
            )
        df = spark.createDataFrame(
            rows, ["row_id", "features", "label", "weight", "base_margin", "is_val"]
        )
        return ClfData(
            X_train, X_test, y_train, y_test, w, base_margin, is_val, X, y, df
        )

    def test_classifier(self, clf_data: ClfData) -> None:
        train_df = clf_data.df
        X = clf_data.X
        y = clf_data.y
        weights = clf_data.weights
        train_rows = np.where(~clf_data.is_val)[0]
        validation_rows = np.where(clf_data.is_val)[0]

        cls_params = {
            "n_estimators": 10,
            "max_depth": 5,
            "eval_metric": "logloss",
        }
        ref = XGBClassifier(**cls_params).fit(
            X[train_rows],
            y[train_rows],
            sample_weight=weights[train_rows],
            eval_set=[
                (X[train_rows], y[train_rows]),
                (X[validation_rows], y[validation_rows]),
            ],
            sample_weight_eval_set=[weights[train_rows], weights[validation_rows]],
        )

        spark_cls = SparkXGBClassifier(
            weight_col="weight",
            validation_indicator_col="is_val",
            **cls_params,
        ).fit(train_df)

        pred_result = spark_cls.transform(train_df)
        preds = (
            pred_result.orderBy("row_id")
            .select("prediction")
            .toPandas()["prediction"]
            .to_numpy()
        )
        proba = np.array(
            pred_result.orderBy("row_id")
            .select("probability")
            .toPandas()["probability"]
            .tolist()
        )

        assert np.allclose(preds, ref.predict(X), rtol=1e-3)
        assert np.allclose(proba, ref.predict_proba(X), rtol=1e-3)
        assert np.allclose(
            ref.evals_result()["validation_0"]["logloss"],
            spark_cls.training_summary.train_objective_history["logloss"],
            atol=1e-6,
        )
        assert np.allclose(
            ref.evals_result()["validation_1"]["logloss"],
            spark_cls.training_summary.validation_objective_history["logloss"],
            atol=1e-6,
        )

    def test_classifier_model_save_load(self, clf_data: ClfData) -> None:
        train_df = clf_data.df.select("features", "label")
        test_df = clf_data.df.select("features")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spark-xgb-clf-model")
            clf = SparkXGBClassifier(n_estimators=5, max_depth=3)
            model = clf.fit(train_df)
            model.save(path)
            loaded_model = SparkXGBClassifierModel.load(path)
            assert model.uid == loaded_model.uid
            pred_before = (
                model.transform(test_df)
                .select("prediction")
                .toPandas()["prediction"]
                .to_numpy()
            )
            pred_after = (
                loaded_model.transform(test_df)
                .select("prediction")
                .toPandas()["prediction"]
                .to_numpy()
            )
            assert np.allclose(pred_before, pred_after, rtol=1e-6)

            with pytest.raises(AssertionError, match="Expected class name"):
                SparkXGBRegressorModel.load(path)

    def test_classifier_model_pipeline_save_load(self, clf_data: ClfData) -> None:
        train_df = clf_data.df.select("features", "label")
        test_df = clf_data.df.select("features")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spark-xgb-clf-pipeline")
            classifier = SparkXGBClassifier()
            pipeline = Pipeline(stages=[classifier])
            pipeline = pipeline.copy(
                extra={
                    getattr(classifier, k): v
                    for k, v in {"max_depth": 5, "n_estimators": 10}.items()
                }
            )
            model = pipeline.fit(train_df)
            model.save(path)

            loaded_model = PipelineModel.load(path)
            pred_before = (
                model.transform(test_df)
                .select("prediction")
                .toPandas()["prediction"]
                .to_numpy()
            )
            pred_after = (
                loaded_model.transform(test_df)
                .select("prediction")
                .toPandas()["prediction"]
                .to_numpy()
            )
            assert np.allclose(pred_before, pred_after, rtol=1e-6)

    def test_classifier_params(self) -> None:
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
        with pytest.raises(ValueError, match="custom 'objective'"):
            SparkXGBClassifier(objective="binary:logistic")._validate_params()
        assert hasattr(py_clf, "arbitrary_params_dict")
        assert py_clf.getOrDefault(py_clf.arbitrary_params_dict) == {}

        # Testing overwritten params via setParams
        py_clf_overwrite = SparkXGBClassifier()
        py_clf_overwrite.setParams(x=1, y=2)
        py_clf_overwrite.setParams(y=3, z=4)
        xgb_params = py_clf_overwrite._gen_xgb_params_dict()
        assert xgb_params["x"] == 1
        assert xgb_params["y"] == 3
        assert xgb_params["z"] == 4
        with pytest.raises(ValueError, match="evals_result"):
            SparkXGBClassifier(evals_result={})

    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_empty_validation_data(self, spark: SparkSession, tree_method: str) -> None:
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
            n_estimators=10,
        )
        model = classifier.fit(df_train)
        pred_result = model.transform(df_train).collect()
        for row in pred_result:
            assert row.prediction == row.label

    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_empty_partition(self, spark: SparkSession, tree_method: str) -> None:
        # raw_df.repartition(4) will result int severe data skew, actually,
        # there is no any data in reducer partition 1, reducer partition 2
        # see https://github.com/dmlc/xgboost/issues/8221
        raw_df = spark.range(0, 40, 1, 50).withColumn(
            "label",
            spark_sql_func.when(spark_sql_func.rand(1) > 0.5, 1).otherwise(0),
        )
        vector_assembler = (
            VectorAssembler().setInputCols(["id"]).setOutputCol("features")
        )
        data_trans = vector_assembler.setHandleInvalid("keep").transform(raw_df)
        classifier = SparkXGBClassifier(
            tree_method=tree_method,
            n_estimators=10,
        )
        model = classifier.fit(data_trans)
        pred_result = model.transform(data_trans).collect()
        for row in pred_result:
            assert row.prediction in [0.0, 1.0]

    def test_classifier_with_cross_validator(self, clf_data: ClfData) -> None:
        xgb_classifier = SparkXGBClassifier(n_estimators=1)
        param_maps = (
            ParamGridBuilder().addGrid(xgb_classifier.max_depth, [1, 2]).build()
        )
        cv_bin = CrossValidator(
            estimator=xgb_classifier,
            estimatorParamMaps=param_maps,
            evaluator=BinaryClassificationEvaluator(),
            seed=1,
            parallelism=4,
            numFolds=2,
        )
        cv_model = cv_bin.fit(clf_data.df.select("features", "label"))
        cv_model.transform(clf_data.df.select("features"))

    def test_convert_to_sklearn_model_clf(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(
            n_estimators=10,
            missing=2.0,
            max_depth=3,
            sketch_eps=0.5,
        )
        clf_model = classifier.fit(clf_data.df.select("features", "label"))

        # Check that regardless of what booster, _convert_to_model converts to the
        # correct class type
        sklearn_classifier = classifier._convert_to_sklearn_model(
            clf_model.get_booster().save_raw("json"),
            clf_model.get_booster().save_config(),
        )
        assert isinstance(sklearn_classifier, XGBClassifier)
        assert sklearn_classifier.n_estimators == 10
        assert sklearn_classifier.missing == 2.0
        assert sklearn_classifier.max_depth == 3

    def test_classifier_array_col_as_feature(self, clf_data: ClfData) -> None:
        train_dataset = clf_data.df.select("features", "label").withColumn(
            "features", vector_to_array(spark_sql_func.col("features"))
        )
        test_dataset = clf_data.df.select("features").withColumn(
            "features", vector_to_array(spark_sql_func.col("features"))
        )
        classifier = SparkXGBClassifier(n_estimators=10)
        model = classifier.fit(train_dataset)

        pred_result = model.transform(test_dataset).collect()
        for row in pred_result:
            assert row.probability is not None

    def test_classifier_with_feature_names_types(self, clf_data: ClfData) -> None:
        n_features = clf_data.X.shape[1]
        classifier = SparkXGBClassifier(
            feature_names=[f"f{i}" for i in range(n_features)],
            feature_types=["float"] * n_features,
            feature_weights=[float(i + 1) for i in range(n_features)],
            n_estimators=10,
        )
        model = classifier.fit(clf_data.df.select("features", "label"))
        model.transform(clf_data.df.select("features")).collect()

    def test_early_stop_param_validation(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(early_stopping_rounds=1)
        with pytest.raises(ValueError, match="early_stopping_rounds"):
            classifier.fit(clf_data.df.select("features", "label"))

    def test_classifier_with_list_eval_metric(self, clf_data: ClfData) -> None:
        classifier = SparkXGBClassifier(eval_metric=["auc", "rmse"], n_estimators=10)
        model = classifier.fit(clf_data.df.select("features", "label"))
        model.transform(clf_data.df.select("features")).collect()

    @pytest.mark.skipif(**no_sparse_unwrap())
    def test_classifier_with_sparse_optim(self, spark: SparkSession) -> None:
        sparse_train = spark.createDataFrame(
            [
                (Vectors.dense(1.0, 0.0, 3.0, 0.0, 0.0), 0),
                (Vectors.sparse(5, {1: 1.0, 3: 5.5}), 1),
                (Vectors.sparse(5, {4: -3.0}), 0),
            ]
            * 5,
            ["features", "label"],
        )
        cls = SparkXGBClassifier(missing=0.0, n_estimators=10)
        model = cls.fit(sparse_train)
        assert model._xgb_sklearn_model.missing == 0.0
        pred_result = model.transform(sparse_train).collect()

        # enable sparse optimization
        cls2 = SparkXGBClassifier(
            missing=0.0,
            enable_sparse_data_optim=True,
            n_estimators=10,
        )
        model2 = cls2.fit(sparse_train)
        assert model2.getOrDefault(model2.enable_sparse_data_optim)
        assert model2._xgb_sklearn_model.missing == 0.0
        pred_result2 = model2.transform(sparse_train).collect()

        for row1, row2 in zip(pred_result, pred_result2):
            assert np.allclose(row1.probability, row2.probability, rtol=1e-3)

    def test_param_alias(self) -> None:
        py_cls = SparkXGBClassifier(features_col="f1", label_col="l1")
        assert py_cls.getOrDefault(py_cls.featuresCol) == "f1"
        assert py_cls.getOrDefault(py_cls.labelCol) == "l1"
        with pytest.raises(
            ValueError, match="Please use param name features_col instead"
        ):
            SparkXGBClassifier(featuresCol="f1")

    def test_param_value_converter(self) -> None:
        py_cls = SparkXGBClassifier(missing=np.float64(1.0), sketch_eps=np.float64(0.3))
        # don't check by isintance(v, float) because for numpy scalar it will also return True
        assert py_cls.getOrDefault(py_cls.missing).__class__.__name__ == "float"
        assert (
            py_cls.getOrDefault(py_cls.arbitrary_params_dict)[
                "sketch_eps"
            ].__class__.__name__
            == "float64"
        )

    def test_device_and_gpu_params(self, clf_data: ClfData) -> None:
        clf = SparkXGBClassifier(device="cuda", tree_method="exact")
        with pytest.raises(ValueError, match="not supported for distributed"):
            clf.fit(clf_data.df.select("features", "label"))

        clf = SparkXGBClassifier(device="cuda", tree_method="approx")
        clf._validate_params()
        clf = SparkXGBClassifier(device="cuda")
        clf._validate_params()

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
        classifier = SparkXGBClassifier(device="cpu", n_estimators=10)
        model: SparkXGBClassifierModel = classifier.fit(
            clf_data.df.select("features", "label")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = "file:" + tmpdir
            model.write().overwrite().save(path)

            # The model trained with CPU, transform defaults to cpu
            assert not model._run_on_gpu()

            # without error
            model.transform(clf_data.df.select("features")).collect()

            model.set_device("cuda")
            assert model._run_on_gpu()

            model_loaded = SparkXGBClassifierModel.load(path)

            # The model trained with CPU, transform defaults to cpu
            assert not model_loaded._run_on_gpu()
            # without error
            model_loaded.transform(clf_data.df.select("features")).collect()

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

    def test_collective_conf(self, spark: SparkSession) -> None:
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

            sparse_train = spark.createDataFrame(
                [
                    (Vectors.dense(1.0, 0.0, 3.0, 0.0, 0.0), 0),
                    (Vectors.sparse(5, {1: 1.0, 3: 5.5}), 1),
                    (Vectors.sparse(5, {4: -3.0}), 0),
                ]
                * 5,
                ["features", "label"],
            )
            model = classifier.fit(sparse_train)
            check_conf(model.getOrDefault(model.coll_cfg))

            model.write().overwrite().save(path)
            loaded_model = SparkXGBClassifierModel.load(path)
            check_conf(loaded_model.getOrDefault(loaded_model.coll_cfg))


LTRData = namedtuple(
    "LTRData",
    (
        "ranker_df",
        "X_train",
        "y_train",
        "qid_train",
        "X_test",
        "y_test",
        "qid_test",
    ),
)


class TestPySparkLocalLETOR:
    @pytest.fixture(scope="class")
    def ltr_data(self, spark: SparkSession) -> LTRData:
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8")
        ranker_df = spark.createDataFrame(
            [
                (Vectors.dense(1.0, 2.0, 3.0), 0, 0, None, False),
                (Vectors.dense(4.0, 5.0, 6.0), 1, 0, None, False),
                (Vectors.dense(9.0, 4.0, 8.0), 2, 0, None, False),
                (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 0, 1, None, False),
                (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, 1, None, False),
                (Vectors.sparse(3, {1: 8.0, 2: 9.5}), 2, 1, None, False),
                (Vectors.dense(1.5, 2.0, 3.0), 1, 0, 0, True),
                (Vectors.dense(4.5, 5.0, 6.0), 0, 0, 1, True),
                (Vectors.dense(9.0, 4.5, 8.0), 2, 0, 2, True),
                (Vectors.sparse(3, {1: 1.0, 2: 6.0}), 1, 1, 3, True),
                (Vectors.sparse(3, {1: 6.0, 2: 7.0}), 1, 1, 4, True),
                (Vectors.sparse(3, {1: 8.0, 2: 10.5}), 2, 1, 5, True),
            ],
            ["features", "label", "qid", "row_id", "isVal"],
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

        return LTRData(
            ranker_df,
            X_train,
            y_train,
            qid_train,
            X_test,
            y_test,
            qid_test,
        )

    def test_ranker(self, ltr_data: LTRData) -> None:
        ref = xgb.XGBRanker(
            tree_method="approx",
            objective="rank:pairwise",
            n_estimators=10,
        )
        ref.fit(
            ltr_data.X_train,
            ltr_data.y_train,
            qid=ltr_data.qid_train,
            eval_set=[(ltr_data.X_test, ltr_data.y_test)],
            eval_qid=[ltr_data.qid_test],
        )
        expected = ref.predict(ltr_data.X_test)

        ranker = SparkXGBRanker(
            qid_col="qid",
            tree_method="approx",
            objective="rank:pairwise",
            validation_indicator_col="isVal",
            n_estimators=10,
        )
        assert ranker.getOrDefault(ranker.objective) == "rank:pairwise"
        model = ranker.fit(ltr_data.ranker_df)
        test_df = ltr_data.ranker_df.where(spark_sql_func.col("isVal"))
        pred_result = (
            model.transform(test_df)
            .orderBy("row_id")
            .select("prediction")
            .toPandas()["prediction"]
            .to_numpy()
        )
        assert np.allclose(pred_result, expected, rtol=1e-3)

    def test_ranker_same_qid_in_same_partition(self, spark: SparkSession) -> None:
        ranker_df_train = spark.createDataFrame(
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
        ranker = SparkXGBRanker(qid_col="qid", num_workers=4, force_repartition=True)
        df, _ = ranker._prepare_input(ranker_df_train)

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
            validation_indicator_col="isVal",
            n_estimators=10,
        ).fit(ltr_data.ranker_df)

        ref = xgb.XGBRanker(
            tree_method="approx",
            objective="rank:pairwise",
            n_estimators=10,
        )
        ref.fit(
            ltr_data.X_train,
            ltr_data.y_train,
            qid=ltr_data.qid_train,
            eval_set=[
                (ltr_data.X_train, ltr_data.y_train),
                (ltr_data.X_test, ltr_data.y_test),
            ],
            eval_qid=[ltr_data.qid_train, ltr_data.qid_test],
        )

        np.testing.assert_allclose(
            ref.evals_result()["validation_0"]["ndcg@32"],
            spark_xgb_model.training_summary.train_objective_history["ndcg@32"],
            atol=1e-3,
        )

        np.testing.assert_allclose(
            ref.evals_result()["validation_1"]["ndcg@32"],
            spark_xgb_model.training_summary.validation_objective_history["ndcg@32"],
            atol=1e-3,
        )
