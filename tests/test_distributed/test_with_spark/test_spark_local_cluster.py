import json
import logging
import os
import random
import tempfile
import uuid
from collections import namedtuple

import numpy as np
import pytest
import xgboost as xgb
from xgboost import testing as tm
from xgboost.callback import LearningRateScheduler

pytestmark = pytest.mark.skipif(**tm.no_spark())

FAST_N_ESTIMATORS = 20
DATA_MULTIPLIER = 20
LOWER_N_ESTIMATORS = 10

from typing import Generator

from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor
from xgboost.spark.utils import _get_max_num_concurrent_tasks

from .utils import SparkLocalClusterTestCase


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    os.environ["XGBOOST_PYSPARK_SHARED_SESSION"] = "1"
    config = {
        "spark.master": "local-cluster[2, 1, 1024]",
        "spark.python.worker.reuse": "true",
        "spark.driver.host": "127.0.0.1",
        "spark.task.maxFailures": "1",
        "spark.sql.shuffle.partitions": "4",
        "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
        "spark.sql.pyspark.jvmStacktrace.enabled": "true",
        "spark.cores.max": "2",
        "spark.task.cpus": "1",
        "spark.executor.cores": "1",
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
        os.environ.pop("XGBOOST_PYSPARK_SHARED_SESSION", None)


RegData = namedtuple("RegData", ("reg_df_train", "reg_df_test", "reg_params"))


@pytest.fixture(scope="module")
def reg_data(spark: SparkSession) -> Generator[RegData, None, None]:
    reg_params = {"max_depth": 5, "n_estimators": 10, "iteration_range": (0, 5)}

    X = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 5.5]])
    y = np.array([0, 1])

    def custom_lr(boosting_round):
        return 1.0 / (boosting_round + 1)

    reg1 = xgb.XGBRegressor(
        n_estimators=FAST_N_ESTIMATORS, callbacks=[LearningRateScheduler(custom_lr)]
    )
    reg1.fit(X, y)
    predt1 = reg1.predict(X)
    # array([0.02406833, 0.97593164], dtype=float32)

    reg2 = xgb.XGBRegressor(max_depth=5, n_estimators=10)
    reg2.fit(X, y)
    predt2 = reg2.predict(X, iteration_range=(0, 5))
    # array([0.22185263, 0.77814734], dtype=float32)

    reg_df_train = spark.createDataFrame(
        [
            (Vectors.dense(1.0, 2.0, 3.0), 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1),
        ],
        ["features", "label"],
    )
    reg_df_test = spark.createDataFrame(
        [
            (Vectors.dense(1.0, 2.0, 3.0), 0.0, float(predt2[0]), float(predt1[0])),
            (
                Vectors.sparse(3, {1: 1.0, 2: 5.5}),
                1.0,
                float(predt2[1]),
                float(predt1[1]),
            ),
        ],
        [
            "features",
            "expected_prediction",
            "expected_prediction_with_params",
            "expected_prediction_with_callbacks",
        ],
    )
    yield RegData(reg_df_train, reg_df_test, reg_params)


class TestPySparkLocalCluster:
    def test_regressor_basic_with_params(self, reg_data: RegData) -> None:
        regressor = SparkXGBRegressor(**reg_data.reg_params)
        model = regressor.fit(reg_data.reg_df_train)
        pred_result = model.transform(reg_data.reg_df_test).collect()
        for row in pred_result:
            assert np.isclose(
                row.prediction, row.expected_prediction_with_params, atol=1e-3
            )

    def test_callbacks(self, reg_data: RegData) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, str(uuid.uuid4()))

            def custom_lr(boosting_round):
                return 1.0 / (boosting_round + 1)

            cb = [LearningRateScheduler(custom_lr)]
            regressor = SparkXGBRegressor(callbacks=cb, n_estimators=FAST_N_ESTIMATORS)

            # Test the save/load of the estimator instead of the model, since
            # the callbacks param only exists in the estimator but not in the model
            regressor.save(path)
            regressor = SparkXGBRegressor.load(path)

            model = regressor.fit(reg_data.reg_df_train)
            pred_result = model.transform(reg_data.reg_df_test).collect()
            for row in pred_result:
                assert np.isclose(
                    row.prediction, row.expected_prediction_with_callbacks, atol=1e-3
                )


class XgboostLocalClusterTestCase(SparkLocalClusterTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        random.seed(2020)

        cls.n_workers = _get_max_num_concurrent_tasks(cls.session)

        # Distributed section
        # Binary classification
        base_cls_rows = [
            (Vectors.dense(1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0]), 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), np.array([0.0, 1.0, 5.5]), 1),
            (Vectors.dense(4.0, 5.0, 6.0), np.array([4.0, 5.0, 6.0]), 0),
            (Vectors.sparse(3, {1: 6.0, 2: 7.5}), np.array([0.0, 6.0, 7.5]), 1),
        ]
        cls_features = np.stack([row[1] for row in base_cls_rows], axis=0)
        cls_labels = np.array([row[2] for row in base_cls_rows])

        cls_train_features = np.tile(cls_features, (DATA_MULTIPLIER, 1))
        cls_train_labels = np.tile(cls_labels, DATA_MULTIPLIER)

        cls.cls_df_train_distributed = cls.session.createDataFrame(
            [(row[0], row[2]) for row in base_cls_rows] * DATA_MULTIPLIER,
            ["features", "label"],
        )

        clf = xgb.XGBClassifier(n_estimators=FAST_N_ESTIMATORS)
        clf.fit(cls_train_features, cls_train_labels)
        cls_prob = clf.predict_proba(cls_features)
        cls.cls_df_test_distributed = cls.session.createDataFrame(
            [
                (base_cls_rows[i][0], int(cls_labels[i]), cls_prob[i].tolist())
                for i in range(len(base_cls_rows))
            ],
            ["features", "expected_label", "expected_probability"],
        )

        # Binary classification with different num_estimators
        clf_low = xgb.XGBClassifier(n_estimators=LOWER_N_ESTIMATORS)
        clf_low.fit(cls_train_features, cls_train_labels)
        cls_prob_low = clf_low.predict_proba(cls_features)
        cls.cls_df_test_distributed_lower_estimators = cls.session.createDataFrame(
            [
                (base_cls_rows[i][0], int(cls_labels[i]), cls_prob_low[i].tolist())
                for i in range(len(base_cls_rows))
            ],
            ["features", "expected_label", "expected_probability"],
        )

        # Multiclass classification
        base_multi_rows = [
            (Vectors.dense(1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0]), 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), np.array([0.0, 1.0, 5.5]), 1),
            (Vectors.dense(4.0, 5.0, 6.0), np.array([4.0, 5.0, 6.0]), 0),
            (Vectors.sparse(3, {1: 6.0, 2: 7.5}), np.array([0.0, 6.0, 7.5]), 2),
        ]
        multi_features = np.stack([row[1] for row in base_multi_rows], axis=0)
        multi_labels = np.array([row[2] for row in base_multi_rows])

        multi_train_features = np.tile(multi_features, (DATA_MULTIPLIER, 1))
        multi_train_labels = np.tile(multi_labels, DATA_MULTIPLIER)

        cls.cls_df_train_distributed_multiclass = cls.session.createDataFrame(
            [(row[0], row[2]) for row in base_multi_rows] * DATA_MULTIPLIER,
            ["features", "label"],
        )

        multi_clf = xgb.XGBClassifier(n_estimators=FAST_N_ESTIMATORS, base_score=0.5)
        multi_clf.fit(multi_train_features, multi_train_labels)
        multi_margins = multi_clf.predict(multi_features, output_margin=True)
        cls.cls_df_test_distributed_multiclass = cls.session.createDataFrame(
            [
                (
                    base_multi_rows[i][0],
                    int(multi_labels[i]),
                    multi_margins[i].tolist(),
                )
                for i in range(len(base_multi_rows))
            ],
            ["features", "expected_label", "expected_margins"],
        )

        # Regression
        base_reg_rows = [
            (Vectors.dense(1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0]), 0),
            (Vectors.sparse(3, {1: 1.0, 2: 5.5}), np.array([0.0, 1.0, 5.5]), 1),
            (Vectors.dense(4.0, 5.0, 6.0), np.array([4.0, 5.0, 6.0]), 0),
            (Vectors.sparse(3, {1: 6.0, 2: 7.5}), np.array([0.0, 6.0, 7.5]), 2),
        ]
        reg_features = np.stack([row[1] for row in base_reg_rows], axis=0)
        reg_labels = np.array([row[2] for row in base_reg_rows], dtype=float)

        reg_train_features = np.tile(reg_features, (DATA_MULTIPLIER, 1))
        reg_train_labels = np.tile(reg_labels, DATA_MULTIPLIER)

        cls.reg_df_train_distributed = cls.session.createDataFrame(
            [(row[0], row[2]) for row in base_reg_rows] * DATA_MULTIPLIER,
            ["features", "label"],
        )

        reg = xgb.XGBRegressor(n_estimators=FAST_N_ESTIMATORS)
        reg.fit(reg_train_features, reg_train_labels)
        reg_pred = reg.predict(reg_features)
        cls.reg_df_test_distributed = cls.session.createDataFrame(
            [
                (base_reg_rows[i][0], float(reg_pred[i]))
                for i in range(len(base_reg_rows))
            ],
            ["features", "expected_label"],
        )

        # Adding weight and validation
        cls.clf_params_with_eval_dist = {
            "validation_indicator_col": "isVal",
            "early_stopping_rounds": 1,
            "eval_metric": "logloss",
            "n_estimators": FAST_N_ESTIMATORS,
        }
        cls.clf_params_with_weight_dist = {"weight_col": "weight"}
        base_eval_rows = [
            (Vectors.dense(1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0]), 0, False, 1.0),
            (
                Vectors.sparse(3, {1: 1.0, 2: 5.5}),
                np.array([0.0, 1.0, 5.5]),
                1,
                False,
                2.0,
            ),
            (Vectors.dense(4.0, 5.0, 6.0), np.array([4.0, 5.0, 6.0]), 0, True, 1.0),
            (
                Vectors.sparse(3, {1: 6.0, 2: 7.5}),
                np.array([0.0, 6.0, 7.5]),
                1,
                True,
                2.0,
            ),
        ]
        eval_features = np.stack([row[1] for row in base_eval_rows], axis=0)
        eval_labels = np.array([row[2] for row in base_eval_rows], dtype=float)
        eval_is_val = np.array([row[3] for row in base_eval_rows], dtype=bool)
        eval_weights = np.array([row[4] for row in base_eval_rows], dtype=float)

        eval_train_features = np.tile(eval_features, (DATA_MULTIPLIER, 1))
        eval_train_labels = np.tile(eval_labels, DATA_MULTIPLIER)
        eval_train_is_val = np.tile(eval_is_val, DATA_MULTIPLIER)
        eval_train_weights = np.tile(eval_weights, DATA_MULTIPLIER)

        train_mask = ~eval_train_is_val
        X_train = eval_train_features[train_mask]
        y_train = eval_train_labels[train_mask]
        w_train = eval_train_weights[train_mask]
        X_val = eval_train_features[eval_train_is_val]
        y_val = eval_train_labels[eval_train_is_val]
        w_val = eval_train_weights[eval_train_is_val]

        cls.cls_df_train_distributed_with_eval_weight = cls.session.createDataFrame(
            [(row[0], row[2], row[3], row[4]) for row in base_eval_rows]
            * DATA_MULTIPLIER,
            ["features", "label", "isVal", "weight"],
        )

        clf_weight = xgb.XGBClassifier(n_estimators=FAST_N_ESTIMATORS)
        clf_weight.fit(
            eval_train_features, eval_train_labels, sample_weight=eval_train_weights
        )
        prob_with_weight = clf_weight.predict_proba(eval_features[:1])[0].tolist()

        clf_eval = xgb.XGBClassifier(
            n_estimators=FAST_N_ESTIMATORS,
            early_stopping_rounds=1,
            eval_metric="logloss",
        )
        clf_eval.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        prob_with_eval = clf_eval.predict_proba(eval_features[:1])[0].tolist()

        clf_weight_eval = xgb.XGBClassifier(
            n_estimators=FAST_N_ESTIMATORS,
            early_stopping_rounds=1,
            eval_metric="logloss",
        )
        clf_weight_eval.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
        )
        prob_with_weight_and_eval = clf_weight_eval.predict_proba(eval_features[:1])[
            0
        ].tolist()

        cls.cls_df_test_distributed_with_eval_weight = cls.session.createDataFrame(
            [
                (
                    base_eval_rows[0][0],
                    prob_with_weight,
                    prob_with_eval,
                    prob_with_weight_and_eval,
                ),
            ],
            [
                "features",
                "expected_prob_with_weight",
                "expected_prob_with_eval",
                "expected_prob_with_weight_and_eval",
            ],
        )
        cls.clf_best_score_eval = clf_eval.best_score
        cls.clf_best_score_weight_and_eval = clf_weight_eval.best_score

        cls.reg_params_with_eval_dist = {
            "validation_indicator_col": "isVal",
            "early_stopping_rounds": 1,
            "eval_metric": "rmse",
            "n_estimators": FAST_N_ESTIMATORS,
        }
        cls.reg_params_with_weight_dist = {"weight_col": "weight"}
        cls.reg_df_train_distributed_with_eval_weight = cls.session.createDataFrame(
            [(row[0], row[2], row[3], row[4]) for row in base_eval_rows]
            * DATA_MULTIPLIER,
            ["features", "label", "isVal", "weight"],
        )

        reg_weight = xgb.XGBRegressor(n_estimators=FAST_N_ESTIMATORS)
        reg_weight.fit(
            eval_train_features, eval_train_labels, sample_weight=eval_train_weights
        )
        pred_with_weight = reg_weight.predict(eval_features[:2])

        reg_eval = xgb.XGBRegressor(
            n_estimators=FAST_N_ESTIMATORS, early_stopping_rounds=1, eval_metric="rmse"
        )
        reg_eval.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        pred_with_eval = reg_eval.predict(eval_features[:2])

        reg_weight_eval = xgb.XGBRegressor(
            n_estimators=FAST_N_ESTIMATORS, early_stopping_rounds=1, eval_metric="rmse"
        )
        reg_weight_eval.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
        )
        pred_with_weight_and_eval = reg_weight_eval.predict(eval_features[:2])

        cls.reg_df_test_distributed_with_eval_weight = cls.session.createDataFrame(
            [
                (
                    base_eval_rows[0][0],
                    float(pred_with_weight[0]),
                    float(pred_with_eval[0]),
                    float(pred_with_weight_and_eval[0]),
                ),
                (
                    base_eval_rows[1][0],
                    float(pred_with_weight[1]),
                    float(pred_with_eval[1]),
                    float(pred_with_weight_and_eval[1]),
                ),
            ],
            [
                "features",
                "expected_prediction_with_weight",
                "expected_prediction_with_eval",
                "expected_prediction_with_weight_and_eval",
            ],
        )
        cls.reg_best_score_eval = reg_eval.best_score
        cls.reg_best_score_weight_and_eval = reg_weight_eval.best_score

    def test_classifier_distributed_basic(self):
        classifier = SparkXGBClassifier(
            num_workers=self.n_workers, n_estimators=FAST_N_ESTIMATORS
        )
        model = classifier.fit(self.cls_df_train_distributed)
        pred_result = model.transform(self.cls_df_test_distributed).collect()
        for row in pred_result:
            self.assertTrue(np.isclose(row.expected_label, row.prediction, atol=1e-3))
            self.assertTrue(
                np.allclose(row.expected_probability, row.probability, atol=1e-3)
            )

    def test_classifier_distributed_multiclass(self):
        # There is no built-in multiclass option for external storage
        classifier = SparkXGBClassifier(
            num_workers=self.n_workers, n_estimators=FAST_N_ESTIMATORS, base_score=0.5
        )
        model = classifier.fit(self.cls_df_train_distributed_multiclass)
        pred_result = model.transform(self.cls_df_test_distributed_multiclass).collect()
        for row in pred_result:
            self.assertTrue(np.isclose(row.expected_label, row.prediction, atol=1e-3))
            self.assertTrue(
                np.allclose(row.expected_margins, row.rawPrediction, atol=1e-3)
            )

    def test_regressor_distributed_basic(self):
        regressor = SparkXGBRegressor(
            num_workers=self.n_workers, n_estimators=FAST_N_ESTIMATORS
        )
        model = regressor.fit(self.reg_df_train_distributed)
        pred_result = model.transform(self.reg_df_test_distributed).collect()
        for row in pred_result:
            self.assertTrue(np.isclose(row.expected_label, row.prediction, atol=1e-3))

    def test_classifier_distributed_weight_eval(self):
        # with weight
        classifier = SparkXGBClassifier(
            num_workers=self.n_workers,
            n_estimators=FAST_N_ESTIMATORS,
            **self.clf_params_with_weight_dist,
        )
        model = classifier.fit(self.cls_df_train_distributed_with_eval_weight)
        pred_result = model.transform(
            self.cls_df_test_distributed_with_eval_weight
        ).collect()
        for row in pred_result:
            self.assertTrue(
                np.allclose(row.probability, row.expected_prob_with_weight, atol=1e-3)
            )

        # with eval only
        classifier = SparkXGBClassifier(
            num_workers=self.n_workers, **self.clf_params_with_eval_dist
        )
        model = classifier.fit(self.cls_df_train_distributed_with_eval_weight)
        pred_result = model.transform(
            self.cls_df_test_distributed_with_eval_weight
        ).collect()
        for row in pred_result:
            self.assertTrue(
                np.allclose(row.probability, row.expected_prob_with_eval, atol=1e-3)
            )
        assert np.isclose(
            float(model.get_booster().attributes()["best_score"]),
            self.clf_best_score_eval,
            rtol=1e-3,
        )

        # with both weight and eval
        classifier = SparkXGBClassifier(
            num_workers=self.n_workers,
            **self.clf_params_with_eval_dist,
            **self.clf_params_with_weight_dist,
        )
        model = classifier.fit(self.cls_df_train_distributed_with_eval_weight)
        pred_result = model.transform(
            self.cls_df_test_distributed_with_eval_weight
        ).collect()
        for row in pred_result:
            self.assertTrue(
                np.allclose(
                    row.probability, row.expected_prob_with_weight_and_eval, atol=1e-3
                )
            )
        np.isclose(
            float(model.get_booster().attributes()["best_score"]),
            self.clf_best_score_weight_and_eval,
            rtol=1e-3,
        )

    def test_regressor_distributed_weight_eval(self):
        # with weight
        regressor = SparkXGBRegressor(
            num_workers=self.n_workers,
            n_estimators=FAST_N_ESTIMATORS,
            **self.reg_params_with_weight_dist,
        )
        model = regressor.fit(self.reg_df_train_distributed_with_eval_weight)
        pred_result = model.transform(
            self.reg_df_test_distributed_with_eval_weight
        ).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(
                    row.prediction, row.expected_prediction_with_weight, atol=1e-3
                )
            )
        # with eval only
        regressor = SparkXGBRegressor(
            num_workers=self.n_workers, **self.reg_params_with_eval_dist
        )
        model = regressor.fit(self.reg_df_train_distributed_with_eval_weight)
        pred_result = model.transform(
            self.reg_df_test_distributed_with_eval_weight
        ).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(row.prediction, row.expected_prediction_with_eval, atol=1e-3)
            )
        assert np.isclose(
            float(model.get_booster().attributes()["best_score"]),
            self.reg_best_score_eval,
            rtol=1e-3,
        )
        # with both weight and eval
        regressor = SparkXGBRegressor(
            num_workers=self.n_workers,
            use_external_storage=False,
            **self.reg_params_with_eval_dist,
            **self.reg_params_with_weight_dist,
        )
        model = regressor.fit(self.reg_df_train_distributed_with_eval_weight)
        pred_result = model.transform(
            self.reg_df_test_distributed_with_eval_weight
        ).collect()
        for row in pred_result:
            self.assertTrue(
                np.isclose(
                    row.prediction,
                    row.expected_prediction_with_weight_and_eval,
                    atol=1e-3,
                )
            )
        assert np.isclose(
            float(model.get_booster().attributes()["best_score"]),
            self.reg_best_score_weight_and_eval,
            rtol=1e-3,
        )

    def test_num_estimators(self):
        classifier = SparkXGBClassifier(
            num_workers=self.n_workers, n_estimators=LOWER_N_ESTIMATORS
        )
        model = classifier.fit(self.cls_df_train_distributed)
        pred_result = model.transform(
            self.cls_df_test_distributed_lower_estimators
        ).collect()
        for row in pred_result:
            self.assertTrue(np.isclose(row.expected_label, row.prediction, atol=1e-3))
            self.assertTrue(
                np.allclose(row.expected_probability, row.probability, atol=1e-3)
            )

    def test_distributed_params(self):
        classifier = SparkXGBClassifier(
            num_workers=self.n_workers, max_depth=7, n_estimators=FAST_N_ESTIMATORS
        )
        model = classifier.fit(self.cls_df_train_distributed)
        self.assertTrue(hasattr(classifier, "max_depth"))
        self.assertEqual(classifier.getOrDefault(classifier.max_depth), 7)
        booster_config = json.loads(model.get_booster().save_config())
        max_depth = booster_config["learner"]["gradient_booster"]["tree_train_param"][
            "max_depth"
        ]
        assert int(max_depth) == 7

    def test_repartition(self):
        # The following test case has a few partitioned datasets that are either
        # well partitioned relative to the number of workers that the user wants
        # or poorly partitioned. We only want to repartition when the dataset
        # is poorly partitioned so _repartition_needed is true in those instances.

        classifier = SparkXGBClassifier(num_workers=self.n_workers)
        basic = self.cls_df_train_distributed
        self.assertTrue(not classifier._repartition_needed(basic))
        bad_repartitioned = basic.repartition(self.n_workers + 1)
        self.assertTrue(classifier._repartition_needed(bad_repartitioned))
        good_repartitioned = basic.repartition(self.n_workers)
        self.assertFalse(classifier._repartition_needed(good_repartitioned))

        # Now testing if force_repartition returns True regardless of whether the data is well partitioned
        classifier = SparkXGBClassifier(
            num_workers=self.n_workers, force_repartition=True
        )
        good_repartitioned = basic.repartition(self.n_workers)
        self.assertTrue(classifier._repartition_needed(good_repartitioned))
