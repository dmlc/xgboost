import logging
from typing import Union

import pytest
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import lit

from xgboost import testing as tm
from xgboost.spark import (
    SparkXGBClassifier,
    SparkXGBClassifierModel,
    SparkXGBRanker,
    SparkXGBRankerModel,
    SparkXGBRegressor,
    SparkXGBRegressorModel,
)

from .test_spark_local import spark as spark_local

logging.getLogger("py4j").setLevel(logging.INFO)

pytestmark = [tm.timeout(60), pytest.mark.skipif(**tm.no_spark())]


@pytest.fixture
def clf_and_reg_df(spark_local: SparkSession) -> DataFrame:
    """
    Fixture to create a DataFrame with example data.
    """
    data = [
        (Vectors.dense([1.0, 2.0, 3.0]), 1),
        (Vectors.dense([4.0, 5.0, 6.0]), 1),
        (Vectors.dense([9.0, 4.0, 8.0]), 0),
        (Vectors.dense([6.0, 2.0, 2.0]), 1),
        (Vectors.dense([5.0, 4.0, 3.0]), 0),
    ]
    columns = ["features", "label"]
    return spark_local.createDataFrame(data, schema=columns)


@pytest.fixture
def clf_and_reg_df_with_validation(clf_and_reg_df: DataFrame) -> DataFrame:
    """
    Fixture to create a DataFrame with example data.
    """
    # split data into training and validation sets
    train_df, validation_df = clf_and_reg_df.randomSplit([0.8, 0.2], seed=42)

    # Add a column to indicate validation rows
    train_df = train_df.withColumn("validation_indicator_col", lit(False))
    validation_df = validation_df.withColumn("validation_indicator_col", lit(True))
    return train_df.union(validation_df)


@pytest.fixture
def ranker_df(spark_local: SparkSession) -> DataFrame:
    """
    Fixture to create a DataFrame with sample data for ranking tasks.
    """
    data = [
        (Vectors.dense([1.0, 2.0, 3.0]), 0, 0),
        (Vectors.dense([4.0, 5.0, 6.0]), 1, 0),
        (Vectors.dense([9.0, 4.0, 8.0]), 0, 0),
        (Vectors.dense([6.0, 2.0, 2.0]), 1, 0),
        (Vectors.dense([5.0, 4.0, 3.0]), 0, 0),
    ]
    columns = ["features", "label", "qid"]
    return spark_local.createDataFrame(data, schema=columns)


@pytest.fixture
def ranker_df_with_validation(ranker_df: DataFrame) -> DataFrame:
    """
    Fixture to split the ranking DataFrame into training and validation sets,
    add validation indicator, and merge them back into a single DataFrame.
    """
    # Split the data into training and validation sets (80-20 split)
    train_df, validation_df = ranker_df.randomSplit([0.8, 0.2], seed=42)

    # Add a column to indicate whether the row is from the validation set
    train_df = train_df.withColumn("validation_indicator_col", lit(False))
    validation_df = validation_df.withColumn("validation_indicator_col", lit(True))

    # Union the training and validation DataFrames
    return train_df.union(validation_df)


class TestXGBoostTrainingSummary:
    @staticmethod
    def assert_empty_validation_objective_history(
        xgb_model: Union[
            SparkXGBClassifierModel, SparkXGBRankerModel, SparkXGBRegressorModel
        ]
    ) -> None:
        assert hasattr(xgb_model.training_summary, "validation_objective_history")
        assert isinstance(xgb_model.training_summary.validation_objective_history, dict)
        assert not xgb_model.training_summary.validation_objective_history

    @staticmethod
    def assert_non_empty_training_objective_history(
        xgb_model: Union[
            SparkXGBClassifierModel, SparkXGBRankerModel, SparkXGBRegressorModel
        ],
        metric: str,
        n_estimators: int,
    ) -> None:
        assert hasattr(xgb_model.training_summary, "train_objective_history")
        assert isinstance(xgb_model.training_summary.train_objective_history, dict)

        assert metric in xgb_model.training_summary.train_objective_history
        assert (
            len(xgb_model.training_summary.train_objective_history[metric])
            == n_estimators
        )

        for (
            training_metric,
            loss_evolution,
        ) in xgb_model.training_summary.train_objective_history.items():
            assert isinstance(training_metric, str)
            assert len(loss_evolution) == n_estimators
            assert all(isinstance(value, float) for value in loss_evolution)

    @staticmethod
    def assert_non_empty_validation_objective_history(
        xgb_model: Union[
            SparkXGBClassifierModel, SparkXGBRankerModel, SparkXGBRegressorModel
        ],
        metric: str,
        n_estimators: int,
    ) -> None:
        assert hasattr(xgb_model.training_summary, "validation_objective_history")
        assert isinstance(xgb_model.training_summary.validation_objective_history, dict)

        assert metric in xgb_model.training_summary.validation_objective_history
        assert (
            len(xgb_model.training_summary.validation_objective_history[metric])
            == n_estimators
        )

        for (
            validation_metric,
            loss_evolution,
        ) in xgb_model.training_summary.validation_objective_history.items():
            assert isinstance(validation_metric, str)
            assert len(loss_evolution) == n_estimators
            assert all(isinstance(value, float) for value in loss_evolution)

    @pytest.mark.parametrize(
        "spark_xgb_estimator, metric",
        [
            (SparkXGBClassifier, "logloss"),
            (SparkXGBClassifier, "error"),
            (SparkXGBRegressor, "rmse"),
            (SparkXGBRegressor, "mae"),
        ],
    )
    def test_xgb_summary_classification_regression(
        self,
        clf_and_reg_df: DataFrame,
        spark_xgb_estimator: Union[SparkXGBClassifier, SparkXGBRegressor],
        metric: str,
    ) -> None:
        n_estimators = 10
        spark_xgb_model = spark_xgb_estimator(
            eval_metric=metric, n_estimators=n_estimators
        ).fit(clf_and_reg_df)
        self.assert_non_empty_training_objective_history(
            spark_xgb_model, metric, n_estimators
        )
        self.assert_empty_validation_objective_history(spark_xgb_model)

    @pytest.mark.parametrize(
        "spark_xgb_estimator, metric",
        [
            (SparkXGBClassifier, "logloss"),
            (SparkXGBClassifier, "error"),
            (SparkXGBRegressor, "rmse"),
            (SparkXGBRegressor, "mae"),
        ],
    )
    def test_xgb_summary_classification_regression_with_validation(
        self,
        clf_and_reg_df_with_validation: DataFrame,
        spark_xgb_estimator: Union[SparkXGBClassifier, SparkXGBRegressor],
        metric: str,
    ) -> None:
        n_estimators = 10
        spark_xgb_model = spark_xgb_estimator(
            eval_metric=metric,
            validation_indicator_col="validation_indicator_col",
            n_estimators=n_estimators,
        ).fit(clf_and_reg_df_with_validation)

        self.assert_non_empty_training_objective_history(
            spark_xgb_model, metric, n_estimators
        )
        self.assert_non_empty_validation_objective_history(
            spark_xgb_model, metric, n_estimators
        )

    @pytest.mark.parametrize("metric", ["ndcg", "map"])
    def test_xgb_summary_ranker(self, ranker_df: DataFrame, metric: str) -> None:
        n_estimators = 10
        xgb_ranker = SparkXGBRanker(
            qid_col="qid", eval_metric=metric, n_estimators=n_estimators
        )
        xgb_ranker_model = xgb_ranker.fit(ranker_df)

        self.assert_non_empty_training_objective_history(
            xgb_ranker_model, metric, n_estimators
        )
        self.assert_empty_validation_objective_history(xgb_ranker_model)

    @pytest.mark.parametrize("metric", ["ndcg", "map"])
    def test_xgb_summary_ranker_with_validation(
        self, ranker_df_with_validation: DataFrame, metric: str
    ) -> None:
        n_estimators = 10
        xgb_ranker_model = SparkXGBRanker(
            qid_col="qid",
            validation_indicator_col="validation_indicator_col",
            eval_metric=metric,
            n_estimators=n_estimators,
        ).fit(ranker_df_with_validation)

        self.assert_non_empty_training_objective_history(
            xgb_ranker_model, metric, n_estimators
        )
        self.assert_non_empty_validation_objective_history(
            xgb_ranker_model, metric, n_estimators
        )
