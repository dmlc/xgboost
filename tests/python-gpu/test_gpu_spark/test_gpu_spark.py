import logging
import sys

import pytest
import sklearn

sys.path.append("tests/python")
import testing as tm

if tm.no_dask()["condition"]:
    pytest.skip(msg=tm.no_spark()["reason"], allow_module_level=True)
if sys.platform.startswith("win"):
    pytest.skip("Skipping PySpark tests on Windows", allow_module_level=True)

from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor

gpu_discovery_script_path = "tests/python-gpu/test_gpu_spark/discover_gpu.sh"
executor_gpu_amount = 4
executor_cores = 4
num_workers = executor_gpu_amount


@pytest.fixture(scope="module", autouse=True)
def spark_session_with_gpu():
    spark_config = {
        "spark.master": f"local-cluster[1, {executor_gpu_amount}, 1024]",
        "spark.python.worker.reuse": "false",
        "spark.driver.host": "127.0.0.1",
        "spark.task.maxFailures": "1",
        "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
        "spark.sql.pyspark.jvmStacktrace.enabled": "true",
        "spark.cores.max": executor_cores,
        "spark.task.cpus": "1",
        "spark.executor.cores": executor_cores,
        "spark.worker.resource.gpu.amount": executor_gpu_amount,
        "spark.task.resource.gpu.amount": "1",
        "spark.executor.resource.gpu.amount": executor_gpu_amount,
        "spark.worker.resource.gpu.discoveryScript": gpu_discovery_script_path,
    }
    builder = SparkSession.builder.appName("xgboost spark python API Tests with GPU")
    for k, v in spark_config.items():
        builder.config(k, v)
    spark = builder.getOrCreate()
    logging.getLogger("pyspark").setLevel(logging.INFO)
    # We run a dummy job so that we block until the workers have connected to the master
    spark.sparkContext.parallelize(
        range(num_workers), num_workers
    ).barrier().mapPartitions(lambda _: []).collect()
    yield spark
    spark.stop()


@pytest.fixture
def spark_iris_dataset(spark_session_with_gpu):
    spark = spark_session_with_gpu
    data = sklearn.datasets.load_iris()
    train_rows = [
        (Vectors.dense(features), float(label))
        for features, label in zip(data.data[0::2], data.target[0::2])
    ]
    train_df = spark.createDataFrame(
        spark.sparkContext.parallelize(train_rows, num_workers), ["features", "label"]
    )
    test_rows = [
        (Vectors.dense(features), float(label))
        for features, label in zip(data.data[1::2], data.target[1::2])
    ]
    test_df = spark.createDataFrame(
        spark.sparkContext.parallelize(test_rows, num_workers), ["features", "label"]
    )
    return train_df, test_df


@pytest.fixture
def spark_iris_dataset_feature_cols(spark_session_with_gpu):
    spark = spark_session_with_gpu
    data = sklearn.datasets.load_iris()
    train_rows = [
        (*features.tolist(), float(label))
        for features, label in zip(data.data[0::2], data.target[0::2])
    ]
    train_df = spark.createDataFrame(
        spark.sparkContext.parallelize(train_rows, num_workers),
        [*data.feature_names, "label"],
    )
    test_rows = [
        (*features.tolist(), float(label))
        for features, label in zip(data.data[1::2], data.target[1::2])
    ]
    test_df = spark.createDataFrame(
        spark.sparkContext.parallelize(test_rows, num_workers),
        [*data.feature_names, "label"],
    )
    return train_df, test_df, data.feature_names


@pytest.fixture
def spark_diabetes_dataset(spark_session_with_gpu):
    spark = spark_session_with_gpu
    data = sklearn.datasets.load_diabetes()
    train_rows = [
        (Vectors.dense(features), float(label))
        for features, label in zip(data.data[0::2], data.target[0::2])
    ]
    train_df = spark.createDataFrame(
        spark.sparkContext.parallelize(train_rows, num_workers), ["features", "label"]
    )
    test_rows = [
        (Vectors.dense(features), float(label))
        for features, label in zip(data.data[1::2], data.target[1::2])
    ]
    test_df = spark.createDataFrame(
        spark.sparkContext.parallelize(test_rows, num_workers), ["features", "label"]
    )
    return train_df, test_df


@pytest.fixture
def spark_diabetes_dataset_feature_cols(spark_session_with_gpu):
    spark = spark_session_with_gpu
    data = sklearn.datasets.load_diabetes()
    train_rows = [
        (*features.tolist(), float(label))
        for features, label in zip(data.data[0::2], data.target[0::2])
    ]
    train_df = spark.createDataFrame(
        spark.sparkContext.parallelize(train_rows, num_workers),
        [*data.feature_names, "label"],
    )
    test_rows = [
        (*features.tolist(), float(label))
        for features, label in zip(data.data[1::2], data.target[1::2])
    ]
    test_df = spark.createDataFrame(
        spark.sparkContext.parallelize(test_rows, num_workers),
        [*data.feature_names, "label"],
    )
    return train_df, test_df, data.feature_names


def test_sparkxgb_classifier_with_gpu(spark_iris_dataset):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    classifier = SparkXGBClassifier(use_gpu=True, num_workers=num_workers)
    train_df, test_df = spark_iris_dataset
    model = classifier.fit(train_df)
    pred_result_df = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1 = evaluator.evaluate(pred_result_df)
    assert f1 >= 0.97


def test_sparkxgb_classifier_feature_cols_with_gpu(spark_iris_dataset_feature_cols):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    train_df, test_df, feature_names = spark_iris_dataset_feature_cols

    classifier = SparkXGBClassifier(
        features_col=feature_names, use_gpu=True, num_workers=num_workers
    )

    model = classifier.fit(train_df)
    pred_result_df = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1 = evaluator.evaluate(pred_result_df)
    assert f1 >= 0.97


def test_cv_sparkxgb_classifier_feature_cols_with_gpu(spark_iris_dataset_feature_cols):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    train_df, test_df, feature_names = spark_iris_dataset_feature_cols

    classifier = SparkXGBClassifier(
        features_col=feature_names, use_gpu=True, num_workers=num_workers
    )
    grid = ParamGridBuilder().addGrid(classifier.max_depth, [6, 8]).build()
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    cv = CrossValidator(
        estimator=classifier, evaluator=evaluator, estimatorParamMaps=grid, numFolds=3
    )
    cvModel = cv.fit(train_df)
    pred_result_df = cvModel.transform(test_df)
    f1 = evaluator.evaluate(pred_result_df)
    assert f1 >= 0.97


def test_sparkxgb_regressor_with_gpu(spark_diabetes_dataset):
    from pyspark.ml.evaluation import RegressionEvaluator

    regressor = SparkXGBRegressor(use_gpu=True, num_workers=num_workers)
    train_df, test_df = spark_diabetes_dataset
    model = regressor.fit(train_df)
    pred_result_df = model.transform(test_df)
    evaluator = RegressionEvaluator(metricName="rmse")
    rmse = evaluator.evaluate(pred_result_df)
    assert rmse <= 65.0


def test_sparkxgb_regressor_feature_cols_with_gpu(spark_diabetes_dataset_feature_cols):
    from pyspark.ml.evaluation import RegressionEvaluator

    train_df, test_df, feature_names = spark_diabetes_dataset_feature_cols
    regressor = SparkXGBRegressor(
        features_col=feature_names, use_gpu=True, num_workers=num_workers
    )

    model = regressor.fit(train_df)
    pred_result_df = model.transform(test_df)
    evaluator = RegressionEvaluator(metricName="rmse")
    rmse = evaluator.evaluate(pred_result_df)
    assert rmse <= 65.0
