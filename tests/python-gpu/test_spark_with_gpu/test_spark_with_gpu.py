import sys

import logging
import pytest
import sklearn

sys.path.append("tests/python")
import testing as tm

if tm.no_dask()["condition"]:
    pytest.skip(msg=tm.no_spark()["reason"], allow_module_level=True)
if sys.platform.startswith("win"):
    pytest.skip("Skipping PySpark tests on Windows", allow_module_level=True)


from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from xgboost.spark import SparkXGBRegressor, SparkXGBClassifier


@pytest.fixture(scope="module", autouse=True)
def spark_session_with_gpu():
    spark_config = {
        "spark.master": "local-cluster[1, 4, 1024]",
        "spark.python.worker.reuse": "false",
        "spark.driver.host": "127.0.0.1",
        "spark.task.maxFailures": "1",
        "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
        "spark.sql.pyspark.jvmStacktrace.enabled": "true",
        "spark.cores.max": "4",
        "spark.task.cpus": "1",
        "spark.executor.cores": "4",
        "spark.worker.resource.gpu.amount": "4",
        "spark.task.resource.gpu.amount": "1",
        "spark.executor.resource.gpu.amount": "4",
        "spark.worker.resource.gpu.discoveryScript": "tests/python-gpu/test_spark_with_gpu/discover_gpu.sh",
    }
    builder = SparkSession.builder.appName("xgboost spark python API Tests with GPU")
    for k, v in spark_config.items():
        builder.config(k, v)
    spark = builder.getOrCreate()
    logging.getLogger("pyspark").setLevel(logging.INFO)
    # We run a dummy job so that we block until the workers have connected to the master
    spark.sparkContext.parallelize(range(4), 4).barrier().mapPartitions(
        lambda _: []
    ).collect()
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
        spark.sparkContext.parallelize(train_rows, 4), ["features", "label"]
    )
    test_rows = [
        (Vectors.dense(features), float(label))
        for features, label in zip(data.data[1::2], data.target[1::2])
    ]
    test_df = spark.createDataFrame(
        spark.sparkContext.parallelize(test_rows, 4), ["features", "label"]
    )
    return train_df, test_df


@pytest.fixture
def spark_diabetes_dataset(spark_session_with_gpu):
    spark = spark_session_with_gpu
    data = sklearn.datasets.load_diabetes()
    train_rows = [
        (Vectors.dense(features), float(label))
        for features, label in zip(data.data[0::2], data.target[0::2])
    ]
    train_df = spark.createDataFrame(
        spark.sparkContext.parallelize(train_rows, 4), ["features", "label"]
    )
    test_rows = [
        (Vectors.dense(features), float(label))
        for features, label in zip(data.data[1::2], data.target[1::2])
    ]
    test_df = spark.createDataFrame(
        spark.sparkContext.parallelize(test_rows, 4), ["features", "label"]
    )
    return train_df, test_df


def test_sparkxgb_classifier_with_gpu(spark_iris_dataset):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    classifier = SparkXGBClassifier(
        use_gpu=True,
        num_workers=4,
    )
    train_df, test_df = spark_iris_dataset
    model = classifier.fit(train_df)
    pred_result_df = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1 = evaluator.evaluate(pred_result_df)
    assert f1 >= 0.97


def test_sparkxgb_regressor_with_gpu(spark_diabetes_dataset):
    from pyspark.ml.evaluation import RegressionEvaluator

    regressor = SparkXGBRegressor(
        use_gpu=True,
        num_workers=4,
    )
    train_df, test_df = spark_diabetes_dataset
    model = regressor.fit(train_df)
    pred_result_df = model.transform(test_df)
    evaluator = RegressionEvaluator(metricName="rmse")
    rmse = evaluator.evaluate(pred_result_df)
    assert rmse <= 65.0
