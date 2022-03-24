"""
Example of training with PySpark on CPU
=======================================

.. versionadded:: 1.6.0

"""
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from xgboost.spark import XGBoostClassifier
import xgboost

version = "1.5.2"

spark = (
    SparkSession.builder.master("local[1]")
    .config(
        "spark.jars.packages",
        f"ml.dmlc:xgboost4j_2.12:{version},ml.dmlc:xgboost4j-spark_2.12:{version}",
    )
    .appName("xgboost-pyspark iris")
    .getOrCreate()
)

schema = StructType(
    [
        StructField("sepal length", DoubleType(), nullable=True),
        StructField("sepal width", DoubleType(), nullable=True),
        StructField("petal length", DoubleType(), nullable=True),
        StructField("petal width", DoubleType(), nullable=True),
        StructField("class", StringType(), nullable=True),
    ]
)
raw_input = spark.read.schema(schema).csv("iris.data")

stringIndexer = StringIndexer(inputCol="class", outputCol="classIndex").fit(raw_input)
labeled_input = stringIndexer.transform(raw_input).drop("class")

vector_assembler = (
    VectorAssembler()
    .setInputCols(("sepal length", "sepal width", "petal length", "petal width"))
    .setOutputCol("features")
)
xgb_input = vector_assembler.transform(labeled_input).select("features", "classIndex")


params = {
    "objective": "multi:softprob",
    "treeMethod": "hist",
    "numWorkers": 1,
    "numRound": 100,
    "numClass": 3,
    "labelCol": "classIndex",
    "featuresCol": "features",
}

classifier = XGBoostClassifier(**params)
classifier.write().overwrite().save("/tmp/xgboost_classifier")
classifier1 = XGBoostClassifier.load("/tmp/xgboost_classifier")


classifier = (
    XGBoostClassifier()
    .setLabelCol("classIndex")
    .setFeaturesCol("features")
    .setTreeMethod("hist")
    .setNumClass(3)
    .setNumRound(100)
    .setObjective("multi:softprob")
)
classifier.setNumWorkers(1)

model = classifier.fit(xgb_input)


model = classifier.fit(xgb_input)
results = model.transform(xgb_input)
results.show()
