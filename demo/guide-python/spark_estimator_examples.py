'''
Collection of examples for using xgboost.spark estimator interface
==================================================

@author: Weichen Xu
'''
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.ml.linalg import Vectors
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost.spark import SparkXGBClassifier, SparkXGBRegressor
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator


spark = SparkSession.builder.master("local[*]").getOrCreate()


def create_spark_df(X, y):
    return spark.createDataFrame(
        spark.sparkContext.parallelize([
            (Vectors.dense(features), float(label))
            for features, label in zip(X, y)
        ]),
        ["features", "label"]
    )


diabetes_X, diabetes_y = sklearn.datasets.load_diabetes(return_X_y=True)
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = \
    train_test_split(diabetes_X, diabetes_y, test_size=0.3, shuffle=True)

diabetes_train_spark_df = create_spark_df(diabetes_X_train, diabetes_y_train)
diabetes_test_spark_df = create_spark_df(diabetes_X_test, diabetes_y_test)

xgb_regressor = SparkXGBRegressor(max_depth=5)
xgb_regressor_model = xgb_regressor.fit(diabetes_train_spark_df)

transformed_diabetes_test_spark_df = xgb_regressor_model.transform(diabetes_test_spark_df)
regressor_evaluator = RegressionEvaluator(metricName="rmse")
print(f"regressor rmse={regressor_evaluator.evaluate(transformed_diabetes_test_spark_df)}")

diabetes_train_spark_df2 = diabetes_train_spark_df.withColumn(
    "validationIndicatorCol",
)

spark.stop()
