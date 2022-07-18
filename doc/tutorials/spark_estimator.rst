###############################
Using XGBoost PySpark Estimator
###############################
Starting from version 2.0, xgboost supports pyspark estimator APIs.
The feature is still experimental and not yet ready for production use.

*****************
SparkXGBRegressor
*****************

SparkXGBRegressor is a PySpark ML estimator. It implements the XGBoost classification
algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
and PySpark ML meta algorithms like CrossValidator/TrainValidationSplit/OneVsRest.

We can create a `SparkXGBRegressor` estimator like:

.. code-block:: python

  from xgboost.spark import SparkXGBRegressor
  spark_reg_estimator = SparkXGBRegressor(num_workers=2, max_depth=5)


The above snippet create an spark estimator which can fit on a spark dataset,
and return a spark model that can transform a spark dataset and generate dataset
with prediction column. We can set almost all of xgboost sklearn estimator parameters
as `SparkXGBRegressor` parameters, but some parameter such as `nthread` is forbidden
in spark estimator, and some parameters are replaced with pyspark specific parameters
such as `weight_col`, `validation_indicator_col`, `use_gpu`, for details please see
`SparkXGBRegressor` doc.

The following code snippet shows how to train a spark xgboost regressor model,
first we need to prepare a training dataset as a spark dataframe contains
"features" and "label" column, the "features" column must be `pyspark.ml.linalg.Vector`
type or spark array type.

.. code-block:: python

  xgb_regressor_model = xgb_regressor.fit(train_spark_dataframe)


The following code snippet shows how to predict test data using a spark xgboost regressor model,
first we need to prepare a test dataset as a spark dataframe contains
"features" and "label" column, the "features" column must be `pyspark.ml.linalg.Vector`
type or spark array type.

.. code-block:: python

  transformed_test_spark_dataframe = xgb_regressor.predict(test_spark_dataframe)


The above snippet code returns a `transformed_test_spark_dataframe` that contains the input
dataset columns and an appended column "prediction" representing the prediction results.


******************
SparkXGBClassifier
******************


`SparkXGBClassifier` estimator has similar API with `SparkXGBRegressor`, but it has some
pyspark classifier specific params, e.g. `raw_prediction_col` and `probability_col` parameters.
Correspondingly, by default, `SparkXGBClassifierModel` transforming test dataset will
generate result dataset with 3 new columns:
 - "prediction": represents the predicted label.
 - "raw_prediction": represents the output margin values.
 - "probability": represents the prediction probability on each label.
