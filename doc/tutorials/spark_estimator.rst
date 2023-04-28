################################
Distributed XGBoost with PySpark
################################

Starting from version 1.7.0, xgboost supports pyspark estimator APIs.

.. note::

   The feature is still experimental and not yet ready for production use.

.. contents::
  :backlinks: none
  :local:

*************************
XGBoost PySpark Estimator
*************************

SparkXGBRegressor
=================

SparkXGBRegressor is a PySpark ML estimator. It implements the XGBoost classification
algorithm based on XGBoost python library, and it can be used in PySpark Pipeline
and PySpark ML meta algorithms like CrossValidator/TrainValidationSplit/OneVsRest.

We can create a ``SparkXGBRegressor`` estimator like:

.. code-block:: python

  from xgboost.spark import SparkXGBRegressor
  spark_reg_estimator = SparkXGBRegressor(
    features_col="features",
    label_col="label",
    num_workers=2,
  )


The above snippet creates a spark estimator which can fit on a spark dataset,
and return a spark model that can transform a spark dataset and generate dataset
with prediction column. We can set almost all of xgboost sklearn estimator parameters
as ``SparkXGBRegressor`` parameters, but some parameter such as ``nthread`` is forbidden
in spark estimator, and some parameters are replaced with pyspark specific parameters
such as ``weight_col``, ``validation_indicator_col``, ``use_gpu``, for details please see
``SparkXGBRegressor`` doc.

The following code snippet shows how to train a spark xgboost regressor model,
first we need to prepare a training dataset as a spark dataframe contains
"label" column and "features" column(s), the "features" column(s) must be ``pyspark.ml.linalg.Vector``
type or spark array type or a list of feature column names.


.. code-block:: python

  xgb_regressor_model = xgb_regressor.fit(train_spark_dataframe)


The following code snippet shows how to predict test data using a spark xgboost regressor model,
first we need to prepare a test dataset as a spark dataframe contains
"features" and "label" column, the "features" column must be ``pyspark.ml.linalg.Vector``
type or spark array type.

.. code-block:: python

  transformed_test_spark_dataframe = xgb_regressor.predict(test_spark_dataframe)


The above snippet code returns a ``transformed_test_spark_dataframe`` that contains the input
dataset columns and an appended column "prediction" representing the prediction results.

SparkXGBClassifier
==================

``SparkXGBClassifier`` estimator has similar API with ``SparkXGBRegressor``, but it has some
pyspark classifier specific params, e.g. ``raw_prediction_col`` and ``probability_col`` parameters.
Correspondingly, by default, ``SparkXGBClassifierModel`` transforming test dataset will
generate result dataset with 3 new columns:

- "prediction": represents the predicted label.
- "raw_prediction": represents the output margin values.
- "probability": represents the prediction probability on each label.


***************************
XGBoost PySpark GPU support
***************************

XGBoost PySpark fully supports GPU acceleration. Users are not only able to enable
efficient training but also utilize their GPUs for the whole PySpark pipeline including
ETL and inference. In below sections, we will walk through an example of training on a
PySpark standalone GPU cluster. To get started, first we need to install some additional
packages, then we can set the ``use_gpu`` parameter to ``True``.

Prepare the necessary packages
==============================

Aside from the PySpark and XGBoost modules, we also need the `cuDF
<https://docs.rapids.ai/api/cudf/stable/>`_ package for handling Spark dataframe. We
recommend using either Conda or Virtualenv to manage python dependencies for PySpark
jobs. Please refer to `How to Manage Python Dependencies in PySpark
<https://www.databricks.com/blog/2020/12/22/how-to-manage-python-dependencies-in-pyspark.html>`_
for more details on PySpark dependency management.

In short, to create a Python environment that can be sent to a remote cluster using
virtualenv and pip:

.. code-block:: bash

  python -m venv xgboost_env
  source xgboost_env/bin/activate
  pip install pyarrow pandas venv-pack xgboost
  # https://docs.rapids.ai/install#pip-install
  pip install cudf-cu11 --extra-index-url=https://pypi.nvidia.com
  venv-pack -o xgboost_env.tar.gz

With Conda:

.. code-block:: bash

  conda create -y -n xgboost_env -c conda-forge conda-pack python=3.9
  conda activate xgboost_env
  # use conda when the supported version of xgboost (1.7) is released on conda-forge
  pip install xgboost
  conda install cudf pyarrow pandas -c rapids -c nvidia -c conda-forge
  conda pack -f -o xgboost_env.tar.gz


Write your PySpark application
==============================

Below snippet is a small example for training xgboost model with PySpark. Notice that we are
using a list of feature names and the additional parameter ``use_gpu``:

.. code-block:: python

  from xgboost.spark import SparkXGBRegressor
  spark = SparkSession.builder.getOrCreate()

  # read data into spark dataframe
  train_data_path = "xxxx/train"
  train_df = spark.read.parquet(data_path)

  test_data_path = "xxxx/test"
  test_df = spark.read.parquet(test_data_path)

  # assume the label column is named "class"
  label_name = "class"

  # get a list with feature column names
  feature_names = [x.name for x in train_df.schema if x.name != label]

  # create a xgboost pyspark regressor estimator and set use_gpu=True
  regressor = SparkXGBRegressor(
    features_col=feature_names,
    label_col=label_name,
    num_workers=2,
    use_gpu=True,
  )

  # train and return the model
  model = regressor.fit(train_df)

  # predict on test data
  predict_df = model.transform(test_df)
  predict_df.show()


Submit the PySpark application
==============================

Assuming you have configured your Spark cluster with GPU support. Otherwise, please
refer to `spark standalone configuration with GPU support <https://nvidia.github.io/spark-rapids/docs/get-started/getting-started-on-prem.html#spark-standalone-cluster>`_.

.. code-block:: bash

  export PYSPARK_DRIVER_PYTHON=python
  export PYSPARK_PYTHON=./environment/bin/python

  spark-submit \
    --master spark://<master-ip>:7077 \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.task.resource.gpu.amount=1 \
    --archives xgboost_env.tar.gz#environment \
    xgboost_app.py


The submit command sends the Python environment created by pip or conda along with the
specification of GPU allocation. We will revisit this command later on.

Model Persistence
=================

Similar to standard PySpark ml estimators, one can persist and reuse the model with ``save`
and ``load`` methods:

.. code-block:: python

  regressor = SparkXGBRegressor()
  model = regressor.fit(train_df)
  # save the model
  model.save("/tmp/xgboost-pyspark-model")
  # load the model
  model2 = SparkXGBRankerModel.load("/tmp/xgboost-pyspark-model")

To export the underlying booster model used by XGBoost:

.. code-block:: python

  regressor = SparkXGBRegressor()
  model = regressor.fit(train_df)
  # the same booster object returned by xgboost.train
  booster: xgb.Booster = model.get_booster()
  booster.predict(...)
  booster.save_model("model.json") # or model.ubj, depending on your choice of format.

This booster is not only shared by other Python interfaces but also used by all the
XGBoost bindings including the C, Java, and the R package. Lastly, one can extract the
booster file directly from a saved spark estimator without going through the getter:

.. code-block:: python

  import xgboost as xgb
  bst = xgb.Booster()
  # Loading the model saved in previous snippet
  bst.load_model("/tmp/xgboost-pyspark-model/model/part-00000")


Accelerate the whole pipeline for xgboost pyspark
=================================================

With `RAPIDS Accelerator for Apache Spark <https://nvidia.github.io/spark-rapids/>`_, you
can leverage GPUs to accelerate the whole pipeline (ETL, Train, Transform) for xgboost
pyspark without any Python code change. An example submit command is shown below with
additional spark configurations and dependencies:

.. code-block:: bash

  export PYSPARK_DRIVER_PYTHON=python
  export PYSPARK_PYTHON=./environment/bin/python

  spark-submit \
    --master spark://<master-ip>:7077 \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.task.resource.gpu.amount=1 \
    --packages com.nvidia:rapids-4-spark_2.12:23.04.0 \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --conf spark.sql.execution.arrow.maxRecordsPerBatch=1000000 \
    --archives xgboost_env.tar.gz#environment \
    xgboost_app.py

When rapids plugin is enabled, both of the JVM rapids plugin and the cuDF Python package
are required. More configuration options can be found in the RAPIDS link above along with
details on the plugin.
