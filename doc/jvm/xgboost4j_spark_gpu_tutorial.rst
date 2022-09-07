#############################################
XGBoost4J-Spark-GPU Tutorial (version 1.6.1+)
#############################################

**XGBoost4J-Spark-GPU** is an open source library aiming to accelerate distributed XGBoost training on Apache Spark cluster from
end to end with GPUs by leveraging the `RAPIDS Accelerator for Apache Spark <https://nvidia.github.io/spark-rapids/>`_ product.

This tutorial will show you how to use **XGBoost4J-Spark-GPU**.

.. contents::
  :backlinks: none
  :local:

************************************************
Build an ML Application with XGBoost4J-Spark-GPU
************************************************

Add XGBoost to Your Project
===========================

Before we go into the tour of how to use XGBoost4J-Spark-GPU, you should first consult
:ref:`Installation from Maven repository <install_jvm_packages>` in order to add XGBoost4J-Spark-GPU as
a dependency for your project. We provide both stable releases and snapshots.

Data Preparation
================

In this section, we use the `Iris <https://archive.ics.uci.edu/ml/datasets/iris>`_ dataset as an example to
showcase how we use Apache Spark to transform a raw dataset and make it fit the data interface of XGBoost.

The Iris dataset is shipped in CSV format. Each instance contains 4 features, "sepal length", "sepal width",
"petal length" and "petal width". In addition, it contains the "class" column, which is essentially the
label with three possible values: "Iris Setosa", "Iris Versicolour" and "Iris Virginica".

Read Dataset with Spark's Built-In Reader
-----------------------------------------

.. code-block:: scala

  import org.apache.spark.sql.SparkSession
  import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

  val spark = SparkSession.builder().getOrCreate()

  val labelName = "class"
  val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField(labelName, StringType, true)))

  val xgbInput = spark.read.option("header", "false")
      .schema(schema)
      .csv(dataPath)

In the first line, we create an instance of a `SparkSession <https://spark.apache.org/docs/latest/sql-getting-started.html#starting-point-sparksession>`_
which is the entry point of any Spark application working with DataFrames. The ``schema`` variable
defines the schema of the DataFrame wrapping Iris data. With this explicitly set schema, we
can define the column names as well as their types; otherwise the column names would be
the default ones derived by Spark, such as ``_col0``, etc. Finally, we can use Spark's
built-in CSV reader to load the Iris CSV file as a DataFrame named ``xgbInput``.

Apache Spark also contains many built-in readers for other formats such as ORC, Parquet, Avro, JSON.


Transform Raw Iris Dataset
--------------------------

To make the Iris dataset recognizable to XGBoost, we need to encode the String-typed
label, i.e. "class", to the Double-typed label.

One way to convert the String-typed label to Double is to use Spark's built-in feature transformer
`StringIndexer <https://spark.apache.org/docs/2.3.1/api/scala/index.html#org.apache.spark.ml.feature.StringIndexer>`_.
But this feature is not accelerated in RAPIDS Accelerator, which means it will fall back
to CPU. Instead, we use an alternative way to achieve the same goal with the following code:

.. code-block:: scala

  import org.apache.spark.sql.expressions.Window
  import org.apache.spark.sql.functions._

  val spec = Window.orderBy(labelName)
  val Array(train, test) = xgbInput
      .withColumn("tmpClassName", dense_rank().over(spec) - 1)
      .drop(labelName)
      .withColumnRenamed("tmpClassName", labelName)
      .randomSplit(Array(0.7, 0.3), seed = 1)

  train.show(5)

.. code-block:: none

	+------------+-----------+------------+-----------+-----+
	|sepal length|sepal width|petal length|petal width|class|
	+------------+-----------+------------+-----------+-----+
	|         4.3|        3.0|         1.1|        0.1|    0|
	|         4.4|        2.9|         1.4|        0.2|    0|
	|         4.4|        3.0|         1.3|        0.2|    0|
	|         4.4|        3.2|         1.3|        0.2|    0|
	|         4.6|        3.2|         1.4|        0.2|    0|
	+------------+-----------+------------+-----------+-----+


With window operations, we have mapped the string column of labels to label indices.

Training
========

The GPU version of XGBoost-Spark supports both regression and classification
models. Although we use the Iris dataset in this tutorial to show how we use
``XGBoost/XGBoost4J-Spark-GPU`` to resolve a multi-classes classification problem, the
usage in Regression is very similar to classification.

To train a XGBoost model for classification, we need to claim a XGBoostClassifier first:

.. code-block:: scala

  import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
  val xgbParam = Map(
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "tree_method" -> "gpu_hist",
      "num_workers" -> 1)

  val featuresNames = schema.fieldNames.filter(name => name != labelName)

  val xgbClassifier = new XGBoostClassifier(xgbParam)
      .setFeaturesCol(featuresNames)
      .setLabelCol(labelName)

The available parameters for training a XGBoost model can be found in :doc:`here </parameter>`.
Similar to the XGBoost4J-Spark package, in addition to the default set of parameters,
XGBoost4J-Spark-GPU also supports the camel-case variant of these parameters to be
consistent with Spark's MLlib naming convention.

Specifically, each parameter in :doc:`this page </parameter>` has its equivalent form in
XGBoost4J-Spark-GPU with camel case. For example, to set ``max_depth`` for each tree, you can pass
parameter just like what we did in the above code snippet (as ``max_depth`` wrapped in a Map), or
you can do it through setters in XGBoostClassifer:

.. code-block:: scala

  val xgbClassifier = new XGBoostClassifier(xgbParam)
      .setFeaturesCol(featuresNames)
      .setLabelCol(labelName)
  xgbClassifier.setMaxDepth(2)

.. note::

  In contrast with XGBoost4j-Spark which accepts both a feature column with VectorUDT type and
  an array of feature column names, XGBoost4j-Spark-GPU only accepts an array of feature
  column names by ``setFeaturesCol(value: Array[String])``.

After setting XGBoostClassifier parameters and feature/label columns, we can build a
transformer, XGBoostClassificationModel by fitting XGBoostClassifier with the input
DataFrame. This ``fit`` operation is essentially the training process and the generated
model can then be used in other tasks like prediction.

.. code-block:: scala

  val xgbClassificationModel = xgbClassifier.fit(train)

Prediction
==========

When we get a model, either a XGBoostClassificationModel or a XGBoostRegressionModel, it takes a DataFrame as an input,
reads the column containing feature vectors, predicts for each feature vector, and outputs a new DataFrame
with the following columns by default:

* XGBoostClassificationModel will output margins (``rawPredictionCol``), probabilities(``probabilityCol``) and the eventual prediction labels (``predictionCol``) for each possible label.
* XGBoostRegressionModel will output prediction a label(``predictionCol``).

.. code-block:: scala

  val xgbClassificationModel = xgbClassifier.fit(train)
  val results = xgbClassificationModel.transform(test)
  results.show()

With the above code snippet, we get a DataFrame as result, which contains the margin, probability for each class,
and the prediction for each instance.

.. code-block:: none

	+------------+-----------+------------------+-------------------+-----+--------------------+--------------------+----------+
	|sepal length|sepal width|      petal length|        petal width|class|       rawPrediction|         probability|prediction|
	+------------+-----------+------------------+-------------------+-----+--------------------+--------------------+----------+
	|         4.5|        2.3|               1.3|0.30000000000000004|    0|[3.16666603088378...|[0.98853939771652...|       0.0|
	|         4.6|        3.1|               1.5|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         4.8|        3.1|               1.6|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         4.8|        3.4|               1.6|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         4.8|        3.4|1.9000000000000001|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         4.9|        2.4|               3.3|                1.0|    1|[-2.1498908996582...|[0.00596602633595...|       1.0|
	|         4.9|        2.5|               4.5|                1.7|    2|[-2.1498908996582...|[0.00596602633595...|       1.0|
	|         5.0|        3.5|               1.3|0.30000000000000004|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.1|        2.5|               3.0|                1.1|    1|[3.16666603088378...|[0.98853939771652...|       0.0|
	|         5.1|        3.3|               1.7|                0.5|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.1|        3.5|               1.4|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.1|        3.8|               1.6|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.2|        3.4|               1.4|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.2|        3.5|               1.5|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.2|        4.1|               1.5|                0.1|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.4|        3.9|               1.7|                0.4|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.5|        2.4|               3.8|                1.1|    1|[-2.1498908996582...|[0.00596602633595...|       1.0|
	|         5.5|        4.2|               1.4|                0.2|    0|[3.25857257843017...|[0.98969423770904...|       0.0|
	|         5.7|        2.5|               5.0|                2.0|    2|[-2.1498908996582...|[0.00280966912396...|       2.0|
	|         5.7|        3.0|               4.2|                1.2|    1|[-2.1498908996582...|[0.00643939292058...|       1.0|
	+------------+-----------+------------------+-------------------+-----+--------------------+--------------------+----------+

**********************
Submit the application
**********************

Here’s an example to submit an end-to-end XGBoost-4j-Spark-GPU Spark application to an
Apache Spark Standalone cluster, assuming the application main class is Iris and the
application jar is iris-1.0.0.jar

.. code-block:: bash

  cudf_version=22.02.0
  rapids_version=22.02.0
  xgboost_version=1.6.1
  main_class=Iris
  app_jar=iris-1.0.0.jar

  spark-submit \
    --master $master \
    --packages ai.rapids:cudf:${cudf_version},com.nvidia:rapids-4-spark_2.12:${rapids_version},ml.dmlc:xgboost4j-gpu_2.12:${xgboost_version},ml.dmlc:xgboost4j-spark-gpu_2.12:${xgboost_version} \
    --conf spark.executor.cores=12 \
    --conf spark.task.cpus=1 \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.task.resource.gpu.amount=0.08 \
    --conf spark.rapids.sql.csv.read.double.enabled=true \
    --conf spark.rapids.sql.hasNans=false \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --class ${main_class} \
     ${app_jar}

* First, we need to specify the ``RAPIDS Accelerator, cudf, xgboost4j-gpu, xgboost4j-spark-gpu`` packages by ``--packages``
* Second, ``RAPIDS Accelerator`` is a Spark plugin, so we need to configure it by specifying ``spark.plugins=com.nvidia.spark.SQLPlugin``

For details about other ``RAPIDS Accelerator`` other configurations, please refer to the `configuration <https://nvidia.github.io/spark-rapids/docs/configs.html>`_.

For ``RAPIDS Accelerator Frequently Asked Questions``, please refer to the
`frequently-asked-questions <https://nvidia.github.io/spark-rapids/docs/FAQ.html#frequently-asked-questions>`_.
