################################
Distributed XGBoost with PySpark
################################

.. versionadded:: 1.6.0

**XGBoost PySpark** is a project allowing XGBoost running on PySpark environment. Alougth the code
of **XGBoost PySpark** is shipping in the **XGBoost Python package**, it is the wrapper of XGBoost4j-Spark
and XGBoost4j-Spark-Gpu, which means all the data preparation and training or infering will be routed
to the logical of xgboost4j-spark or xgboost4j-spark-gpu.

.. contents::
  :backlinks: none
  :local:

********************************************
Build an ML Application with XGBoost PySpark
********************************************

Installation
===================================

Let's create a new Conda environment to manage all the dependencies there. You can use Python Virtual
Environment if you prefer or not have any enviroment.


.. code-block:: shell

  conda create -n xgboost python=3.8 -y
  conda activate xgboost
  pip install xgboost==1.6.0 pyspark==3.1.2

Data Preparation
================

In this section, we use `Iris <https://archive.ics.uci.edu/ml/datasets/iris>`_ dataset as an example to
showcase how we use Spark to transform raw dataset and make it fit to the data interface of XGBoost PySpark.

Iris dataset is shipped in CSV format. Each instance contains 4 features, "sepal length", "sepal width",
"petal length" and "petal width". In addition, it contains the "class" column, which is essentially the label
with three possible values: "Iris Setosa", "Iris Versicolour" and "Iris Virginica".


Start SparkSession
------------------

.. code-block:: python

  from pyspark.sql import SparkSession

  spark = SparkSession.builder\
      .master("local[1]")\
      .config("spark.jars.packages", "ml.dmlc:xgboost4j_2.12:1.6.0,ml.dmlc:xgboost4j-spark_2.12:1.6.0")\
      .appName("xgboost-pyspark iris").getOrCreate()

As aforementioned, XGBoost-PySpark is based on XGBoost4j-Spark or XGBoost4j-Spark-Gpu, we need to specify `spark.jars.packages`
with maven coordinates of XGBoost4j-Spark or XGBoost4j-Spark-Gpu jars.

If you would like to submit your xgboost application (eg, iris.py) to the Spark cluster, you need to manually specify
the packages by

.. code-block:: shell

  spark-submit \
    --master local[1] \
    --packages ml.dmlc:xgboost4j_2.12:1.6.0,ml.dmlc:xgboost4j-spark_2.12:1.6.0 \
    iris.py

Read Dataset with Spark's Built-In Reader
-----------------------------------------

The first thing in data transformation is to load the dataset as Spark's structured data abstraction, DataFrame.

.. code-block:: python


  from pyspark.sql.types import *

  schema = StructType([
      StructField("sepal length", DoubleType(), nullable=True),
      StructField("sepal width", DoubleType(), nullable=True),
      StructField("petal length", DoubleType(), nullable=True),
      StructField("petal width", DoubleType(), nullable=True),
      StructField("class", StringType(), nullable=True),
  ])
  raw_input = spark.read.schema(schema).csv("input_path")


Transform Raw Iris Dataset
--------------------------

To make Iris dataset be recognizable to XGBoost, we need to

1. Transform String-typed label, i.e. "class", to Double-typed label.
2. Assemble the feature columns as a vector to fit to the data interface of Spark ML framework.

To convert String-typed label to Double, we can use PySpark's built-in feature transformer `StringIndexer <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html>`_.

.. code-block:: python

  from pyspark.ml.feature import StringIndexer

  stringIndexer = StringIndexer(inputCol="class", outputCol="classIndex").fit(raw_input)
  labeled_input = stringIndexer.transform(raw_input).drop("class")

With a newly created StringIndexer instance:

1. we set input column, i.e. the column containing String-typed label
2. we set output column, i.e. the column to contain the Double-typed label.
3. Then we ``fit`` StringIndex with our input DataFrame ``raw_input``, so that Spark internals can get information like total number of distinct values, etc.

Now we have a StringIndexer which is ready to be applied to our input DataFrame. To execute the transformation logic of StringIndexer, we ``transform`` the input DataFrame ``raw_input`` and to keep a concise DataFrame,
we drop the column "class" and only keeps the feature columns and the transformed Double-typed label column (in the last line of the above code snippet).

The ``fit`` and ``transform`` are two key operations in MLLIB. Basically, ``fit`` produces a "transformer", e.g. StringIndexer, and each transformer applies ``transform`` method on DataFrame to add new column(s) containing transformed features/labels or prediction results, etc. To understand more about ``fit`` and ``transform``, You can find more details in `here <http://spark.apache.org/docs/latest/ml-pipeline.html#pipeline-components>`_.

Similarly, we can use another transformer, `VectorAssembler <https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html>`_, to assemble feature columns "sepal length", "sepal width", "petal length" and "petal width" as a vector.

.. code-block:: python

  from pyspark.ml.feature import  VectorAssembler
  vector_assembler = VectorAssembler()\
      .setInputCols(("sepal length", "sepal width", "petal length", "petal width"))\
      .setOutputCol("features")
  xgb_input = vector_assembler.transform(labeled_input).select("features", "classIndex")


Now, we have a DataFrame containing only two columns, "features" which contains vector-represented
"sepal length", "sepal width", "petal length" and "petal width" and "classIndex" which has Double-typed
labels. A DataFrame like this (containing vector-represented features and numeric labels) can be fed to training engine directly.

Training
========

XGBoost supports both regression and classification. While we use Iris dataset in this tutorial to show how we use xgboost-pyspark to resolve a multi-classes classification problem, the usage in Regression is very similar to classification.

To train a XGBoost model for classification, we need to claim a XGBoostClassifier first:

.. code-block:: python

  from xgboost.spark import XGBoostClassifier

  params = {
      'objective': 'multi:softprob',
      'treeMethod': 'hist',
      'numWorkers': 1,
      'numRound': 100,
      'numClass': 3,
      'labelCol': 'classIndex',
      'featuresCol': 'features'
  }

  classifier = XGBoostClassifier(**params)
  classifier.write().overwrite().save("/tmp/xgboost_classifier")
  classifier1 = XGBoostClassifier.load("/tmp/xgboost_classifier")

Equivalently, we can call the corresponding **setXXX** API to set the parameter,

.. code-block:: python

  classifier = XGBoostClassifier()\
      .setLabelCol("classIndex")\
      .setFeaturesCol("features")\
      .setTreeMethod("hist")\
      .setNumClass(3)\
      .setNumRound(100)\
      .setObjective("multi:softprob")
  classifier.setNumWorkers(1)


After we set XGBoostClassifier parameters and feature/label column, we can build a transformer, XGBoostClassificationModel by fitting XGBoostClassifier with the input DataFrame. This ``fit`` operation is essentially the training process and the generated model can then be used in prediction.

.. code-block:: python

  model = classifier.fit(xgb_input)

Prediction
==========

When we get a model, either XGBoostClassificationModel or XGBoostRegressionModel, it takes a DataFrame, read the column containing feature vectors, predict for each feature vector, and output a new DataFrame with the following columns by default:

* XGBoostClassificationModel will output margins (``rawPredictionCol``), probabilities(``probabilityCol``) and the eventual prediction labels (``predictionCol``) for each possible label.
* XGBoostRegressionModel will output prediction label(``predictionCol``).

.. code-block:: python

  model = classifier.fit(xgb_input)
  results = model.transform(xgb_input)
  results.show()

With the above code snippet, we get a result DataFrame, result containing margin, probability for each class and the prediction for each instance

.. code-block:: none

  +-----------------+----------+--------------------+--------------------+----------+
  |         features|classIndex|       rawPrediction|         probability|prediction|
  +-----------------+----------+--------------------+--------------------+----------+
  |[5.1,3.5,1.4,0.2]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[4.9,3.0,1.4,0.2]|       0.0|[3.08765506744384...|[0.99636262655258...|       0.0|
  |[4.7,3.2,1.3,0.2]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[4.6,3.1,1.5,0.2]|       0.0|[3.08765506744384...|[0.99679487943649...|       0.0|
  |[5.0,3.6,1.4,0.2]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[5.4,3.9,1.7,0.4]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[4.6,3.4,1.4,0.3]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[5.0,3.4,1.5,0.2]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[4.4,2.9,1.4,0.2]|       0.0|[3.08765506744384...|[0.99636262655258...|       0.0|
  |[4.9,3.1,1.5,0.1]|       0.0|[3.08765506744384...|[0.99679487943649...|       0.0|
  |[5.4,3.7,1.5,0.2]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[4.8,3.4,1.6,0.2]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[4.8,3.0,1.4,0.1]|       0.0|[3.08765506744384...|[0.99636262655258...|       0.0|
  |[4.3,3.0,1.1,0.1]|       0.0|[3.08765506744384...|[0.99636262655258...|       0.0|
  |[5.8,4.0,1.2,0.2]|       0.0|[3.08765506744384...|[0.99072486162185...|       0.0|
  |[5.7,4.4,1.5,0.4]|       0.0|[3.08765506744384...|[0.99072486162185...|       0.0|
  |[5.4,3.9,1.3,0.4]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[5.1,3.5,1.4,0.3]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  |[5.7,3.8,1.7,0.3]|       0.0|[3.08765506744384...|[0.99072486162185...|       0.0|
  |[5.1,3.8,1.5,0.3]|       0.0|[3.08765506744384...|[0.99680268764495...|       0.0|
  +-----------------+----------+--------------------+--------------------+----------+
