## Introduction 

On March 2016, we released the first version of [XGBoost4J](http://dmlc.ml/2016/03/14/xgboost4j-portable-distributed-xgboost-in-spark-flink-and-dataflow.html), which is a set of packages providing Java/Scala interfaces of XGBoost and the integration with the prevalent JVM-based distributed data processing platforms, like Spark/Flink. 

The integration with Spark/Flink receives the tremendous positive feedbacks from the community. It enables users to build an unified pipeline, embedding the powerful XGBoost into the data processing system based on the prevalent frameworks like Spark. The following figure shows the general architecture of such a pipeline with the first version of XGBoost4J, where the execution of data processing is based on the low-level Spark Resilient Distributed Dataset (RDD) abstraction.

![XGBoost4J Architecture](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/unified_pipeline.png)

With the advancement of Spark, [DataFrame/Dataset-based APIs](http://spark.apache.org/docs/latest/sql-programming-guide.html) have been widely adopted. Unlike the low-level RDD abstraction, the interfaces of DataFrame/Dataset provide Spark with more information about data and computation. With this extra information, Spark is able to perform optimizations to gain the [significant performance improvement](https://databricks.com/blog/2016/07/26/introducing-apache-spark-2-0.html). 

To make the latest features of Spark be available to XGBoost users, we have been working on a full integration of XGBoost and Spark since September 2016. Last week, we have finished the integration and we will introduce the new functionalities of XGBoost4J in this blog.

## A Full Integration of XGBoost and Spark

The following figure illustrates the new pipeline architecture with the latest version of XGBoost4J. 

![XGBoost4J New Architecture](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/unified_pipeline_new.png)

Being different with the previous version, users are able to use both low- and high-level memory abstraction in Spark, i.e. RDD and DataFrame/Dataset. The DataFrame/Dataset abstraction grants the user to manipulate structured datasets and utilize the built-in routines in Spark or User Defined Functions (UDF) to explore the value distribution in columns before they feed data into the machine learning phase in the pipeline. For instance, the structured sales record can be organized as a json file and then feed to train XGBoost model.

```scala
// load sales records saved in json files
val salesDF = spark.read.json("sales.json")
// call XGBoost API to train with the DataFrame-represented training set
val xgboostModel = XGBoost.trainWithDataFrame(
      salesDF, paramMap, numRound, nWorkers, useExternalMemory)
```

In the machine learning phase, we have integrated XGBoost with Spark ML package seamlessly so that users would enjoy the battle-tested feature extractors/transformers/selectors in Spark. For example, users can easily transform the string-typed categorical feature to the numeric value and serve the transformed feature to train XGBoost model. 

```scala
import org.apache.spark.ml.feature.StringIndexer

// load sales records saved in json files
val salesDF = spark.read.json("sales.json")

// transfrom the string-represented storeType feature to numeric storeTypeIndex
val indexer = new StringIndexer()
  .setInputCol("storeType")
  .setOutputCol("storeTypeIndex")
// drop the extra column
val indexed = indexer.fit(salesDF).transform(df).drop("storeType")

// use the transformed dataframe as training dataset
val xgboostModel = XGBoost.trainWithDataFrame(
      indexed, paramMap, numRound, nWorkers, useExternalMemory)
```

The most charming benefit brought by the full integration of XGBoost and Spark DataFrame/Dataset might be the availability of pipeline and parameter tuning tool of Spark ML package to XGBoost users. The following example shows how to build such a pipeline consisting of feature transformers and the XGBoost estimator with the default learning parameters.


```scala
import org.apache.spark.ml.feature.StringIndexer

// load sales records saved in json files
val salesDF = spark.read.json("sales.json")

// transfrom the string-represented storeType feature to numeric storeTypeIndex
val indexer = new StringIndexer()
  .setInputCol("storeType")
  .setOutputCol("storeTypeIndex")

// assemble the columns in dataframe into a vector
val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("storeId", "storeTypeIndex", ...))
      .setOutputCol("features")

// construct the pipeline       
val pipeline = new Pipeline().setStages(
      Array(storeTypeIndexer, ..., vectorAssembler, new XGBoostEstimator(Map[String, Any]()))

// use the transformed dataframe as training dataset
val xgboostModel = pipeline.fit(salesDF).transform(salesDF)
```

The most critical operation to maximize the power of XGBoost is to tune the parameters. Tuning parameters manually is a tedious and labor-consuming process. With the integration of XGBoost and Spark, we can utilize the Spark model selecting tool to automate this process. The following example shows the code snippet utilizing the TrainValidationSplit and RegressionEvaluator tool to search the optimal combination of two xgboost parameters, [max_depth and eta] (https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)

```scala
val xgbEstimator = new XGBoostEstimator(xgboostParam).setFeaturesCol("features").
      setLabelCol("sales")
val paramGrid = new ParamGridBuilder()
      .addGrid(xgbEstimator.maxDepth, Array(5, 6))
      .addGrid(xgbEstimator.eta, Array(0.1, 0.4))
      .build()
val tv = new TrainValidationSplit()
      .setEstimator(xgbEstimator)
      .setEvaluator(new RegressionEvaluator().setLabelCol("sales"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)  // Use 3+ in practice
tv.fit(trainingData)
```



