## Introduction

On March 2016, we released the first version of [XGBoost4J](http://dmlc.ml/2016/03/14/xgboost4j-portable-distributed-xgboost-in-spark-flink-and-dataflow.html), which is a set of packages providing Java/Scala interfaces of XGBoost and the integration with prevalent JVM-based distributed data processing platforms, like Spark/Flink.

The integrations with Spark/Flink, a.k.a. <b>XGBoost4J-Spark</b> and <b>XGBoost-Flink</b>, receive the tremendous positive feedbacks from the community. It enables users to build a unified pipeline, embedding  XGBoost into the data processing system based on the widely-deployed frameworks like Spark. The following figure shows the general architecture of such a pipeline with the first version of <b>XGBoost4J-Spark</b>, where the data processing is based on the low-level [Resilient Distributed Dataset (RDD)](http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds) abstraction.

![XGBoost4J Architecture](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/unified_pipeline.png)

In the last months, we have a lot of communication with the users and gain the deeper understanding of the users' latest usage scenario and requirements:

* XGBoost keeps gaining more and more deployments in the production environment and the adoption in machine learning competitions [Link](http://datascience.la/xgboost-workshop-and-meetup-talk-with-tianqi-chen/).

* While Spark is still the mainstream data processing tool in most of scenarios, more and more users are porting their RDD-based Spark programs to [DataFrame/Dataset APIs](http://spark.apache.org/docs/latest/sql-programming-guide.html) for the well-designed interfaces to manipulate structured data and the [significant performance improvement](https://databricks.com/blog/2016/07/26/introducing-apache-spark-2-0.html).

* Spark itself has presented a clear roadmap that DataFrame/Dataset would be the base of the latest and future features, e.g. latest version of [ML pipeline](http://spark.apache.org/docs/latest/ml-guide.html) and [Structured Streaming](http://spark.apache.org/docs/latest/structured-streaming-programming-guide.html).

Based on these feedbacks from the users, we observe a gap between the original RDD-based XGBoost4J-Spark and the users' latest usage scenario as well as the future direction of Spark ecosystem. To fill this gap, we start working on the <b><i>integration of XGBoost and Spark's DataFrame/Dataset abstraction</i></b> in September. In this blog, we will introduce <b>the latest version of XGBoost4J-Spark</b> which allows the user to work with DataFrame/Dataset directly and embed XGBoost to Spark's ML pipeline seamlessly.


## A Full Integration of XGBoost and DataFrame/Dataset

The following figure illustrates the new pipeline architecture with the latest XGBoost4J-Spark.

![XGBoost4J New Architecture](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/unified_pipeline_new.png)

Being different with the previous version, users are able to use both low- and high-level memory abstraction in Spark, i.e. RDD and DataFrame/Dataset. The DataFrame/Dataset abstraction grants the user to manipulate structured datasets and utilize the built-in routines in Spark or User Defined Functions (UDF) to explore the value distribution in columns before they feed data into the machine learning phase in the pipeline. In the following example, the structured sales records can be saved in a JSON file, parsed as DataFrame through Spark's API and feed to train XGBoost model in two lines of Scala code.

```scala
// load sales records saved in json files
val salesDF = spark.read.json("sales.json")
// call XGBoost API to train with the DataFrame-represented training set
val xgboostModel = XGBoost.trainWithDataFrame(
      salesDF, paramMap, numRound, nWorkers, useExternalMemory)
```

By integrating with DataFrame/Dataset, XGBoost4J-Spark not only enables users to call DataFrame/Dataset APIs directly but also make  DataFrame/Dataset-based Spark features available to XGBoost users, e.g. ML Package.

### Integration with ML Package

ML package of Spark provides a set of convenient tools for feature extraction/transformation/selection. Additionally, with the model selection tool in ML package, users can select the best model through an automatic parameter searching process which is defined with through ML package APIs. After integrating with DataFrame/Dataset abstraction, these charming features in ML package are also available to XGBoost users.

#### Feature Extraction/Transformation/Selection

The following example shows a feature transformer which converts the string-typed storeType feature to the numeric storeTypeIndex. The transformed DataFrame is then fed to train XGBoost model.

```scala
import org.apache.spark.ml.feature.StringIndexer

// load sales records saved in json files
val salesDF = spark.read.json("sales.json")

// transform the string-represented storeType feature to numeric storeTypeIndex
val indexer = new StringIndexer()
  .setInputCol("storeType")
  .setOutputCol("storeTypeIndex")
// drop the extra column
val indexed = indexer.fit(salesDF).transform(df).drop("storeType")

// use the transformed dataframe as training dataset
val xgboostModel = XGBoost.trainWithDataFrame(
      indexed, paramMap, numRound, nWorkers, useExternalMemory)
```

#### Pipelining

Spark ML package allows the user to build a complete pipeline from feature extraction/transformation/selection to model training. We integrate XGBoost with ML package and make it feasible to embed XGBoost into such a pipeline seamlessly. The following example shows how to build such a pipeline consisting of feature transformers and the XGBoost estimator.

```scala
import org.apache.spark.ml.feature.StringIndexer

// load sales records saved in json files
val salesDF = spark.read.json("sales.json")

// transform the string-represented storeType feature to numeric storeTypeIndex
val indexer = new StringIndexer()
  .setInputCol("storeType")
  .setOutputCol("storeTypeIndex")

// assemble the columns in dataframe into a vector
val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("storeId", "storeTypeIndex", ...))
      .setOutputCol("features")

// construct the pipeline       
val pipeline = new Pipeline().setStages(
      Array(storeTypeIndexer, ..., vectorAssembler, new XGBoostEstimator(Map[String, Any]("num_rounds" -> 100)))

// use the transformed dataframe as training dataset
val xgboostModel = pipeline.fit(salesDF)

// predict with the trained model
val salesTestDF = spark.read.json("sales_test.json")
val salesRecordsWithPred = xgboostModel.transform(salesTestDF)

```

#### Model Selection

The most critical operation to maximize the power of XGBoost is to select the optimal parameters for the model. Tuning parameters manually is a tedious and labor-consuming process. With the latest version of XGBoost4J-Spark, we can utilize the Spark model selecting tool to automate this process. The following example shows the code snippet utilizing [TrainValidationSplit](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.tuning.TrainValidationSplit) and [RegressionEvaluator](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.evaluation.RegressionEvaluator) to search the optimal combination of two XGBoost parameters, [max_depth and eta] (https://github.com/dmlc/xgboost/blob/master/doc/parameter.md). The model producing the minimum cost function value defined by RegressionEvaluator is selected and used to generate the prediction for the test set.

```scala
// create XGBoostEstimator
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
      .setTrainRatio(0.8)  
val salesTestDF = spark.read.json("sales_test.json")
val salesRecordsWithPred = xgboostModel.transform(salesTestDF)
```

## Summary

Through the latest XGBoost4J-Spark, XGBoost users can build a more efficient data processing pipeline which works with DataFrame/Dataset APIs to handle the structured data with the excellent performance, and simultaneously embrace the powerful XGBoost to explore the insights from the dataset and transform this insight into action. Additionally, XGBoost4J-Spark seamlessly connect XGBoost with Spark ML package which makes the job of feature extraction/transformation/selection and parameter model much easier than before.

The latest version of XGBoost4J-Spark has been available in the [GitHub Repository] (https://github.com/dmlc/xgboost), and the latest API docs are in [here](http://xgboost.readthedocs.io/en/latest/jvm/index.html).


## Portable Machine Learning Systems

XGBoost is one of the projects incubated by [Distributed Machine Learning Community (DMLC)](http://dmlc.ml/), which also creates several other popular projects on machine learning systems ([Link](https://github.com/dmlc/)), e.g. one of the most popular deep learning frameworks, [MXNet](http://mxnet.io/). We strongly believe that machine learning solution should not be restricted to certain language or certain platform. We realize this design philosophy in several projects, like XGBoost and MXNet. We are willing to see more contributions from the community in this direction.


## Further Readings

If you are interested in knowing more about XGBoost, you can find rich resources in

- [The github repository of XGBoost](https://github.com/dmlc/xgboost)
- [The comprehensive documentation site for XGBoostl](http://xgboost.readthedocs.org/en/latest/index.html)
- [An introduction to the gradient boosting model](http://xgboost.readthedocs.org/en/latest/model.html)
- [Tutorials for the R package](xgboost.readthedocs.org/en/latest/R-package/index.html)
- [Introduction of the Parameters](http://xgboost.readthedocs.org/en/latest/parameter.html)
- [Awesome XGBoost, a curated list of examples, tutorials, blogs about XGBoost usecases](https://github.com/dmlc/xgboost/tree/master/demo)
