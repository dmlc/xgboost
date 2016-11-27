---
layout: post
title:  XGBoost4J: Portable Distributed Tree Boosting in DataFlow
date:   2016-03-15 12:00:00
author: Nan Zhu, Tianqi Chen
comments: true
---

## Introduction
[XGBoost](https://github.com/dmlc/xgboost) is a library designed and optimized for tree boosting. Gradient boosting trees model is originally proposed by Friedman et al. By embracing multi-threads and introducing regularization, XGBoost delivers higher computational power and more accurate prediction.  **More than half of the winning solutions in machine learning challenges** hosted at Kaggle adopt XGBoost ([Incomplete list](https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions)).
XGBoost has provided native interfaces for  C++, R, python, Julia and Java users.
It is used by both [data exploration and production scenarios](https://github.com/dmlc/xgboost/tree/master/demo#usecases) to solve real world machine learning problems.

The distributed XGBoost is described in the [recently published paper](http://arxiv.org/abs/1603.02754).
In short, the XGBoost system runs magnitudes faster than existing alternatives of distributed ML,
and uses far fewer resources. The reader is more than welcomed to refer to the paper for more details.

Despite the current great success, one of our ultimate goals is to make XGBoost even more available for all production scenario.
Programming languages and data processing/storage systems based on Java Virtual Machine (JVM) play the significant roles in the BigData ecosystem. [Hadoop](http://hadoop.apache.org/), [Spark](http://spark.apache.org/) and more recently introduced [Flink](http://flink.apache.org/) are very useful solutions to general large-scale data processing.

On the other side, the emerging demands of machine learning and deep learning
inspires many excellent machine learning libraries.
Many of these machine learning libraries(e.g. [XGBoost](https://github.com/dmlc/xgboost)/[MxNet](https://github.com/dmlc/mxnet))
requires new computation abstraction and native support (e.g. C++ for GPU computing).
They are also often [much more efficient](http://arxiv.org/abs/1603.02754).

The gap between the implementation fundamentals of the general data processing frameworks and the more specific machine learning libraries/systems prohibits the smooth connection between these two types of systems, thus brings unnecessary inconvenience to the end user. The common workflow to the user is to utilize the systems like Spark/Flink to preprocess/clean data, pass the results to machine learning systems like [XGBoost](https://github.com/dmlc/xgboost)/[MxNet](https://github.com/dmlc/mxnet))  via the file systems and then conduct the following machine learning phase. This process jumping across two types of systems creates certain inconvenience for the users and brings additional overhead to the operators of the infrastructure.

We want best of both worlds, so we can use the data processing frameworks like Spark and Flink together with
the best distributed machine learning solutions.
To resolve the situation, we introduce the new-brewed [XGBoost4J](https://github.com/dmlc/xgboost/tree/master/jvm-packages),
<b>XGBoost</b> for <b>J</b>VM Platform. We aim to provide the clean Java/Scala APIs and the integration with the most popular data processing systems developed in JVM-based languages.

## Unix Philosophy in Machine Learning

XGBoost and XGBoost4J adopts Unix Philosophy.
XGBoost **does its best in one thing -- tree boosting** and is **being designed to work with other systems**.
We strongly believe that machine learning solution should not be restricted to certain language or certain platform.

Specifically, users will be able to use distributed XGBoost in both Spark and Flink, and possibly more frameworks in Future.
We have made the API in a portable way so it **can be easily ported to other Dataflow frameworks provided by the Cloud**.
XGBoost4J shares its core with other XGBoost libraries, which means data scientists can use R/python
read and visualize the model trained distributedly.
It also means that user can start with single machine version for exploration,
which already can handle hundreds of million examples.

## System Overview

In the following Figure, we describe the overall architecture of XGBoost4J. XGBoost4J provides the Java/Scala API calling the core functionality of XGBoost library. Most importantly, it not only supports the single-machine model training, but also provides an abstraction layer which masks the difference of the underlying data processing engines and scales training to the distributed servers.

![XGBoost4J Architecture](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/xgboost4j.png)


By calling the XGBoost4J API, users can scale the model training to the cluster. XGBoost4J calls the running instance of XGBoost worker in Spark/Flink task and run them across the cluster. The communication among the distributed model training tasks and the XGBoost4J runtime environment go through [Rabit] (https://github.com/dmlc/rabit).

With the abstraction of XGBoost4J, users can build an unified data analytic application ranging from Extract-Transform-Loading, data exploration, machine learning model training and the final data product service. The following figure illustrate an example application built on top of Apache Spark. The application seamlessly embeds XGBoost into the processing pipeline and exchange data with other Spark-based processing phase through Spark's distributed memory layer.

![XGBoost4J Architecture](https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/unified_pipeline.png)


## Single-machine Training Walk-through

In this section, we will work through the APIs of XGBoost4J by examples.
We will be using scala for demonstration, but we also have a complete API for java users.

To start the model training and evaluation, we need to prepare the training and test set:

```scala
val trainMax = new DMatrix("../../demo/data/agaricus.txt.train")
val testMax = new DMatrix("../../demo/data/agaricus.txt.test")
```

After preparing the data, we can train our model:

```scala
val params = new mutable.HashMap[String, Any]()
params += "eta" -> 1.0
params += "max_depth" -> 2
params += "silent" -> 1
params += "objective" -> "binary:logistic"

val watches = new mutable.HashMap[String, DMatrix]
watches += "train" -> trainMax
watches += "test" -> testMax

val round = 2
// train a model
val booster = XGBoost.train(trainMax, params.toMap, round, watches.toMap)
```

We then evaluate our model:

```scala
val predicts = booster.predict(testMax)
```

`predict` can output the predict results and you can define a customized evaluation method to derive your own metrics (see the example in ([Customized Evaluation Metric in Java](https://github.com/dmlc/xgboost/blob/master/jvm-packages/xgboost4j-example/src/main/java/ml/dmlc/xgboost4j/java/example/CustomObjective.java), [Customized Evaluation Metric in Scala] (https://github.com/dmlc/xgboost/blob/master/jvm-packages/xgboost4j-example/src/main/scala/ml/dmlc/xgboost4j/scala/example/CustomObjective.scala)).


## Distributed Model Training with Distributed Dataflow Frameworks

The most exciting part in this XGBoost4J release is the integration with the Distributed Dataflow Framework. The most popular data processing frameworks fall into this category, e.g. [Apache Spark](http://spark.apache.org/), [Apache Flink] (http://flink.apache.org/), etc. In this part, we will walk through the steps to build the unified data analytic applications containing data preprocessing and distributed model training with Spark and Flink. (currently, we only provide Scala API for the integration with Spark and Flink)

Similar to the single-machine training, we need to prepare the training and test dataset.

### Spark Example

In Spark, the dataset is represented as the [Resilient Distributed Dataset (RDD)](http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds), we can utilize the Spark-distributed tools to parse libSVM file and wrap it as the RDD:

```scala
val trainRDD = MLUtils.loadLibSVMFile(sc, inputTrainPath).repartition(args(1).toInt)
```

We move forward to train the models:

```scala
val xgboostModel = XGBoost.train(trainRDD, paramMap, numRound, numWorkers)

```

The next step is to evaluate the model, you can either predict in local side or in a distributed fashion


```scala
// testSet is an RDD containing testset data represented as
// org.apache.spark.mllib.regression.LabeledPoint
val testSet = MLUtils.loadLibSVMFile(sc, inputTestPath)

// local prediction
// import methods in DataUtils to convert Iterator[org.apache.spark.mllib.regression.LabeledPoint]
// to Iterator[ml.dmlc.xgboost4j.LabeledPoint] in automatic
import DataUtils._
xgboostModel.predict(new DMatrix(testSet.collect().iterator)

// distributed prediction
xgboostModel.predict(testSet)

```
### Flink example

In Flink, we represent training data as Flink's [DataSet](https://ci.apache.org/projects/flink/flink-docs-master/apis/batch/index.html)

```scala
val trainData = MLUtils.readLibSVM(env, "/path/to/data/agaricus.txt.train")
```

Model Training can be done as follows

```scala
val xgboostModel = XGBoost.train(trainData, paramMap, round)

```

Training and prediction.

```scala
// testData is a Dataset containing testset data represented as
// org.apache.flink.ml.math.Vector.LabeledVector
val testData = MLUtils.readLibSVM(env, "/path/to/data/agaricus.txt.test")

// local prediction
xgboostModel.predict(testData.collect().iterator)

// distributed prediction
xgboostModel.predict(testData.map{x => x.vector})
```

## Road Map

It is the first release of XGBoost4J package, we are actively move forward for more charming features in the next release. You can watch our progress in [XGBoost4J Road Map](https://github.com/dmlc/xgboost/issues/935).

While we are trying our best to keep the minimum changes to the APIs, it is still subject to the incompatible changes.

## Further Readings

If you are interested in knowing more about XGBoost, you can find rich resources in

- [The github repository of XGBoost](https://github.com/dmlc/xgboost)
- [The comprehensive documentation site for XGBoostl](http://xgboost.readthedocs.org/en/latest/index.html)
- [An introduction to the gradient boosting model](http://xgboost.readthedocs.org/en/latest/model.html)
- [Tutorials for the R package](xgboost.readthedocs.org/en/latest/R-package/index.html)
- [Introduction of the Parameters](http://xgboost.readthedocs.org/en/latest/parameter.html)
- [Awesome XGBoost, a curated list of examples, tutorials, blogs about XGBoost usecases](https://github.com/dmlc/xgboost/tree/master/demo)

## Acknowledgements

We would like to send many thanks to [Zixuan Huang](https://github.com/yanqingmen), the early developer of XGBoost for Java (XGBoost for Java).
