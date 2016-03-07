# XGBoost4J: Distributed XGBoost for Scala/Java
[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost) 
[![Documentation Status](https://readthedocs.org/projects/xgboost/badge/?version=latest)](https://xgboost.readthedocs.org/en/latest/jvm/index.html) 
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](../LICENSE)

[Documentation](https://xgboost.readthedocs.org/en/latest/jvm/index.html) |
[Resources](../demo/README.md) |
[Release Notes](../NEWS.md)

XGBoost4J is the JVM package of xgboost. It brings all the optimizations
and power xgboost into JVM ecosystem.

- Train XGBoost models on scala and java with easy customizations.
- Run distributed xgboost natively on jvm frameworks such as Flink and Spark.

You can find more about XGBoost on [Documentation](https://xgboost.readthedocs.org/en/latest/jvm/index.html) and [Resource Page](../demo/README.md).

## Hello World
### XGBoost Scala
```scala
import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.XGBoost

object XGBoostScalaExample {
  def main(args: Array[String]) {
    // read trainining data, available at xgboost/demo/data
    val trainData =
      new DMatrix("/path/to/agaricus.txt.train")
    // define parameters
    val paramMap = List(
      "eta" -> 0.1,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap
    // number of iterations
    val round = 2
    // train the model
    val model = XGBoost.train(paramMap, trainData, round)
    // run prediction
    val predTrain = model.predict(trainData)
    // save model to the file.
    model.saveModel("/local/path/to/model")
  }
}
```

### XGBoost Flink
```scala
import ml.dmlc.xgboost4j.scala.flink.XGBoost
import org.apache.flink.api.scala._
import org.apache.flink.api.scala.ExecutionEnvironment
import org.apache.flink.ml.MLUtils

object DistTrainWithFlink {
  def main(args: Array[String]) {
    val env: ExecutionEnvironment = ExecutionEnvironment.getExecutionEnvironment
    // read trainining data
    val trainData =
      MLUtils.readLibSVM(env, "/path/to/data/agaricus.txt.train")
    // define parameters
    val paramMap = List(
      "eta" -> 0.1,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap
    // number of iterations
    val round = 2
    // train the model
    val model = XGBoost.train(paramMap, trainData, round)
    val predTrain = model.predict(trainData.map{x => x.vector})
    model.saveModelToHadoop("file:///path/to/xgboost.model")
  }
}
```

### XGBoost Spark
