/*
 Copyright (c) 2014-2022 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.example.spark

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

// this example works with Iris dataset (https://archive.ics.uci.edu/ml/datasets/iris)
object SparkTraining {

  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      // scalastyle:off
      println("Usage: program input_path [cpu|gpu]")
      sys.exit(1)
    }

    val (treeMethod, numWorkers) = if (args.length == 2 && args(1) == "gpu") {
      ("gpu_hist", 1)
    } else ("auto", 2)

    val spark = SparkSession.builder().getOrCreate()
    val inputPath = args(0)
    val results: DataFrame = run(spark, inputPath, treeMethod, numWorkers)
    results.show()
  }

private[spark] def run(spark: SparkSession, inputPath: String,
                       treeMethod: String, numWorkers: Int): DataFrame =  {
    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField("class", StringType, true)))
    val rawInput = spark.read.schema(schema).csv(inputPath)

    // transform class to index to make xgboost happy
    val stringIndexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("classIndex")
      .fit(rawInput)
    val labelTransformed = stringIndexer.transform(rawInput).drop("class")
    // compose all feature columns as vector
    val vectorAssembler = new VectorAssembler().
      setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
      setOutputCol("features")
    val xgbInput = vectorAssembler.transform(labelTransformed).select("features",
      "classIndex")

    val Array(train, eval1, eval2, test) = xgbInput.randomSplit(Array(0.6, 0.2, 0.1, 0.1))

    /**
     * setup spark.scheduler.barrier.maxConcurrentTasksCheck.interval and
     * spark.scheduler.barrier.maxConcurrentTasksCheck.maxFailures to make this application
     * if it cannot get enough resources to get 2 workers within interval * maxFailures s
     *
     * setup "checkpoint_path" -> "/checkpoints" and "checkpoint_interval" -> 2 to save
     * checkpoint for every two iterations
     */
    val xgbParam = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> numWorkers,
      "tree_method" -> treeMethod,
      "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2))
    val xgbClassifier = new XGBoostClassifier(xgbParam).
      setFeaturesCol("features").
      setLabelCol("classIndex")
    val xgbClassificationModel = xgbClassifier.fit(train)
    xgbClassificationModel.transform(test)
  }
}
