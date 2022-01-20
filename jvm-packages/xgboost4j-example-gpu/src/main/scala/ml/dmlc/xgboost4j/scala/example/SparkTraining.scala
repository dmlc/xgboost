/*
 Copyright (c) 2022 by Contributors

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

package ml.dmlc.xgboost4j.scala.example

import java.util.{Locale, TimeZone}

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.dense_rank
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

// this example works with Iris dataset (https://archive.ics.uci.edu/ml/datasets/iris)
object SparkTraining {

  def withSparkSession[T](f: SparkSession => T): T = {
    // Timezone is fixed to UTC to allow timestamps to work by default
    TimeZone.setDefault(TimeZone.getTimeZone("UTC"))
    // Add Locale setting
    Locale.setDefault(Locale.US)

    val spark = SparkSession.builder()
      .appName("Train IRIS on GPU")
      .getOrCreate()
    try {
      f(spark)
    } finally {
      spark.stop()
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length != 1) {
      // scalastyle:off
      println("Usage: program input_path")
      sys.exit(1)
    }

    val inputPath = args(0)

    val labelName = "class"

    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField(labelName, StringType, true)))

    withSparkSession { spark =>
      val rawInput = spark.read.schema(schema).csv(inputPath)

      // 1. transform class to index to make xgboost happy
      // We'd better avoid using StringIndexer, since it has not been been accelerated by GPU
      // There are many ways to do that. This sample chooses dense_rank to implement it.
      val spec = Window.orderBy(labelName)
      val xgbInput = rawInput
        .withColumn("tmpClassName", dense_rank().over(spec) - 1)
        .drop(labelName)
        .withColumnRenamed("tmpClassName", labelName)

      // 2. prepare feature columns
      val features = schema.fieldNames.filter(_.equals(labelName))

      // 3. prepare train/evals/test Datasets
      val Array(train, eval1, eval2, test) = xgbInput.randomSplit(Array(0.6, 0.2, 0.1, 0.1))
      train.explain(true)

      val xgbParam = Map(
        "eta" -> 0.1f,
        "max_depth" -> 2,
        "objective" -> "multi:softprob",
        "num_class" -> 3,
        "num_round" -> 100,
        "num_workers" -> 1,
        "tree_method" -> "gpu_hist",
        "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2))

      val xgbClassifier = new XGBoostClassifier(xgbParam)
        .setFeaturesCols(features)
        .setLabelCol(labelName)

      val xgbClassificationModel = xgbClassifier.fit(train)
      val results = xgbClassificationModel.transform(test)

      results.show()
    }

  }
}
