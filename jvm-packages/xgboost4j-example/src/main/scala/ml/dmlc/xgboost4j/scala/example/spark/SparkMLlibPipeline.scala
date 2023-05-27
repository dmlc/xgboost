/*
 Copyright (c) 2014 by Contributors

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

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}

// this example works with Iris dataset (https://archive.ics.uci.edu/ml/datasets/iris)

object SparkMLlibPipeline {

  def main(args: Array[String]): Unit = {

    if (args.length != 3 && args.length != 4) {
      println("Usage: SparkMLlibPipeline input_path native_model_path pipeline_model_path " +
        "[cpu|gpu]")
      sys.exit(1)
    }

    val inputPath = args(0)
    val nativeModelPath = args(1)
    val pipelineModelPath = args(2)

    val (treeMethod, numWorkers) = if (args.length == 4 && args(3) == "gpu") {
      ("gpu_hist", 1)
    } else ("auto", 2)

    val spark = SparkSession
      .builder()
      .appName("XGBoost4J-Spark Pipeline Example")
      .getOrCreate()

    run(spark, inputPath, nativeModelPath, pipelineModelPath, treeMethod, numWorkers)
      .show(false)
  }
  private[spark] def run(spark: SparkSession, inputPath: String, nativeModelPath: String,
                         pipelineModelPath: String, treeMethod: String,
                         numWorkers: Int): DataFrame = {

    // Load dataset
    val schema = new StructType(Array(
      StructField("sepal length", DoubleType, true),
      StructField("sepal width", DoubleType, true),
      StructField("petal length", DoubleType, true),
      StructField("petal width", DoubleType, true),
      StructField("class", StringType, true)))

    val rawInput = spark.read.schema(schema).csv(inputPath)

    // Split training and test dataset
    val Array(training, test) = rawInput.randomSplit(Array(0.8, 0.2), 123)

    // Build ML pipeline, it includes 4 stages:
    // 1, Assemble all features into a single vector column.
    // 2, From string label to indexed double label.
    // 3, Use XGBoostClassifier to train classification model.
    // 4, Convert indexed double label back to original string label.
    val assembler = new VectorAssembler()
      .setInputCols(Array("sepal length", "sepal width", "petal length", "petal width"))
      .setOutputCol("features")
    val labelIndexer = new StringIndexer()
      .setInputCol("class")
      .setOutputCol("classIndex")
      .fit(training)
    val booster = new XGBoostClassifier(
      Map("eta" -> 0.1f,
        "max_depth" -> 2,
        "objective" -> "multi:softprob",
        "num_class" -> 3,
        "num_round" -> 100,
        "num_workers" -> numWorkers,
        "tree_method" -> treeMethod
      )
    )
    booster.setFeaturesCol("features")
    booster.setLabelCol("classIndex")
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("realLabel")
      .setLabels(labelIndexer.labelsArray(0))

    val pipeline = new Pipeline()
      .setStages(Array(assembler, labelIndexer, booster, labelConverter))
    val model: PipelineModel = pipeline.fit(training)

    // Batch prediction
    val prediction = model.transform(test)
    prediction.show(false)

    // Model evaluation
    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("classIndex")
    evaluator.setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(prediction)
    println("The model accuracy is : " + accuracy)

    // Tune model using cross validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(booster.maxDepth, Array(3, 8))
      .addGrid(booster.eta, Array(0.2, 0.6))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(training)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(2)
      .asInstanceOf[XGBoostClassificationModel]
    println("The params of best XGBoostClassification model : " +
      bestModel.extractParamMap())
    println("The training summary of best XGBoostClassificationModel : " +
      bestModel.summary)

    // Export the XGBoostClassificationModel as local XGBoost model,
    // then you can load it back in local Python environment.
    bestModel.nativeBooster.saveModel(nativeModelPath)

    // ML pipeline persistence
    model.write.overwrite().save(pipelineModelPath)

    // Load a saved model and serving
    val model2 = PipelineModel.load(pipelineModelPath)
    model2.transform(test)
  }
}
