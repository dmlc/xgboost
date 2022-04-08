/*
 Copyright (c) 2021-2022 by Contributors

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

package ml.dmlc.xgboost4j.scala.rapids.spark

import java.io.File

import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel, XGBoostRegressor}

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}

class GpuXGBoostRegressorSuite extends GpuTestSuite {

  val labelName = "label_col"
  val groupName = "group_col"
  val schema = StructType(Seq(
    StructField(labelName, FloatType),
    StructField("f1", FloatType),
    StructField("f2", FloatType),
    StructField("f3", FloatType),
    StructField(groupName, IntegerType)))
  val featureNames = schema.fieldNames.filter(s =>
    !(s.equals(labelName) || s.equals(groupName)))

  test("The transform result should be same for several runs on same model") {
    withGpuSparkSession(enableCsvConf()) { spark =>
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "reg:squarederror",
        "num_round" -> 10, "num_workers" -> 1, "tree_method" -> "gpu_hist",
        "features_cols" -> featureNames, "label_col" -> labelName)
      val Array(originalDf, testDf) = spark.read.option("header", "true").schema(schema)
        .csv(getResourcePath("/rank.train.csv")).randomSplit(Array(0.7, 0.3), seed = 1)
      // Get a model
      val model = new XGBoostRegressor(xgbParam)
        .fit(originalDf)
      val left = model.transform(testDf).collect()
      val right = model.transform(testDf).collect()
      // The left should be same with right
      assert(compareResults(true, 0.000001, left, right))
    }
  }

  test("use weight") {
    withGpuSparkSession(enableCsvConf()) { spark =>
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "reg:squarederror",
        "num_round" -> 10, "num_workers" -> 1, "tree_method" -> "gpu_hist",
        "features_cols" -> featureNames, "label_col" -> labelName)
      val Array(originalDf, testDf) = spark.read.option("header", "true").schema(schema)
        .csv(getResourcePath("/rank.train.csv")).randomSplit(Array(0.7, 0.3), seed = 1)
      val getWeightFromF1 = udf({ f1: Float => if (f1.toInt % 2 == 0) 1.0f else 0.001f })
      val dfWithWeight = originalDf.withColumn("weight", getWeightFromF1(col("f1")))

      val model = new XGBoostRegressor(xgbParam)
        .fit(originalDf)
      val model2 = new XGBoostRegressor(xgbParam)
        .setWeightCol("weight")
        .fit(dfWithWeight)

      val left = model.transform(testDf).collect()
      val right = model2.transform(testDf).collect()
      // left should be different with right
      assert(!compareResults(true, 0.000001, left, right))
    }
  }

  test("Save model and transform GPU dataset") {
    // Train a model on GPU
    val (gpuModel, testDf) = withGpuSparkSession(enableCsvConf()) { spark =>
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "binary:logistic",
        "num_round" -> 10, "num_workers" -> 1)
      val Array(rawInput, testDf) = spark.read.option("header", "true").schema(schema)
        .csv(getResourcePath("/rank.train.csv")).randomSplit(Array(0.7, 0.3), seed = 1)

      val classifier = new XGBoostRegressor(xgbParam)
        .setFeaturesCol(featureNames)
        .setLabelCol(labelName)
        .setTreeMethod("gpu_hist")
      (classifier.fit(rawInput), testDf)
    }

    val xgbrModel = new File(tempDir.toFile, "xgbrModel").getPath
    gpuModel.write.overwrite().save(xgbrModel)
    val gpuModelFromFile = XGBoostRegressionModel.load(xgbrModel)

    // transform on GPU
    withGpuSparkSession() { spark =>
      val left = gpuModel
        .transform(testDf)
        .select(labelName, "prediction")
        .collect()

      val right = gpuModelFromFile
        .transform(testDf)
        .select(labelName, "prediction")
        .collect()

      assert(compareResults(true, 0.000001, left, right))
    }
  }

  test("Model trained on CPU can transform GPU dataset") {
    // Train a model on CPU
    val cpuModel = withCpuSparkSession() { spark =>
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "reg:squarederror",
        "num_round" -> 10, "num_workers" -> 1)
      val Array(rawInput, _) = spark.read.option("header", "true").schema(schema)
        .csv(getResourcePath("/rank.train.csv")).randomSplit(Array(0.7, 0.3), seed = 1)

      val vectorAssembler = new VectorAssembler()
        .setHandleInvalid("keep")
        .setInputCols(featureNames)
        .setOutputCol("features")
      val trainingDf = vectorAssembler.transform(rawInput).select("features", labelName)

      val classifier = new XGBoostRegressor(xgbParam)
        .setFeaturesCol("features")
        .setLabelCol(labelName)
        .setTreeMethod("auto")
      classifier.fit(trainingDf)
    }

    val xgbrModel = new File(tempDir.toFile, "xgbrModel").getPath
    cpuModel.write.overwrite().save(xgbrModel)
    val cpuModelFromFile = XGBoostRegressionModel.load(xgbrModel)

    // transform on GPU
    withGpuSparkSession() { spark =>
      val Array(_, testDf) = spark.read.option("header", "true").schema(schema)
        .csv(getResourcePath("/rank.train.csv")).randomSplit(Array(0.7, 0.3), seed = 1)

      // Since CPU model does not know the information about the features cols that GPU transform
      // pipeline requires. End user needs to setFeaturesCol(features: Array[String]) in the model
      // manually
      val thrown = intercept[NoSuchElementException](cpuModel
        .transform(testDf)
        .collect())
      assert(thrown.getMessage.contains("Failed to find a default value for featuresCols"))

      val left = cpuModel
        .setFeaturesCol(featureNames)
        .transform(testDf)
        .collect()

      val right = cpuModelFromFile
        .setFeaturesCol(featureNames)
        .transform(testDf)
        .collect()

      assert(compareResults(true, 0.000001, left, right))
    }
  }

  test("Model trained on GPU can transform CPU dataset") {
    // Train a model on GPU
    val gpuModel = withGpuSparkSession(enableCsvConf()) { spark =>
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "reg:squarederror",
        "num_round" -> 10, "num_workers" -> 1)
      val Array(rawInput, _) = spark.read.option("header", "true").schema(schema)
        .csv(getResourcePath("/rank.train.csv")).randomSplit(Array(0.7, 0.3), seed = 1)

      val classifier = new XGBoostRegressor(xgbParam)
        .setFeaturesCol(featureNames)
        .setLabelCol(labelName)
        .setTreeMethod("gpu_hist")
      classifier.fit(rawInput)
    }

    val xgbrModel = new File(tempDir.toFile, "xgbrModel").getPath
    gpuModel.write.overwrite().save(xgbrModel)
    val gpuModelFromFile = XGBoostRegressionModel.load(xgbrModel)

    // transform on CPU
    withCpuSparkSession() { spark =>
      val Array(_, rawInput) = spark.read.option("header", "true").schema(schema)
        .csv(getResourcePath("/rank.train.csv")).randomSplit(Array(0.7, 0.3), seed = 1)

      val featureColName = "feature_col"
      val vectorAssembler = new VectorAssembler()
        .setHandleInvalid("keep")
        .setInputCols(featureNames)
        .setOutputCol(featureColName)
      val testDf = vectorAssembler.transform(rawInput).select(featureColName, labelName)

      // Since GPU model does not know the information about the features col name that CPU
      // transform pipeline requires. End user needs to setFeaturesCol in the model manually
      intercept[IllegalArgumentException](
        gpuModel
        .transform(testDf)
        .collect())

      val left = gpuModel
        .setFeaturesCol(featureColName)
        .transform(testDf)
        .select(labelName, "prediction")
        .collect()

      val right = gpuModelFromFile
        .setFeaturesCol(featureColName)
        .transform(testDf)
        .select(labelName, "prediction")
        .collect()

      assert(compareResults(true, 0.000001, left, right))
    }
  }

  test("Ranking: train with Group") {
    withGpuSparkSession(enableCsvConf()) { spark =>
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "rank:pairwise",
        "num_round" -> 10, "num_workers" -> 1, "tree_method" -> "gpu_hist",
        "features_cols" -> featureNames, "label_col" -> labelName)
      val Array(trainingDf, testDf) = spark.read.option("header", "true").schema(schema)
        .csv(getResourcePath("/rank.train.csv")).randomSplit(Array(0.7, 0.3), seed = 1)

      val model = new XGBoostRegressor(xgbParam)
        .setGroupCol(groupName)
        .fit(trainingDf)

      val ret = model.transform(testDf).collect()
      assert(testDf.count() === ret.length)
    }
  }

}
