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

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier}

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType

class GpuXGBoostGeneralSuite extends GpuTestSuite {

  private val labelName = "label_col"
  private val weightName = "weight_col"
  private val baseMarginName = "margin_col"
  private val featureNames = Array("f1", "f2", "f3")
  private val allColumnNames = featureNames :+ weightName :+ baseMarginName :+ labelName
  private val trainingData = Seq(
    // f1,  f2,  f3, weight, margin, label
    (1.0f, 2.0f, 3.0f, 1.0f, 0.5f, 0),
    (2.0f, 3.0f, 4.0f, 2.0f, 0.6f, 0),
    (1.2f, 2.1f, 3.1f, 1.1f, 0.51f, 0),
    (2.3f, 3.1f, 4.1f, 2.1f, 0.61f, 0),
    (3.0f, 4.0f, 5.0f, 1.5f, 0.3f, 1),
    (4.0f, 5.0f, 6.0f, 2.5f, 0.4f, 1),
    (3.1f, 4.1f, 5.1f, 1.6f, 0.4f, 1),
    (4.1f, 5.1f, 6.1f, 2.6f, 0.5f, 1),
    (5.0f, 6.0f, 7.0f, 1.0f, 0.2f, 2),
    (6.0f, 7.0f, 8.0f, 1.3f, 0.6f, 2),
    (5.1f, 6.1f, 7.1f, 1.2f, 0.1f, 2),
    (6.1f, 7.1f, 8.1f, 1.4f, 0.7f, 2),
    (6.2f, 7.2f, 8.2f, 1.5f, 0.8f, 2))

  test("MLlib way setting features_cols should work") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val trainingDf = trainingData.toDF(allColumnNames: _*)
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "multi:softprob",
        "num_class" -> 3, "num_round" -> 5, "num_workers" -> 1, "tree_method" -> "gpu_hist",
        "features_cols" -> featureNames, "label_col" -> labelName)
      new XGBoostClassifier(xgbParam)
        .fit(trainingDf)
    }
  }

  test("disorder feature columns should work") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      var trainingDf = trainingData.toDF(allColumnNames: _*)

      trainingDf = trainingDf.select(labelName, "f2", weightName, "f3", baseMarginName, "f1")

      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "multi:softprob",
        "num_class" -> 3, "num_round" -> 5, "num_workers" -> 1, "tree_method" -> "gpu_hist")
      new XGBoostClassifier(xgbParam)
        .setFeaturesCol(featureNames)
        .setLabelCol(labelName)
        .fit(trainingDf)
    }
  }

  test("Throw exception when feature/label columns are not numeric type") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val originalDf = trainingData.toDF(allColumnNames: _*)
      var trainingDf = originalDf.withColumn("f2", col("f2").cast(StringType))

      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "multi:softprob",
        "num_class" -> 3, "num_round" -> 5, "num_workers" -> 1, "tree_method" -> "gpu_hist")
      val thrown1 = intercept[IllegalArgumentException] {
        new XGBoostClassifier(xgbParam)
          .setFeaturesCol(featureNames)
          .setLabelCol(labelName)
          .fit(trainingDf)
      }
      assert(thrown1.getMessage.contains("Column f2 must be of NumericType but found: string."))

      trainingDf = originalDf.withColumn(labelName, col(labelName).cast(StringType))
      val thrown2 = intercept[IllegalArgumentException] {
        new XGBoostClassifier(xgbParam)
          .setFeaturesCol(featureNames)
          .setLabelCol(labelName)
          .fit(trainingDf)
      }
      assert(thrown2.getMessage.contains(
        s"Column $labelName must be of NumericType but found: string."))
    }
  }

  test("Throw exception when features_cols or label_col is not set") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val trainingDf = trainingData.toDF(allColumnNames: _*)
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "multi:softprob",
        "num_class" -> 3, "num_round" -> 5, "num_workers" -> 1, "tree_method" -> "gpu_hist")

      // GPU train requires featuresCols. If not specified,
      // then NoSuchElementException will be thrown
      val thrown = intercept[NoSuchElementException] {
        new XGBoostClassifier(xgbParam)
          .setLabelCol(labelName)
          .fit(trainingDf)
      }
      assert(thrown.getMessage.contains("Failed to find a default value for featuresCols"))

      val thrown1 = intercept[IllegalArgumentException] {
        new XGBoostClassifier(xgbParam)
          .setFeaturesCol(featureNames)
          .fit(trainingDf)
      }
      assert(thrown1.getMessage.contains("label does not exist."))
    }
  }

  test("Throw exception when tree method is not set to gpu_hist") {
    withGpuSparkSession() { spark =>
      import spark.implicits._
      val trainingDf = trainingData.toDF(allColumnNames: _*)
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "multi:softprob",
        "num_class" -> 3, "num_round" -> 5, "num_workers" -> 1, "tree_method" -> "hist")
      val thrown = intercept[IllegalArgumentException] {
        new XGBoostClassifier(xgbParam)
          .setFeaturesCol(featureNames)
          .setLabelCol(labelName)
          .fit(trainingDf)
      }
      assert(thrown.getMessage.contains("GPU train requires tree_method set to gpu_hist"))
    }
  }

  test("Train with eval") {

    withGpuSparkSession() { spark =>
      import spark.implicits._
      val Array(trainingDf, eval1, eval2) = trainingData.toDF(allColumnNames: _*)
        .randomSplit(Array(0.6, 0.2, 0.2), seed = 1)
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "multi:softprob",
        "num_class" -> 3, "num_round" -> 5, "num_workers" -> 1, "tree_method" -> "gpu_hist")
      val model1 = new XGBoostClassifier(xgbParam)
        .setFeaturesCol(featureNames)
        .setLabelCol(labelName)
        .setEvalSets(Map("eval1" -> eval1, "eval2" -> eval2))
        .fit(trainingDf)

      assert(model1.summary.validationObjectiveHistory.length === 2)
      assert(model1.summary.validationObjectiveHistory.map(_._1).toSet === Set("eval1", "eval2"))
      assert(model1.summary.validationObjectiveHistory(0)._2.length === 5)
      assert(model1.summary.validationObjectiveHistory(1)._2.length === 5)
      assert(model1.summary.trainObjectiveHistory !== model1.summary.validationObjectiveHistory(0))
      assert(model1.summary.trainObjectiveHistory !== model1.summary.validationObjectiveHistory(1))
    }
  }

  test("test persistence of XGBoostClassifier and XGBoostClassificationModel") {
    val xgbcPath = new File(tempDir.toFile, "xgbc").getPath
    withGpuSparkSession() { spark =>
      val xgbParam = Map("eta" -> 0.1f, "max_depth" -> 2, "objective" -> "multi:softprob",
        "num_class" -> 3, "num_round" -> 5, "num_workers" -> 1, "tree_method" -> "gpu_hist",
        "features_cols" -> featureNames, "label_col" -> labelName)
      val xgbc = new XGBoostClassifier(xgbParam)
      xgbc.write.overwrite().save(xgbcPath)
      val paramMap2 = XGBoostClassifier.load(xgbcPath).MLlib2XGBoostParams
      xgbParam.foreach {
        case (k, v: Array[String]) =>
          assert(v.sameElements(paramMap2(k).asInstanceOf[Array[String]]))
        case (k, v) =>
          assert(v.toString == paramMap2(k).toString)
      }
    }
  }

}
