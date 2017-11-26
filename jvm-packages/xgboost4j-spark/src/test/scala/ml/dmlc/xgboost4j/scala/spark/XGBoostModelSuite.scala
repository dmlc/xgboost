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

package ml.dmlc.xgboost4j.scala.spark

import java.nio.file.Files

import ml.dmlc.xgboost4j.scala.DMatrix
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite

class XGBoostModelSuite extends FunSuite with PerTest {
  test("test model consistency after save and load") {
    import DataUtils._
    val eval = new EvalError()
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testSetDMatrix = new DMatrix(Classification.test.iterator)
    val tempDir = Files.createTempDirectory("xgboosttest-")
    val tempFile = Files.createTempFile(tempDir, "", "")
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic")
    val xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    val evalResults = eval.eval(xgBoostModel.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix)
    assert(evalResults < 0.1)
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    val loadedXGBooostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    val predicts = loadedXGBooostModel.booster.predict(testSetDMatrix, outPutMargin = true)
    val loadedEvalResults = eval.eval(predicts, testSetDMatrix)
    assert(loadedEvalResults == evalResults)
  }

  test("test save and load of different types of models") {
    import DataUtils._
    val tempDir = Files.createTempDirectory("xgboosttest-")
    val tempFile = Files.createTempFile(tempDir, "", "")
    var trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    var paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "reg:linear")
    // validate regression model
    var xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers, useExternalMemory = false)
    xgBoostModel.setFeaturesCol("feature_col")
    xgBoostModel.setLabelCol("label_col")
    xgBoostModel.setPredictionCol("prediction_col")
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    var loadedXGBoostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    assert(loadedXGBoostModel.isInstanceOf[XGBoostRegressionModel])
    assert(loadedXGBoostModel.getFeaturesCol == "feature_col")
    assert(loadedXGBoostModel.getLabelCol == "label_col")
    assert(loadedXGBoostModel.getPredictionCol == "prediction_col")
    // classification model
    paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic")
    xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers, useExternalMemory = false)
    xgBoostModel.asInstanceOf[XGBoostClassificationModel].setRawPredictionCol("raw_col")
    xgBoostModel.asInstanceOf[XGBoostClassificationModel].setThresholds(Array(0.5, 0.5))
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    loadedXGBoostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    assert(loadedXGBoostModel.isInstanceOf[XGBoostClassificationModel])
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].getRawPredictionCol ==
        "raw_col")
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].getThresholds.deep ==
        Array(0.5, 0.5).deep)
    assert(loadedXGBoostModel.getFeaturesCol == "features")
    assert(loadedXGBoostModel.getLabelCol == "label")
    assert(loadedXGBoostModel.getPredictionCol == "prediction")
    // (multiclass) classification model
    trainingRDD = sc.parallelize(MultiClassification.train).map(_.asML)
    paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6")
    xgBoostModel = XGBoost.trainWithRDD(trainingRDD, paramMap, round = 5,
      nWorkers = numWorkers, useExternalMemory = false)
    xgBoostModel.asInstanceOf[XGBoostClassificationModel].setRawPredictionCol("raw_col")
    xgBoostModel.asInstanceOf[XGBoostClassificationModel].setThresholds(
      Array(0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    xgBoostModel.saveModelAsHadoopFile(tempFile.toFile.getAbsolutePath)
    loadedXGBoostModel = XGBoost.loadModelFromHadoopFile(tempFile.toFile.getAbsolutePath)
    assert(loadedXGBoostModel.isInstanceOf[XGBoostClassificationModel])
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].getRawPredictionCol ==
        "raw_col")
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].getThresholds.deep ==
        Array(0.5, 0.5, 0.5, 0.5, 0.5, 0.5).deep)
    assert(loadedXGBoostModel.asInstanceOf[XGBoostClassificationModel].numOfClasses == 6)
    assert(loadedXGBoostModel.getFeaturesCol == "features")
    assert(loadedXGBoostModel.getLabelCol == "label")
    assert(loadedXGBoostModel.getPredictionCol == "prediction")
  }

  test("copy and predict ClassificationModel") {
    import DataUtils._
    val trainingRDD = sc.parallelize(Classification.train).map(_.asML)
    val testRDD = sc.parallelize(Classification.test).map(_.features)
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "binary:logistic")
    val model = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    testCopy(model, testRDD)
  }

  test("copy and predict RegressionModel") {
    import DataUtils._
    val trainingRDD = sc.parallelize(Regression.train).map(_.asML)
    val testRDD = sc.parallelize(Regression.test).map(_.features)
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "1",
      "objective" -> "reg:linear")
    val model = XGBoost.trainWithRDD(trainingRDD, paramMap, 5, numWorkers)
    testCopy(model, testRDD)
  }

  private def testCopy(model: XGBoostModel, testRDD: RDD[Vector]): Unit = {
    val modelCopy = model.copy(ParamMap.empty)
    modelCopy.summary  // Ensure no exception.

    val expected = model.predict(testRDD).collect
    assert(modelCopy.predict(testRDD).collect === expected)
  }
}
