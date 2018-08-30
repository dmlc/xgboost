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

import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql._
import org.scalatest.FunSuite

class XGBoostClassifierSuite extends FunSuite with PerTest {

  test("XGBoost-Spark XGBoostClassifier ouput should match XGBoost4j") {
    val trainingDM = new DMatrix(Classification.train.iterator)
    val testDM = new DMatrix(Classification.test.iterator)
    val trainingDF = buildDataFrame(Classification.train)
    val testDF = buildDataFrame(Classification.test)
    val round = 5

    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "silent" -> "1",
      "objective" -> "binary:logistic")

    val model1 = ScalaXGBoost.train(trainingDM, paramMap, round)
    val prediction1 = model1.predict(testDM)

    val model2 = new XGBoostClassifier(paramMap ++ Array("num_round" -> round,
      "num_workers" -> numWorkers)).fit(trainingDF)

    val prediction2 = model2.transform(testDF).
      collect().map(row => (row.getAs[Int]("id"), row.getAs[DenseVector]("probability"))).toMap

    assert(testDF.count() === prediction2.size)
    // the vector length in probability column is 2 since we have to fit to the evaluator in Spark
    for (i <- prediction1.indices) {
      assert(prediction1(i).length === prediction2(i).values.length - 1)
      for (j <- prediction1(i).indices) {
        assert(prediction1(i)(j) === prediction2(i)(j + 1))
      }
    }

    val prediction3 = model1.predict(testDM, outPutMargin = true)
    val prediction4 = model2.transform(testDF).
      collect().map(row => (row.getAs[Int]("id"), row.getAs[DenseVector]("rawPrediction"))).toMap

    assert(testDF.count() === prediction4.size)
    for (i <- prediction3.indices) {
      assert(prediction3(i).length === prediction4(i).values.length)
      for (j <- prediction3(i).indices) {
        assert(prediction3(i)(j) === prediction4(i)(j))
      }
    }

    // check the equality of single instance prediction
    val firstOfDM = testDM.slice(Array(0))
    val firstOfDF = testDF.head().getAs[Vector]("features")
    val prediction5 = math.round(model1.predict(firstOfDM)(0)(0))
    val prediction6 = model2.predict(firstOfDF)
    assert(prediction5 === prediction6)
  }

  test("Set params in XGBoost and MLlib way should produce same model") {
    val trainingDF = buildDataFrame(Classification.train)
    val testDF = buildDataFrame(Classification.test)
    val round = 5

    val paramMap = Map(
      "eta" -> "1",
      "max_depth" -> "6",
      "silent" -> "1",
      "objective" -> "binary:logistic",
      "num_round" -> round,
      "num_workers" -> numWorkers)

    // Set params in XGBoost way
    val model1 = new XGBoostClassifier(paramMap).fit(trainingDF)
    // Set params in MLlib way
    val model2 = new XGBoostClassifier()
      .setEta(1)
      .setMaxDepth(6)
      .setSilent(1)
      .setObjective("binary:logistic")
      .setNumRound(round)
      .setNumWorkers(numWorkers)
      .fit(trainingDF)

    val prediction1 = model1.transform(testDF).select("prediction").collect()
    val prediction2 = model2.transform(testDF).select("prediction").collect()

    prediction1.zip(prediction2).foreach { case (Row(p1: Double), Row(p2: Double)) =>
      assert(p1 === p2)
    }
  }

  test("test schema of XGBoostClassificationModel") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "num_workers" -> numWorkers)
    val trainingDF = buildDataFrame(Classification.train)
    val testDF = buildDataFrame(Classification.test)

    val model = new XGBoostClassifier(paramMap).fit(trainingDF)

    model.setRawPredictionCol("raw_prediction")
      .setProbabilityCol("probability_prediction")
      .setPredictionCol("final_prediction")
    var predictionDF = model.transform(testDF)
    assert(predictionDF.columns.contains("id"))
    assert(predictionDF.columns.contains("features"))
    assert(predictionDF.columns.contains("label"))
    assert(predictionDF.columns.contains("raw_prediction"))
    assert(predictionDF.columns.contains("probability_prediction"))
    assert(predictionDF.columns.contains("final_prediction"))
    model.setRawPredictionCol("").setPredictionCol("final_prediction")
    predictionDF = model.transform(testDF)
    assert(predictionDF.columns.contains("raw_prediction") === false)
    assert(predictionDF.columns.contains("final_prediction"))
    model.setRawPredictionCol("raw_prediction").setPredictionCol("")
    predictionDF = model.transform(testDF)
    assert(predictionDF.columns.contains("raw_prediction"))
    assert(predictionDF.columns.contains("final_prediction") === false)

    assert(model.summary.trainObjectiveHistory.length === 5)
    assert(model.summary.testObjectiveHistory.isEmpty)
  }

  test("XGBoost and Spark parameters synchronize correctly") {
    val xgbParamMap = Map("eta" -> "1", "objective" -> "binary:logistic",
      "obj_type" -> "classification")
    // from xgboost params to spark params
    val xgb = new XGBoostClassifier(xgbParamMap)
    assert(xgb.getEta === 1.0)
    assert(xgb.getObjective === "binary:logistic")
    assert(xgb.getObjectiveType === "classification")
    // from spark to xgboost params
    val xgbCopy = xgb.copy(ParamMap.empty)
    assert(xgbCopy.MLlib2XGBoostParams("eta").toString.toDouble === 1.0)
    assert(xgbCopy.MLlib2XGBoostParams("objective").toString === "binary:logistic")
    assert(xgbCopy.MLlib2XGBoostParams("obj_type").toString === "classification")
    val xgbCopy2 = xgb.copy(ParamMap.empty.put(xgb.evalMetric, "logloss"))
    assert(xgbCopy2.MLlib2XGBoostParams("eval_metric").toString === "logloss")
  }

  test("multi class classification") {
    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6", "num_round" -> 5,
      "num_workers" -> numWorkers)
    val trainingDF = buildDataFrame(MultiClassification.train)
    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(trainingDF)
    assert(model.getEta == 0.1)
    assert(model.getMaxDepth == 6)
    assert(model.numClasses == 6)
  }

  test("use base margin") {
    val training1 = buildDataFrame(Classification.train)
    val training2 = training1.withColumn("margin", functions.rand())
    val test = buildDataFrame(Classification.test)
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "test_train_split" -> "0.5",
      "num_round" -> 5, "num_workers" -> numWorkers)

    val xgb = new XGBoostClassifier(paramMap)
    val model1 = xgb.fit(training1)
    val model2 = xgb.setBaseMarginCol("margin").fit(training2)
    val prediction1 = model1.transform(test).select(model1.getProbabilityCol)
      .collect().map(row => row.getAs[Vector](0))
    val prediction2 = model2.transform(test).select(model2.getProbabilityCol)
      .collect().map(row => row.getAs[Vector](0))
    var count = 0
    for ((r1, r2) <- prediction1.zip(prediction2)) {
      if (!r1.equals(r2)) count = count + 1
    }
    assert(count != 0)
  }

  test("training summary") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "num_round" -> 5, "nWorkers" -> numWorkers)

    val trainingDF = buildDataFrame(Classification.train)
    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(trainingDF)

    assert(model.summary.trainObjectiveHistory.length === 5)
    assert(model.summary.testObjectiveHistory.isEmpty)
  }

  test("train/test split") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "train_test_ratio" -> "0.5",
      "num_round" -> 5, "num_workers" -> numWorkers)
    val training = buildDataFrame(Classification.train)

    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(training)
    val Some(testObjectiveHistory) = model.summary.testObjectiveHistory
    assert(testObjectiveHistory.length === 5)
    assert(model.summary.trainObjectiveHistory !== testObjectiveHistory)
  }

  test("test predictionLeaf") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "train_test_ratio" -> "0.5",
      "num_round" -> 5, "num_workers" -> numWorkers)
    val training = buildDataFrame(Classification.train)
    val test = buildDataFrame(Classification.test)
    val groundTruth = test.count()
    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(training)
    model.setLeafPredictionCol("predictLeaf")
    val resultDF = model.transform(test)
    assert(resultDF.count == groundTruth)
    assert(resultDF.columns.contains("predictLeaf"))
  }

  test("test predictionLeaf with empty column name") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "train_test_ratio" -> "0.5",
      "num_round" -> 5, "num_workers" -> numWorkers)
    val training = buildDataFrame(Classification.train)
    val test = buildDataFrame(Classification.test)
    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(training)
    model.setLeafPredictionCol("")
    val resultDF = model.transform(test)
    assert(!resultDF.columns.contains("predictLeaf"))
  }

  test("test predictionContrib") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "train_test_ratio" -> "0.5",
      "num_round" -> 5, "num_workers" -> numWorkers)
    val training = buildDataFrame(Classification.train)
    val test = buildDataFrame(Classification.test)
    val groundTruth = test.count()
    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(training)
    model.setContribPredictionCol("predictContrib")
    val resultDF = model.transform(buildDataFrame(Classification.test))
    assert(resultDF.count == groundTruth)
    assert(resultDF.columns.contains("predictContrib"))
  }

  test("test predictionContrib with empty column name") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "train_test_ratio" -> "0.5",
      "num_round" -> 5, "num_workers" -> numWorkers)
    val training = buildDataFrame(Classification.train)
    val test = buildDataFrame(Classification.test)
    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(training)
    model.setContribPredictionCol("")
    val resultDF = model.transform(test)
    assert(!resultDF.columns.contains("predictContrib"))
  }

  test("test predictionLeaf and predictionContrib") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic", "train_test_ratio" -> "0.5",
      "num_round" -> 5, "num_workers" -> numWorkers)
    val training = buildDataFrame(Classification.train)
    val test = buildDataFrame(Classification.test)
    val groundTruth = test.count()
    val xgb = new XGBoostClassifier(paramMap)
    val model = xgb.fit(training)
    model.setLeafPredictionCol("predictLeaf")
    model.setContribPredictionCol("predictContrib")
    val resultDF = model.transform(buildDataFrame(Classification.test))
    assert(resultDF.count == groundTruth)
    assert(resultDF.columns.contains("predictLeaf"))
    assert(resultDF.columns.contains("predictContrib"))
  }
}
