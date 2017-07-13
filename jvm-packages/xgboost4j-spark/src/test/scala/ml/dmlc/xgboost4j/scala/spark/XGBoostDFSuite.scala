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

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}

import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql._

class XGBoostDFSuite extends SharedSparkContext with Utils {

  after {
    cleanExternalCache("XGBoostDFSuite")
  }

  private def buildTestDataFrame(test: Seq[MLLabeledPoint]): DataFrame = {
    val it = test.iterator.zipWithIndex.map {
      case (instance: MLLabeledPoint, id: Int) =>
        (id, instance.features, instance.label)
    }

    sparkSession.createDataFrame(it.toList).toDF("id", "features", "label")
  }

  private def buildTrainDataFrame(
      train: Seq[MLLabeledPoint],
      numPartitions: Int = numWorkers
  ): DataFrame = {
    val ss = sparkSession
    import ss.implicits._
    sparkSession.createDataset(sc.parallelize(train, numPartitions)).toDF
  }

  test("test consistency and order preservation of dataframe-based model") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic")
    val trainingItr = Classification.train.iterator
    val testItr = Classification.test.iterator
    import DataUtils._
    val round = 5
    val trainDMatrix = new DMatrix(new JDMatrix(trainingItr, null))
    val testDMatrix = new DMatrix(new JDMatrix(testItr, null))
    val xgboostModel = ScalaXGBoost.train(trainDMatrix, paramMap, round)
    val predResultFromSeq = xgboostModel.predict(testDMatrix)
    val trainingDF = buildTrainDataFrame(Classification.train)
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = round, nWorkers = numWorkers)
    val testDF = buildTestDataFrame(Classification.test)
    val predResultsFromDF = xgBoostModelWithDF.setExternalMemory(true).transform(testDF).
      collect().map(row => (row.getAs[Int]("id"), row.getAs[DenseVector]("probabilities"))).toMap
    assert(testDF.count() === predResultsFromDF.size)
    // the vector length in probabilties column is 2 since we have to fit to the evaluator in
    // Spark
    for (i <- predResultFromSeq.indices) {
      assert(predResultFromSeq(i).length === predResultsFromDF(i).values.length - 1)
      for (j <- predResultFromSeq(i).indices) {
        assert(predResultFromSeq(i)(j) === predResultsFromDF(i)(j + 1))
      }
    }
  }

  test("test transformLeaf") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic")
    val trainingDF = buildTrainDataFrame(Classification.train)
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers)
    val testDF = buildTestDataFrame(Classification.test)
    xgBoostModelWithDF.transformLeaf(testDF).show()
  }

  test("test schema of XGBoostRegressionModel") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "reg:linear")
    val trainingDF = buildTrainDataFrame(Regression.train)
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = true)
    xgBoostModelWithDF.setPredictionCol("final_prediction")
    val testDF = buildTestDataFrame(Regression.test)
    val predictionDF = xgBoostModelWithDF.setExternalMemory(true).transform(testDF)
    assert(predictionDF.columns.contains("id"))
    assert(predictionDF.columns.contains("features"))
    assert(predictionDF.columns.contains("label"))
    assert(predictionDF.columns.contains("final_prediction"))
    predictionDF.show()
  }

  test("test schema of XGBoostClassificationModel") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic")
    val trainingDF = buildTrainDataFrame(Classification.train)
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = true)
    xgBoostModelWithDF.asInstanceOf[XGBoostClassificationModel].setRawPredictionCol(
      "raw_prediction").setPredictionCol("final_prediction")
    val testDF = buildTestDataFrame(Classification.test)
    var predictionDF = xgBoostModelWithDF.setExternalMemory(true).transform(testDF)
    assert(predictionDF.columns.contains("id"))
    assert(predictionDF.columns.contains("features"))
    assert(predictionDF.columns.contains("label"))
    assert(predictionDF.columns.contains("raw_prediction"))
    assert(predictionDF.columns.contains("final_prediction"))
    xgBoostModelWithDF.asInstanceOf[XGBoostClassificationModel].setRawPredictionCol("").
      setPredictionCol("final_prediction")
    predictionDF = xgBoostModelWithDF.transform(testDF)
    assert(predictionDF.columns.contains("id"))
    assert(predictionDF.columns.contains("features"))
    assert(predictionDF.columns.contains("label"))
    assert(predictionDF.columns.contains("raw_prediction") === false)
    assert(predictionDF.columns.contains("final_prediction"))
    xgBoostModelWithDF.asInstanceOf[XGBoostClassificationModel].
      setRawPredictionCol("raw_prediction").setPredictionCol("")
    predictionDF = xgBoostModelWithDF.transform(testDF)
    assert(predictionDF.columns.contains("id"))
    assert(predictionDF.columns.contains("features"))
    assert(predictionDF.columns.contains("label"))
    assert(predictionDF.columns.contains("raw_prediction"))
    assert(predictionDF.columns.contains("final_prediction") === false)
  }

  test("xgboost and spark parameters synchronize correctly") {
    val xgbParamMap = Map("eta" -> "1", "objective" -> "binary:logistic")
    // from xgboost params to spark params
    val xgbEstimator = new XGBoostEstimator(xgbParamMap)
    assert(xgbEstimator.get(xgbEstimator.eta).get === 1.0)
    assert(xgbEstimator.get(xgbEstimator.objective).get === "binary:logistic")
    // from spark to xgboost params
    val xgbEstimatorCopy = xgbEstimator.copy(ParamMap.empty)
    assert(xgbEstimatorCopy.fromParamsToXGBParamMap("eta").toString.toDouble === 1.0)
    assert(xgbEstimatorCopy.fromParamsToXGBParamMap("objective").toString === "binary:logistic")
  }

  test("eval_metric is configured correctly") {
    val xgbParamMap = Map("eta" -> "1", "objective" -> "binary:logistic")
    val xgbEstimator = new XGBoostEstimator(xgbParamMap)
    assert(xgbEstimator.get(xgbEstimator.evalMetric).get === "error")
    val sparkParamMap = ParamMap.empty
    val xgbEstimatorCopy = xgbEstimator.copy(sparkParamMap)
    assert(xgbEstimatorCopy.fromParamsToXGBParamMap("eval_metric") === "error")
    val xgbEstimatorCopy1 = xgbEstimator.copy(sparkParamMap.put(xgbEstimator.evalMetric, "logloss"))
    assert(xgbEstimatorCopy1.fromParamsToXGBParamMap("eval_metric") === "logloss")
  }

  ignore("fast histogram algorithm parameters are exposed correctly") {
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "max_depth" -> "2", "max_bin" -> "2",
      "eval_metric" -> "error")
    val testItr = Classification.test.iterator
    val trainingDF = buildTrainDataFrame(Classification.train)
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 10, nWorkers = math.min(2, numWorkers))
    val error = new EvalError
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testItr, null))
    assert(error.eval(xgBoostModelWithDF.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
  }

  test("multi_class classification test") {
    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6")
    val trainingDF = buildTrainDataFrame(MultiClassification.train)
    XGBoost.trainWithDataFrame(trainingDF.toDF(), paramMap, round = 5, nWorkers = numWorkers)
  }

  test("test DF use nested groupData") {
    val trainingDF = buildTrainDataFrame(Ranking.train0, 1)
        .union(buildTrainDataFrame(Ranking.train1, 1))
    val trainGroupData: Seq[Seq[Int]] = Seq(Ranking.trainGroup0, Ranking.trainGroup1)
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "rank:pairwise", "groupData" -> trainGroupData)

    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = 2)
    val testDF = buildTestDataFrame(Ranking.test)
    val predResultsFromDF = xgBoostModelWithDF.setExternalMemory(true).transform(testDF).
      collect().map(row => (row.getAs[Int]("id"), row.getAs[DenseVector]("features"))).toMap
    assert(testDF.count() === predResultsFromDF.size)
  }

  test("params of estimator and produced model are coordinated correctly") {
    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6")
    val trainingDF = buildTrainDataFrame(MultiClassification.train)
    val model = XGBoost.trainWithDataFrame(trainingDF, paramMap, round = 5, nWorkers = numWorkers)
    assert(model.get[Double](model.eta).get == 0.1)
    assert(model.get[Int](model.maxDepth).get == 6)
  }
}
