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

import java.io.File

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.Random

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._

class XGBoostDFSuite extends SharedSparkContext with Utils {

  private var trainingDF: DataFrame = null

  after {
    cleanExternalCache("XGBoostDFSuite")
  }

  private def buildTrainingDataframe(sparkContext: Option[SparkContext] = None): DataFrame = {
    if (trainingDF == null) {
      val rowList = loadLabelPoints(getClass.getResource("/agaricus.txt.train").getFile)
      val labeledPointsRDD = sparkContext.getOrElse(sc).parallelize(rowList, numWorkers)
      val sparkSession = SparkSession.builder().appName("XGBoostDFSuite").getOrCreate()
      import sparkSession.implicits._
      trainingDF = sparkSession.createDataset(labeledPointsRDD).toDF
    }
    trainingDF
  }

  test("test consistency and order preservation of dataframe-based model") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic")
    val trainingItr = loadLabelPoints(getClass.getResource("/agaricus.txt.train").getFile).
      iterator
    val (testItr, auxTestItr) =
      loadLabelPoints(getClass.getResource("/agaricus.txt.test").getFile).iterator.duplicate
    import DataUtils._
    val round = 5
    val trainDMatrix = new DMatrix(new JDMatrix(trainingItr, null))
    val testDMatrix = new DMatrix(new JDMatrix(testItr, null))
    val xgboostModel = ScalaXGBoost.train(trainDMatrix, paramMap, round)
    val predResultFromSeq = xgboostModel.predict(testDMatrix)
    val testSetItr = auxTestItr.zipWithIndex.map {
      case (instance: LabeledPoint, id: Int) => (id, instance.features, instance.label)
    }
    val trainingDF = buildTrainingDataframe()
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = round, nWorkers = numWorkers)
    val testDF = trainingDF.sparkSession.createDataFrame(testSetItr.toList).toDF(
      "id", "features", "label")
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
    val testItr = loadLabelPoints(getClass.getResource("/agaricus.txt.test").getFile).iterator
    val trainingDF = buildTrainingDataframe()
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers)
    val testSetItr = testItr.zipWithIndex.map {
      case (instance: LabeledPoint, id: Int) =>
        (id, instance.features, instance.label)
    }
    val testDF = trainingDF.sparkSession.createDataFrame(testSetItr.toList).toDF(
      "id", "features", "label")
    xgBoostModelWithDF.transformLeaf(testDF).show()
  }

  test("test schema of XGBoostRegressionModel") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "reg:linear")
    val testItr = loadLabelPoints(getClass.getResource("/machine.txt.test").getFile,
      zeroBased = true).iterator.
      zipWithIndex.map { case (instance: LabeledPoint, id: Int) =>
      (id, instance.features, instance.label)
    }
    val trainingDF = {
      val rowList = loadLabelPoints(getClass.getResource("/machine.txt.train").getFile,
        zeroBased = true)
      val labeledPointsRDD = sc.parallelize(rowList, numWorkers)
      val sparkSession = SparkSession.builder().appName("XGBoostDFSuite").getOrCreate()
      import sparkSession.implicits._
      sparkSession.createDataset(labeledPointsRDD).toDF
    }
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = true)
    xgBoostModelWithDF.setPredictionCol("final_prediction")
    val testDF = trainingDF.sparkSession.createDataFrame(testItr.toList).toDF(
      "id", "features", "label")
    val predictionDF = xgBoostModelWithDF.setExternalMemory(true).transform(testDF)
    assert(predictionDF.columns.contains("id") === true)
    assert(predictionDF.columns.contains("features") === true)
    assert(predictionDF.columns.contains("label") === true)
    assert(predictionDF.columns.contains("final_prediction") === true)
    predictionDF.show()
  }

  test("test schema of XGBoostClassificationModel") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "binary:logistic")
    val testItr = loadLabelPoints(getClass.getResource("/agaricus.txt.test").getFile).iterator.
      zipWithIndex.map { case (instance: LabeledPoint, id: Int) =>
      (id, instance.features, instance.label)
    }
    val trainingDF = buildTrainingDataframe()
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = true)
    xgBoostModelWithDF.asInstanceOf[XGBoostClassificationModel].setRawPredictionCol(
      "raw_prediction").setPredictionCol("final_prediction")
    val testDF = trainingDF.sparkSession.createDataFrame(testItr.toList).toDF(
      "id", "features", "label")
    var predictionDF = xgBoostModelWithDF.setExternalMemory(true).transform(testDF)
    assert(predictionDF.columns.contains("id") === true)
    assert(predictionDF.columns.contains("features") === true)
    assert(predictionDF.columns.contains("label") === true)
    assert(predictionDF.columns.contains("raw_prediction") === true)
    assert(predictionDF.columns.contains("final_prediction") === true)
    xgBoostModelWithDF.asInstanceOf[XGBoostClassificationModel].setRawPredictionCol("").
      setPredictionCol("final_prediction")
    predictionDF = xgBoostModelWithDF.transform(testDF)
    assert(predictionDF.columns.contains("id") === true)
    assert(predictionDF.columns.contains("features") === true)
    assert(predictionDF.columns.contains("label") === true)
    assert(predictionDF.columns.contains("raw_prediction") === false)
    assert(predictionDF.columns.contains("final_prediction") === true)
    xgBoostModelWithDF.asInstanceOf[XGBoostClassificationModel].
      setRawPredictionCol("raw_prediction").setPredictionCol("")
    predictionDF = xgBoostModelWithDF.transform(testDF)
    assert(predictionDF.columns.contains("id") === true)
    assert(predictionDF.columns.contains("features") === true)
    assert(predictionDF.columns.contains("label") === true)
    assert(predictionDF.columns.contains("raw_prediction") === true)
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

  test("fast histogram algorithm parameters are exposed correctly") {
    val paramMap = Map("eta" -> "1", "gamma" -> "0.5", "max_depth" -> "0", "silent" -> "0",
      "objective" -> "binary:logistic", "tree_method" -> "hist",
      "grow_policy" -> "depthwise", "max_depth" -> "2", "max_bin" -> "2",
      "eval_metric" -> "error")
    val testItr = loadLabelPoints(getClass.getResource("/agaricus.txt.test").getFile).iterator
    val trainingDF = buildTrainingDataframe()
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 10, nWorkers = math.min(2, numWorkers))
    val error = new EvalError
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testItr, null))
    assert(error.eval(xgBoostModelWithDF.booster.predict(testSetDMatrix, outPutMargin = true),
      testSetDMatrix) < 0.1)
  }

  private def convertCSVPointToLabelPoint(valueArray: Array[String]): LabeledPoint = {
    val intValueArray = new Array[Double](valueArray.length)
    intValueArray(valueArray.length - 2) = {
      if (valueArray(valueArray.length - 2) == "?") {
        1
      } else {
        0
      }
    }
    intValueArray(valueArray.length - 1) = valueArray(valueArray.length - 1).toDouble - 1
    for (i <- 0 until intValueArray.length - 2) {
      intValueArray(i) = valueArray(i).toDouble
    }
    LabeledPoint(intValueArray.last, new DenseVector(intValueArray.take(intValueArray.length - 1)))
  }

  private def loadCSVPoints(filePath: String, zeroBased: Boolean = false): Seq[LabeledPoint] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[LabeledPoint]
    for (sample <- file.getLines()) {
      sampleList += convertCSVPointToLabelPoint(sample.split(","))
    }
    sampleList
  }

  test("multi_class classification test") {
    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6")
    val trainingSet = loadCSVPoints(getClass.getResource("/dermatology.data").getFile)
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    XGBoost.trainWithDataFrame(trainingSet.toDF(), paramMap, round = 5, nWorkers = numWorkers)
  }

  test("test DF use nested groupData") {
    val testItr = loadLabelPoints(getClass.getResource("/rank-demo.txt.test").getFile).iterator.
      zipWithIndex.map { case (instance: LabeledPoint, id: Int) =>
      (id, instance.features, instance.label)
    }
    val trainingDF = {
      val rowList0 = loadLabelPoints(getClass.getResource("/rank-demo-0.txt.train").getFile)
      val labeledPointsRDD0 = sc.parallelize(rowList0, numSlices = 1)
      val rowList1 = loadLabelPoints(getClass.getResource("/rank-demo-1.txt.train").getFile)
      val labeledPointsRDD1 = sc.parallelize(rowList1, numSlices = 1)
      val labeledPointsRDD = labeledPointsRDD0.union(labeledPointsRDD1)
      val sparkSession = SparkSession.builder().appName("XGBoostDFSuite").getOrCreate()
      import sparkSession.implicits._
      sparkSession.createDataset(labeledPointsRDD).toDF
    }
    val trainGroupData0: Seq[Int] = Source.fromFile(
      getClass.getResource("/rank-demo-0.txt.train.group").getFile).getLines().map(_.toInt).toList
    val trainGroupData1: Seq[Int] = Source.fromFile(
      getClass.getResource("/rank-demo-1.txt.train.group").getFile).getLines().map(_.toInt).toList
    val trainGroupData: Seq[Seq[Int]] = Seq(trainGroupData0, trainGroupData1)
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "rank:pairwise", "groupData" -> trainGroupData)

    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = 2)
    val testDF = trainingDF.sparkSession.createDataFrame(testItr.toList).toDF(
      "id", "features", "label")
    val predResultsFromDF = xgBoostModelWithDF.setExternalMemory(true).transform(testDF).
      collect().map(row => (row.getAs[Int]("id"), row.getAs[DenseVector]("features"))).toMap
    assert(testDF.count() === predResultsFromDF.size)
  }

  test("params of estimator and produced model are coordinated correctly") {
    val paramMap = Map("eta" -> "0.1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "multi:softmax", "num_class" -> "6")
    val trainingSet = loadCSVPoints(getClass.getResource("/dermatology.data").getFile)
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    val model =
      XGBoost.trainWithDataFrame(trainingSet.toDF(), paramMap, round = 5, nWorkers = numWorkers)
    assert(model.get[Double](model.eta).get == 0.1)
    assert(model.get[Int](model.maxDepth).get == 6)
  }
}
