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

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

class XGBoostDFSuite extends SharedSparkContext with Utils {

  private def loadRow(filePath: String): List[Row] = {
    val file = Source.fromFile(new File(filePath))
    val rowList = new ListBuffer[Row]
    for (rowLine <- file.getLines()) {
      rowList += fromSVMStringToRow(rowLine)
    }
    rowList.toList
  }

  private def buildTrainingDataframe(sparkContext: Option[SparkContext] = None):
      DataFrame = {
    val rowList = loadRow(getClass.getResource("/agaricus.txt.train").getFile)
    val rowRDD = sparkContext.getOrElse(sc).parallelize(rowList, numWorkers)
    val sparkSession = SparkSession.builder().appName("XGBoostDFSuite").getOrCreate()
    sparkSession.createDataFrame(rowRDD,
      StructType(Array(StructField("label", DoubleType, nullable = false),
        StructField("features", new VectorUDT, nullable = false))))
  }

  private def fromSVMStringToRow(line: String): Row = {
    val (label, sv) = fromSVMStringToLabelAndVector(line)
    Row(label, sv)
  }

  test("test consistency between training with dataframe and RDD") {
    val trainingDF = buildTrainingDataframe()
    val trainingRDD = buildTrainingRDD(sc)
    val paramMap = List("eta" -> "1", "max_depth" -> "6", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = false)
    val xgBoostModelWithRDD = XGBoost.trainWithRDD(trainingRDD, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = false)
    val eval = new EvalError()
    val testSet = loadLabelPoints(getClass.getResource("/agaricus.txt.test").getFile).iterator
    import DataUtils._
    val testSetDMatrix = new DMatrix(new JDMatrix(testSet, null))
    assert(
      eval.eval(xgBoostModelWithDF.booster.predict(testSetDMatrix, outPutMargin = true),
        testSetDMatrix) ===
        eval.eval(xgBoostModelWithRDD.booster.predict(testSetDMatrix, outPutMargin = true),
          testSetDMatrix))
  }

  test("test transform of dataframe-based model") {
    val trainingDF = buildTrainingDataframe()
    val paramMap = List("eta" -> "1", "max_depth" -> "6", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = false)
    val testSet = loadLabelPoints(getClass.getResource("/agaricus.txt.test").getFile)
    val testRowsRDD = sc.parallelize(testSet.zipWithIndex, numWorkers).map{
      case (instance: LabeledPoint, id: Int) =>
        Row(id, instance.features, instance.label)
    }
    val testDF = trainingDF.sparkSession.createDataFrame(testRowsRDD, StructType(
      Array(StructField("id", IntegerType),
        StructField("features", new VectorUDT), StructField("label", DoubleType))))
    xgBoostModelWithDF.transform(testDF).show()
  }

  test("test order preservation of dataframe-based model") {
    val paramMap = List("eta" -> "1", "max_depth" -> "6", "silent" -> "0",
      "objective" -> "binary:logistic").toMap
    val trainingItr = loadLabelPoints(getClass.getResource("/agaricus.txt.train").getFile).
      iterator
    val (testItr, auxTestItr) =
      loadLabelPoints(getClass.getResource("/agaricus.txt.test").getFile).iterator.duplicate
    import DataUtils._
    val trainDMatrix = new DMatrix(new JDMatrix(trainingItr, null))
    val testDMatrix = new DMatrix(new JDMatrix(testItr, null))
    val xgboostModel = ScalaXGBoost.train(trainDMatrix, paramMap, 5)
    val predResultFromSeq = xgboostModel.predict(testDMatrix)
    val testRowsRDD = sc.parallelize(
      auxTestItr.toList.zipWithIndex, numWorkers).map {
      case (instance: LabeledPoint, id: Int) =>
        Row(id, instance.features, instance.label)
    }
    val trainingDF = buildTrainingDataframe()
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = false)
    val testDF = trainingDF.sqlContext.createDataFrame(testRowsRDD, StructType(
      Array(StructField("id", IntegerType), StructField("features", new VectorUDT),
        StructField("label", DoubleType))))
    val predResultsFromDF =
      xgBoostModelWithDF.transform(testDF).collect().map(row => (row.getAs[Int]("id"),
        row.getAs[mutable.WrappedArray[Float]]("prediction"))).toMap
    for (i <- predResultFromSeq.indices) {
      assert(predResultFromSeq(i).length === predResultsFromDF(i).length)
      for (j <- predResultFromSeq(i).indices) {
        assert(predResultFromSeq(i)(j) === predResultsFromDF(i)(j))
      }
    }
  }
}
