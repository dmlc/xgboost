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

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, VectorUDT}
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

class XGBoostDFSuite extends SharedSparkContext with Utils {

  private var trainingDF: DataFrame = null

  private def buildTrainingDataframe(sparkContext: Option[SparkContext] = None): DataFrame = {
    if (trainingDF == null) {
      val rowList = loadLabelAndVector(getClass.getResource("/agaricus.txt.train").getFile)
      val sparkSession = SparkSession.builder().appName("XGBoostDFSuite").getOrCreate()
      trainingDF = sparkSession.createDataFrame(rowList).toDF("label", "features")
    }
    trainingDF
  }

  test("test consistency and order preservation of dataframe-based model") {
    val paramMap = List("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
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
    val testSetItr = auxTestItr.zipWithIndex.map {
      case (instance: LabeledPoint, id: Int) =>
        (id, instance.features, instance.label)
    }
    val trainingDF = buildTrainingDataframe()
    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(trainingDF, paramMap,
      round = 5, nWorkers = numWorkers, useExternalMemory = false)
    val testDF = trainingDF.sparkSession.createDataFrame(testSetItr.toList).toDF(
      "id", "features", "label")
    val predResultsFromDF = xgBoostModelWithDF.setExternalMemory(true).transform(testDF).
      collect().map(row =>
      (row.getAs[Int]("id"), row.getAs[mutable.WrappedArray[Float]]("rawPrediction"))
    ).toMap
    for (i <- predResultFromSeq.indices) {
      assert(predResultFromSeq(i).length === predResultsFromDF(i).length)
      for (j <- predResultFromSeq(i).indices) {
        assert(predResultFromSeq(i)(j) === predResultsFromDF(i)(j))
      }
    }
    cleanExternalCache("XGBoostDFSuite")
  }
}
