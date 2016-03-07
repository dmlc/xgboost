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

package ml.dmlc.xgboost4j.scala.spark.example

import java.io.File

import scala.collection.mutable.ListBuffer
import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

import ml.dmlc.xgboost4j.scala.DMatrix
import ml.dmlc.xgboost4j.scala.spark.XGBoost


object DistTrainWithSpark {

  private def readFile(filePath: String): List[LabeledPoint] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[LabeledPoint]
    for (sample <- file.getLines()) {
      sampleList += fromSVMStringToLabeledPoint(sample)
    }
    sampleList.toList
  }

  private def fromSVMStringToLabeledPoint(line: String): LabeledPoint = {
    val labelAndFeatures = line.split(" ")
    val label = labelAndFeatures(0).toInt
    val features = labelAndFeatures.tail
    val denseFeature = new Array[Double](129)
    for (feature <- features) {
      val idAndValue = feature.split(":")
      denseFeature(idAndValue(0).toInt) = idAndValue(1).toDouble
    }
    LabeledPoint(label, new DenseVector(denseFeature))
  }

  def main(args: Array[String]): Unit = {
    import ml.dmlc.xgboost4j.scala.spark.DataUtils._
    if (args.length != 4) {
      println(
        "usage: program number_of_trainingset_partitions num_of_rounds training_path test_path")
      sys.exit(1)
    }
    val sc = new SparkContext()
    val inputTrainPath = args(2)
    val inputTestPath = args(3)
    val trainingLabeledPoints = readFile(inputTrainPath)
    val trainRDD = sc.parallelize(trainingLabeledPoints, args(0).toInt)
    val testLabeledPoints = readFile(inputTestPath).iterator
    val testMatrix = new DMatrix(testLabeledPoints, null)
    val booster = XGBoost.train(trainRDD,
      List("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
        "objective" -> "binary:logistic").toMap, args(1).toInt, null, null)
    booster.map(boosterInstance => boosterInstance.predict(testMatrix))
  }
}
