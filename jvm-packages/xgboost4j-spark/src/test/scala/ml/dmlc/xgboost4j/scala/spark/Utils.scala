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

import scala.collection.mutable.ListBuffer
import scala.io.Source

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, Vector => SparkVector}
import org.apache.spark.rdd.RDD

trait Utils extends Serializable {
  protected val numWorkers = Runtime.getRuntime().availableProcessors()

  protected var labeledPointsRDD: RDD[LabeledPoint] = null

  protected def cleanExternalCache(prefix: String): Unit = {
    val dir = new File(".")
    for (file <- dir.listFiles() if file.getName.startsWith(prefix)) {
      file.delete()
    }
  }

  protected def loadLabelPoints(filePath: String, zeroBased: Boolean = false):
      List[LabeledPoint] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[LabeledPoint]
    for (sample <- file.getLines()) {
      sampleList += fromColValueStringToLabeledPoint(sample, zeroBased)
    }
    sampleList.toList
  }

  protected def loadLabelAndVector(filePath: String, zeroBased: Boolean = false):
      List[(Double, SparkVector)] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[(Double, SparkVector)]
    for (sample <- file.getLines()) {
      sampleList += fromColValueStringToLabelAndVector(sample, zeroBased)
    }
    sampleList.toList
  }

  protected def fromColValueStringToLabelAndVector(line: String, zeroBased: Boolean):
      (Double, SparkVector) = {
    val labelAndFeatures = line.split(" ")
    val label = labelAndFeatures(0).toDouble
    val features = labelAndFeatures.tail
    val denseFeature = new Array[Double](126)
    for (feature <- features) {
      val idAndValue = feature.split(":")
      if (!zeroBased) {
        denseFeature(idAndValue(0).toInt - 1) = idAndValue(1).toDouble
      } else {
        denseFeature(idAndValue(0).toInt) = idAndValue(1).toDouble
      }
    }
    (label, new DenseVector(denseFeature))
  }

  protected def fromColValueStringToLabeledPoint(line: String, zeroBased: Boolean): LabeledPoint = {
    val (label, sv) = fromColValueStringToLabelAndVector(line, zeroBased)
    LabeledPoint(label, sv)
  }

  protected def buildTrainingRDD(sparkContext: SparkContext): RDD[LabeledPoint] = {
    if (labeledPointsRDD == null) {
      val sampleList = loadLabelPoints(getClass.getResource("/agaricus.txt.train").getFile)
      labeledPointsRDD = sparkContext.parallelize(sampleList, numWorkers)
    }
    labeledPointsRDD
  }
}
