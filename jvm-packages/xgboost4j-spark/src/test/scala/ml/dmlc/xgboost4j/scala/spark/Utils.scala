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

import ml.dmlc.xgboost4j.java.XGBoostError
import ml.dmlc.xgboost4j.scala.{DMatrix, EvalTrait}
import org.apache.commons.logging.LogFactory
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Vector => SparkVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

trait Utils extends Serializable {
  protected val numWorkers = Runtime.getRuntime().availableProcessors()

  protected def loadLabelPoints(filePath: String): List[LabeledPoint] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[LabeledPoint]
    for (sample <- file.getLines()) {
      sampleList += fromSVMStringToLabeledPoint(sample)
    }
    sampleList.toList
  }

  protected def fromSVMStringToLabelAndVector(line: String): (Double, SparkVector) = {
    val labelAndFeatures = line.split(" ")
    val label = labelAndFeatures(0).toDouble
    val features = labelAndFeatures.tail
    val denseFeature = new Array[Double](129)
    for (feature <- features) {
      val idAndValue = feature.split(":")
      denseFeature(idAndValue(0).toInt) = idAndValue(1).toDouble
    }
    (label, new DenseVector(denseFeature))
  }

  protected def fromSVMStringToLabeledPoint(line: String): LabeledPoint = {
    val (label, sv) = fromSVMStringToLabelAndVector(line)
    LabeledPoint(label, sv)
  }

  protected def buildTrainingRDD(sparkContext: SparkContext): RDD[LabeledPoint] = {
    val sampleList = loadLabelPoints(getClass.getResource("/agaricus.txt.train").getFile)
    sparkContext.parallelize(sampleList, numWorkers)
  }
}
