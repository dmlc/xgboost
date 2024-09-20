/*
 Copyright (c) 2014-2024 by Contributors

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

import scala.io.Source
import scala.util.Random

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}

trait TrainTestData {
  protected def getResourceLines(resource: String): Iterator[String] = {
    require(resource.startsWith("/"), "resource must start with /")
    val is = getClass.getResourceAsStream(resource)
    if (is == null) {
      sys.error(s"failed to resolve resource $resource")
    }

    Source.fromInputStream(is).getLines()
  }

  protected def getLabeledPoints(resource: String, featureSize: Int,
                                 zeroBased: Boolean): Seq[XGBLabeledPoint] = {
    getResourceLines(resource).map { line =>
      val labelAndFeatures = line.split(" ")
      val label = labelAndFeatures.head.toFloat
      val values = new Array[Float](featureSize)
      for (feature <- labelAndFeatures.tail) {
        val idAndValue = feature.split(":")
        if (!zeroBased) {
          values(idAndValue(0).toInt - 1) = idAndValue(1).toFloat
        } else {
          values(idAndValue(0).toInt) = idAndValue(1).toFloat
        }
      }

      XGBLabeledPoint(label, featureSize, null, values)
    }.toList
  }

  protected def getLabeledPointsWithGroup(resource: String): Seq[XGBLabeledPoint] = {
    getResourceLines(resource).map { line =>
      val original = line.split(",")
      val length = original.length
      val label = original.head.toFloat
      val group = original.last.toInt
      val values = original.slice(1, length - 1).map(_.toFloat)
      XGBLabeledPoint(label, values.size, null, values, 1f, group, Float.NaN)
    }.toList
  }
}

object Classification extends TrainTestData {
  val train: Seq[XGBLabeledPoint] = getLabeledPoints("/agaricus.txt.train", 126, zeroBased = false)
  val test: Seq[XGBLabeledPoint] = getLabeledPoints("/agaricus.txt.test", 126, zeroBased = false)

  Random.setSeed(10)
  val randomWeights = Array.fill(train.length)(Random.nextFloat())
  val trainWithWeight = train.zipWithIndex.map { case (v, index) =>
    XGBLabeledPoint(v.label, v.size, v.indices, v.values,
      randomWeights(index), v.group, v.baseMargin)
  }
}

object MultiClassification extends TrainTestData {

  private def split(): (Seq[XGBLabeledPoint], Seq[XGBLabeledPoint]) = {
    val tmp: Seq[XGBLabeledPoint] = getLabeledPoints("/dermatology.data")
    Random.setSeed(100)
    val randomizedTmp = Random.shuffle(tmp)
    val splitIndex = (randomizedTmp.length * 0.8).toInt
    (randomizedTmp.take(splitIndex), randomizedTmp.drop(splitIndex))
  }

  val (train, test) = split()
  Random.setSeed(10)
  val randomWeights = Array.fill(train.length)(Random.nextFloat())
  val trainWithWeight = train.zipWithIndex.map { case (v, index) =>
    XGBLabeledPoint(v.label, v.size, v.indices, v.values,
      randomWeights(index), v.group, v.baseMargin)
  }

  private def getLabeledPoints(resource: String): Seq[XGBLabeledPoint] = {
    getResourceLines(resource).map { line =>
      val featuresAndLabel = line.split(",")
      val label = featuresAndLabel.last.toFloat - 1
      val values = new Array[Float](featuresAndLabel.length - 1)
      values(values.length - 1) =
        if (featuresAndLabel(featuresAndLabel.length - 2) == "?") 1 else 0
      for (i <- 0 until values.length - 2) {
        values(i) = featuresAndLabel(i).toFloat
      }

      XGBLabeledPoint(label, values.length - 1, null, values.take(values.length - 1))
    }.toList
  }
}

object Regression extends TrainTestData {
  val MACHINE_COL_NUM = 36
  val train: Seq[XGBLabeledPoint] = getLabeledPoints(
    "/machine.txt.train", MACHINE_COL_NUM, zeroBased = true)
  val test: Seq[XGBLabeledPoint] = getLabeledPoints(
    "/machine.txt.test", MACHINE_COL_NUM, zeroBased = true)

  Random.setSeed(10)
  val randomWeights = Array.fill(train.length)(Random.nextFloat())
  val trainWithWeight = train.zipWithIndex.map { case (v, index) =>
    XGBLabeledPoint(v.label, v.size, v.indices, v.values,
      randomWeights(index), v.group, v.baseMargin)
  }

  object Ranking extends TrainTestData {
    val RANK_COL_NUM = 3
    val train: Seq[XGBLabeledPoint] = getLabeledPointsWithGroup("/rank.train.csv")
    // use the group as the weight
    val trainWithWeight = train.map { labelPoint =>
      XGBLabeledPoint(labelPoint.label, labelPoint.size, labelPoint.indices, labelPoint.values,
        labelPoint.group, labelPoint.group, labelPoint.baseMargin)
    }
    val trainGroups = train.map(_.group)
    val test: Seq[XGBLabeledPoint] = getLabeledPoints(
      "/rank.test.txt", RANK_COL_NUM, zeroBased = false)
  }

}
