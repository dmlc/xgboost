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

import scala.collection.mutable
import scala.io.Source
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

  protected def getLabeledPoints(resource: String, zeroBased: Boolean): Seq[XGBLabeledPoint] = {
    getResourceLines(resource).map { line =>
      val labelAndFeatures = line.split(" ")
      val label = labelAndFeatures.head.toFloat
      val values = new Array[Float](126)
      for (feature <- labelAndFeatures.tail) {
        val idAndValue = feature.split(":")
        if (!zeroBased) {
          values(idAndValue(0).toInt - 1) = idAndValue(1).toFloat
        } else {
          values(idAndValue(0).toInt) = idAndValue(1).toFloat
        }
      }

      XGBLabeledPoint(label, null, values)
    }.toList
  }

  protected def getLabeledPointsWithGroup(resource: String): Seq[XGBLabeledPoint] = {
    getResourceLines(resource).map { line =>
      val original = line.split(",")
      val length = original.length
      val label = original.head.toFloat
      val group = original.last.toInt
      val values = original.slice(1, length - 1).map(_.toFloat)
      XGBLabeledPoint(label, null, values, 1f, group, Float.NaN)
    }.toList
  }
}

object Classification extends TrainTestData {
  val train: Seq[XGBLabeledPoint] = getLabeledPoints("/agaricus.txt.train", zeroBased = false)
  val test: Seq[XGBLabeledPoint] = getLabeledPoints("/agaricus.txt.test", zeroBased = false)
}

object MultiClassification extends TrainTestData {
  val train: Seq[XGBLabeledPoint] = getLabeledPoints("/dermatology.data")

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

      XGBLabeledPoint(label, null, values.take(values.length - 1))
    }.toList
  }
}

object Regression extends TrainTestData {
  val train: Seq[XGBLabeledPoint] = getLabeledPoints("/machine.txt.train", zeroBased = true)
  val test: Seq[XGBLabeledPoint] = getLabeledPoints("/machine.txt.test", zeroBased = true)
}

object Ranking extends TrainTestData {
  val train: Seq[XGBLabeledPoint] = getLabeledPointsWithGroup("/rank.train.csv")
  val test: Seq[XGBLabeledPoint] = getLabeledPoints("/rank.test.txt", zeroBased = false)

  private def getGroups(resource: String): Seq[Int] = {
    getResourceLines(resource).map(_.toInt).toList
  }
}

object Synthetic extends {
  val train: Seq[XGBLabeledPoint] = Seq(
    XGBLabeledPoint(1.0f, Array(0, 1), Array(1.0f, 2.0f)),
    XGBLabeledPoint(0.0f, Array(0, 1, 2), Array(1.0f, 2.0f, 3.0f)),
    XGBLabeledPoint(0.0f, Array(0, 1, 2), Array(1.0f, 2.0f, 3.0f)),
    XGBLabeledPoint(1.0f, Array(0, 1), Array(1.0f, 2.0f))
  )
}
