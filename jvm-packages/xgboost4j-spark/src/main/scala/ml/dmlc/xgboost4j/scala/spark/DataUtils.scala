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

import scala.collection.JavaConverters._

import ml.dmlc.xgboost4j.LabeledPoint
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

object DataUtils extends Serializable {

  implicit def fromSparkPointsToXGBoostPointsJava(sps: Iterator[MLLabeledPoint])
    : java.util.Iterator[LabeledPoint] = {
    fromSparkPointsToXGBoostPoints(sps).asJava
  }

  implicit def fromSparkPointsToXGBoostPoints(sps: Iterator[MLLabeledPoint]):
      Iterator[LabeledPoint] = {
    for (p <- sps) yield {
      p.features match {
        case denseFeature: DenseVector =>
          LabeledPoint.fromDenseVector(p.label.toFloat, denseFeature.values.map(_.toFloat))
        case sparseFeature: SparseVector =>
          LabeledPoint.fromSparseVector(p.label.toFloat, sparseFeature.indices,
            sparseFeature.values.map(_.toFloat))
      }
    }
  }

  implicit def fromSparkVectorToXGBoostPointsJava(sps: Iterator[Vector])
    : java.util.Iterator[LabeledPoint] = {
    fromSparkVectorToXGBoostPoints(sps).asJava
  }

  implicit def fromSparkVectorToXGBoostPoints(sps: Iterator[Vector])
    : Iterator[LabeledPoint] = {
    for (p <- sps) yield {
      p match {
        case denseFeature: DenseVector =>
          LabeledPoint.fromDenseVector(0.0f, denseFeature.values.map(_.toFloat))
        case sparseFeature: SparseVector =>
          LabeledPoint.fromSparseVector(0.0f, sparseFeature.indices,
            sparseFeature.values.map(_.toFloat))
      }
    }
  }
}
