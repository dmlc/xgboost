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

import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector}
import org.apache.spark.mllib.regression.{LabeledPoint => SparkLabeledPoint}

import ml.dmlc.xgboost4j.LabeledPoint

object DataUtils extends Serializable {

  implicit def fromSparkToXGBoostLabeledPointsAsJava(
      sps: Iterator[SparkLabeledPoint]): java.util.Iterator[LabeledPoint] = {
    fromSparkToXGBoostLabeledPoints(sps).asJava
  }

  implicit def fromSparkToXGBoostLabeledPoints(sps: Iterator[SparkLabeledPoint]):
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
}
