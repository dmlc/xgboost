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

import java.util.{Iterator => JIterator}

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

import ml.dmlc.xgboost4j.DataBatch
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint

private[spark] object DataUtils extends Serializable {

  private def fetchUpdateFromSparseVector(sparseFeature: SparseVector): (List[Int], List[Float]) = {
    (sparseFeature.indices.toList, sparseFeature.values.map(_.toFloat).toList)
  }

  private def fetchUpdateFromVector(feature: Vector) = feature match {
    case denseFeature: DenseVector =>
      fetchUpdateFromSparseVector(denseFeature.toSparse)
    case sparseFeature: SparseVector =>
      fetchUpdateFromSparseVector(sparseFeature)
  }

  def fromLabeledPointsToSparseMatrix(points: Iterator[LabeledPoint]): JIterator[DataBatch] = {
    // TODO: support weight
    var samplePos = 0
    // TODO: change hard value
    val loadingBatchSize = 100
    val rowOffset = new ListBuffer[Long]
    val label = new ListBuffer[Float]
    val featureIndices = new ListBuffer[Int]
    val featureValues = new ListBuffer[Float]
    val dataBatches = new ListBuffer[DataBatch]
    for (point <- points) {
      val (nonZeroIndices, nonZeroValues) = fetchUpdateFromVector(point.features)
      rowOffset(samplePos) = rowOffset.size
      label(samplePos) = point.label.toFloat
      for (i <- nonZeroIndices.indices) {
        featureIndices += nonZeroIndices(i)
        featureValues += nonZeroValues(i)
      }
      samplePos += 1
      if (samplePos % loadingBatchSize == 0) {
        // create a data batch
        dataBatches += new DataBatch(
          rowOffset.toArray.clone(),
          null, label.toArray.clone(), featureIndices.toArray.clone(),
          featureValues.toArray.clone())
        rowOffset.clear()
        label.clear()
        featureIndices.clear()
        featureValues.clear()
      }
    }
    dataBatches.iterator.asJava
  }
}
