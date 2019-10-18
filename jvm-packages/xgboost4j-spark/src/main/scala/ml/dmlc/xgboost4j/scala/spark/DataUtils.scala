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

import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}

import org.apache.spark.HashPartitioner
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.Param
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{FloatType, IntegerType}

object DataUtils extends Serializable {
  private[spark] implicit class XGBLabeledPointFeatures(
      val labeledPoint: XGBLabeledPoint
  ) extends AnyVal {
    /** Converts the point to [[MLLabeledPoint]]. */
    private[spark] def asML: MLLabeledPoint = {
      MLLabeledPoint(labeledPoint.label, labeledPoint.features)
    }

    /**
     * Returns feature of the point as [[org.apache.spark.ml.linalg.Vector]].
     *
     * If the point is sparse, the dimensionality of the resulting sparse
     * vector would be [[Int.MaxValue]]. This is the only safe value, since
     * XGBoost does not store the dimensionality explicitly.
     */
    def features: Vector = if (labeledPoint.indices == null) {
      Vectors.dense(labeledPoint.values.map(_.toDouble))
    } else {
      Vectors.sparse(Int.MaxValue, labeledPoint.indices, labeledPoint.values.map(_.toDouble))
    }
  }

  private[spark] implicit class MLLabeledPointToXGBLabeledPoint(
      val labeledPoint: MLLabeledPoint
  ) extends AnyVal {
    /** Converts an [[MLLabeledPoint]] to an [[XGBLabeledPoint]]. */
    def asXGB: XGBLabeledPoint = {
      labeledPoint.features.asXGB.copy(label = labeledPoint.label.toFloat)
    }
  }

  private[spark] implicit class MLVectorToXGBLabeledPoint(val v: Vector) extends AnyVal {
    /**
     * Converts a [[Vector]] to a data point with a dummy label.
     *
     * This is needed for constructing a [[ml.dmlc.xgboost4j.scala.DMatrix]]
     * for prediction.
     */
    def asXGB: XGBLabeledPoint = v match {
      case v: DenseVector =>
        XGBLabeledPoint(0.0f, null, v.values.map(_.toFloat))
      case v: SparseVector =>
        XGBLabeledPoint(0.0f, v.indices, v.values.map(_.toFloat))
    }
  }

  private def featureValueOfDenseVector(rowHashCode: Int, features: DenseVector): Float = {
    val featureId = {
      if (rowHashCode > 0) {
        rowHashCode % features.size
      } else {
        // prevent overflow
        math.abs(rowHashCode + 1) % features.size
      }
    }
    features.values(featureId).toFloat
  }

  private def featureValueOfSparseVector(rowHashCode: Int, features: SparseVector): Float = {
    val featureId = {
      if (rowHashCode > 0) {
        rowHashCode % features.indices.length
      } else {
        // prevent overflow
        math.abs(rowHashCode + 1) % features.indices.length
      }
    }
    features.values(featureId).toFloat
  }

  private def calculatePartitionKey(row: Row, numPartitions: Int): Int = {
    val Row(_, features: Vector, _, _) = row
    val rowHashCode = row.hashCode()
    val featureValue = features match {
      case denseVector: DenseVector =>
        featureValueOfDenseVector(rowHashCode, denseVector)
      case sparseVector: SparseVector =>
        featureValueOfSparseVector(rowHashCode, sparseVector)
    }
    math.abs((rowHashCode.toLong + featureValue).toString.hashCode % numPartitions)
  }

  private def attachPartitionKey(
      row: Row,
      deterministicPartition: Boolean,
      numWorkers: Int,
      xgbLp: XGBLabeledPoint): (Int, XGBLabeledPoint) = {
    if (deterministicPartition) {
      (calculatePartitionKey(row, numWorkers), xgbLp)
    } else {
      (1, xgbLp)
    }
  }

  private def repartitionRDDs(
      deterministicPartition: Boolean,
      numWorkers: Int,
      arrayOfRDDs: Array[RDD[(Int, XGBLabeledPoint)]]): Array[RDD[XGBLabeledPoint]] = {
    if (deterministicPartition) {
      arrayOfRDDs.map {rdd => rdd.partitionBy(new HashPartitioner(numWorkers))}.map {
        rdd => rdd.map(_._2)
      }
    } else {
      arrayOfRDDs.map(rdd => {
        if (rdd.getNumPartitions != numWorkers) {
          rdd.map(_._2).repartition(numWorkers)
        } else {
          rdd.map(_._2)
        }
      })
    }
  }

  private[spark] def convertDataFrameToXGBLabeledPointRDDs(
      labelCol: Column,
      featuresCol: Column,
      weight: Column,
      baseMargin: Column,
      group: Option[Column],
      numWorkers: Int,
      deterministicPartition: Boolean,
      dataFrames: DataFrame*): Array[RDD[XGBLabeledPoint]] = {
    val selectedColumns = group.map(groupCol => Seq(labelCol.cast(FloatType),
      featuresCol,
      weight.cast(FloatType),
      groupCol.cast(IntegerType),
      baseMargin.cast(FloatType))).getOrElse(Seq(labelCol.cast(FloatType),
      featuresCol,
      weight.cast(FloatType),
      baseMargin.cast(FloatType)))
    val arrayOfRDDs = dataFrames.toArray.map {
      df => df.select(selectedColumns: _*).rdd.map {
        case row @ Row(label: Float, features: Vector, weight: Float, group: Int,
          baseMargin: Float) =>
          val (indices, values) = features match {
            case v: SparseVector => (v.indices, v.values.map(_.toFloat))
            case v: DenseVector => (null, v.values.map(_.toFloat))
          }
          val xgbLp = XGBLabeledPoint(label, indices, values, weight, group, baseMargin)
          attachPartitionKey(row, deterministicPartition, numWorkers, xgbLp)
        case row @ Row(label: Float, features: Vector, weight: Float, baseMargin: Float) =>
          val (indices, values) = features match {
            case v: SparseVector => (v.indices, v.values.map(_.toFloat))
            case v: DenseVector => (null, v.values.map(_.toFloat))
          }
          val xgbLp = XGBLabeledPoint(label, indices, values, weight, baseMargin = baseMargin)
          attachPartitionKey(row, deterministicPartition, numWorkers, xgbLp)
      }
    }
    repartitionRDDs(deterministicPartition, numWorkers, arrayOfRDDs)
  }

}
