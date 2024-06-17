/*
 Copyright (c) 2024 by Contributors

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

import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.seqAsJavaListConverter

import com.nvidia.spark.rapids.ColumnarRdd
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, Dataset}

import ml.dmlc.xgboost4j.java.{CudfColumnBatch, GpuColumnBatch}
import ml.dmlc.xgboost4j.scala.QuantileDMatrix

private[spark] case class ColumnIndices(
  labelId: Int,
  featuresId: Seq[Int],
  weightId: Option[Int],
  marginId: Option[Int],
  groupId: Option[Int])

class GPUXGBoostPlugin extends XGBoostPlugin {

  /**
   * Whether the plugin is enabled or not, if not enabled, fallback
   * to the regular CPU pipeline
   *
   * @param dataset the input dataset
   * @return Boolean
   */
  override def isEnabled(dataset: Dataset[_]): Boolean = {
    val conf = dataset.sparkSession.conf
    val hasRapidsPlugin = conf.get("spark.sql.extensions", "").split(",").contains(
      "com.nvidia.spark.rapids.SQLExecPlugin")
    val rapidsEnabled = conf.get("spark.rapids.sql.enabled", "false").toBoolean
    hasRapidsPlugin && rapidsEnabled
  }

  /**
   * Convert Dataset to RDD[Watches] which will be fed into XGBoost
   *
   * @param estimator which estimator to be handled.
   * @param dataset   to be converted.
   * @return RDD[Watches]
   */
  override def buildRddWatches[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]](
    estimator: XGBoostEstimator[T, M],
    dataset: Dataset[_]): RDD[Watches] = {
    println("buildRddWatches ---")

    // TODO, check if the feature in featuresCols is numeric.

    val features = estimator.getFeaturesCols
    val maxBin = estimator.getMaxBins
    val nthread = estimator.getNthread
    // TODO cast features to float if possible

    val label = estimator.getLabelCol
    val missing = Float.NaN

    val selectedCols: ArrayBuffer[Column] = ArrayBuffer.empty
    (features.toSeq ++ Seq(estimator.getLabelCol)).foreach {name =>
      val col = estimator.castToFloatIfNeeded(dataset.schema, name)
      selectedCols.append(col)
    }
    var input = dataset.select(selectedCols: _*)
    input = input.repartition(estimator.getNumWorkers)

    val schema = input.schema
    val indices = ColumnIndices(
      schema.fieldIndex(label),
      features.map(schema.fieldIndex),
      None, None, None
    )

    ColumnarRdd(input).mapPartitions { iter =>
      val colBatchIter = iter.map { table =>
        withResource(new GpuColumnBatch(table, null)) { batch =>
          new CudfColumnBatch(
            batch.slice(indices.featuresId.map(Integer.valueOf).asJava),
            batch.slice(indices.labelId),
            batch.slice(indices.weightId.getOrElse(-1)),
            batch.slice(indices.marginId.getOrElse(-1)));
        }
      }

      val dm = new QuantileDMatrix(colBatchIter, missing, maxBin, nthread)
      Iterator.single(new Watches(Array(dm), Array(Utils.TRAIN_NAME), None))
    }
  }

  /** Executes the provided code block and then closes the resource */
  def withResource[T <: AutoCloseable, V](r: T)(block: T => V): V = {
    try {
      block(r)
    } finally {
      r.close()
    }
  }

}
