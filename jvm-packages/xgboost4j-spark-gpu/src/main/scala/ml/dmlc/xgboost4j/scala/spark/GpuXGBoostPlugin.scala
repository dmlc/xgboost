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

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.ColumnarRdd
import org.apache.spark.ml.param.Param
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, Dataset}

import ml.dmlc.xgboost4j.java.{CudfColumnBatch, GpuColumnBatch}
import ml.dmlc.xgboost4j.scala.QuantileDMatrix
import ml.dmlc.xgboost4j.scala.spark.params.HasGroupCol

/**
 * GpuXGBoostPlugin is the XGBoost plugin which leverage spark-rapids
 * to accelerate the XGBoost from ETL to train.
 */
class GpuXGBoostPlugin extends XGBoostPlugin {

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

  // TODO, support numeric type
  private[spark] def preprocess[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]](
      estimator: XGBoostEstimator[T, M], dataset: Dataset[_]): Dataset[_] = {

    // Columns to be selected for XGBoost training
    val selectedCols: ArrayBuffer[Column] = ArrayBuffer.empty
    val schema = dataset.schema

    def selectCol(c: Param[String]) = {
      // TODO support numeric types
      if (estimator.isDefinedNonEmpty(c)) {
        selectedCols.append(estimator.castToFloatIfNeeded(schema, estimator.getOrDefault(c)))
      }
    }

    Seq(estimator.labelCol, estimator.weightCol, estimator.baseMarginCol).foreach(selectCol)
    estimator match {
      case p: HasGroupCol => selectCol(p.groupCol)
      case _ =>
    }

    // TODO support array/vector feature
    estimator.getFeaturesCols.foreach { name =>
      val col = estimator.castToFloatIfNeeded(dataset.schema, name)
      selectedCols.append(col)
    }
    val input = dataset.select(selectedCols: _*)
    estimator.repartitionIfNeeded(input)
  }

  // visiable for testing
  private[spark] def validate[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]](
      estimator: XGBoostEstimator[T, M],
      dataset: Dataset[_]): Unit = {
    require(estimator.getTreeMethod == "gpu_hist" || estimator.getDevice != "cpu",
      "Using Spark-Rapids to accelerate XGBoost must set device=cuda")
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

    validate(estimator, dataset)

    val train = preprocess(estimator, dataset)
    val schema = train.schema

    val indices = estimator.buildColumnIndices(schema)

    val maxBin = estimator.getMaxBins
    val nthread = estimator.getNthread
    val missing = estimator.getMissing

    /** build QuantilDMatrix on the executor side */
    def buildQuantileDMatrix(iter: Iterator[Table]): QuantileDMatrix = {
      val colBatchIter = iter.map { table =>
        withResource(new GpuColumnBatch(table, null)) { batch =>
          new CudfColumnBatch(
            batch.slice(indices.featureIds.get.map(Integer.valueOf).asJava),
            batch.slice(indices.labelId),
            batch.slice(indices.weightId.getOrElse(-1)),
            batch.slice(indices.marginId.getOrElse(-1)));
        }
      }
      new QuantileDMatrix(colBatchIter, missing, maxBin, nthread)
    }

    estimator.getEvalDataset().map { evalDs =>
      val evalProcessed = preprocess(estimator, evalDs)
      ColumnarRdd(train.toDF()).zipPartitions(ColumnarRdd(evalProcessed.toDF())) {
        (trainIter, evalIter) =>
          val trainDM = buildQuantileDMatrix(trainIter)
          val evalDM = buildQuantileDMatrix(evalIter)
          Iterator.single(new Watches(Array(trainDM, evalDM),
            Array(Utils.TRAIN_NAME, Utils.VALIDATION_NAME), None))
      }
    }.getOrElse(
      ColumnarRdd(train.toDF()).mapPartitions { iter =>
        val dm = buildQuantileDMatrix(iter)
        Iterator.single(new Watches(Array(dm), Array(Utils.TRAIN_NAME), None))
      }
    )
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
