/*
 Copyright (c) 2024-2025 by Contributors

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
import scala.jdk.CollectionConverters._

import ai.rapids.cudf.Table
import com.nvidia.spark.rapids.{ColumnarRdd, GpuColumnVectorUtils}
import org.apache.commons.logging.LogFactory
import org.apache.spark.TaskContext
import org.apache.spark.ml.param.Param
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection
import org.apache.spark.sql.types.{DataType, FloatType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

import ml.dmlc.xgboost4j.java.CudfColumnBatch
import ml.dmlc.xgboost4j.scala.{DMatrix, QuantileDMatrix}
import ml.dmlc.xgboost4j.scala.spark.Utils.withResource
import ml.dmlc.xgboost4j.scala.spark.params.HasGroupCol

/**
 * GpuXGBoostPlugin is the XGBoost plugin which leverages spark-rapids
 * to accelerate the XGBoost from ETL to train.
 */
class GpuXGBoostPlugin extends XGBoostPlugin {

  private val logger = LogFactory.getLog("XGBoostSparkGpuPlugin")

  /**
   * Whether the plugin is enabled or not, if not enabled, fallback
   * to the regular CPU pipeline
   *
   * @param dataset the input dataset
   * @return Boolean
   */
  override def isEnabled(dataset: Dataset[_]): Boolean = {
    val conf = dataset.sparkSession.conf
    val hasRapidsPlugin = conf.get("spark.plugins", "").split(",").contains(
      "com.nvidia.spark.SQLPlugin")
    val rapidsEnabled = try {
      conf.get("spark.rapids.sql.enabled").toBoolean
    } catch {
      // Rapids plugin has default "spark.rapids.sql.enabled" to true
      case _: NoSuchElementException => true
      case _: Throwable => false // Any exception will return false
    }
    hasRapidsPlugin && rapidsEnabled
  }

  // TODO, support numeric type
  private[spark] def preprocess[T <: XGBoostEstimator[T, M], M <: XGBoostModel[M]](
      estimator: XGBoostEstimator[T, M], dataset: Dataset[_]): Dataset[_] = {

    // Columns to be selected for XGBoost training
    val selectedCols: ArrayBuffer[Column] = ArrayBuffer.empty
    val schema = dataset.schema

    def selectCol(c: Param[String], targetType: DataType = FloatType) = {
      // TODO support numeric types
      if (estimator.isDefinedNonEmpty(c)) {
        selectedCols.append(estimator.castIfNeeded(schema, estimator.getOrDefault(c), targetType))
      }
    }

    Seq(estimator.labelCol, estimator.weightCol, estimator.baseMarginCol)
      .foreach(p => selectCol(p))
    estimator match {
      case p: HasGroupCol => selectCol(p.groupCol, IntegerType)
      case _ =>
    }

    // TODO support array/vector feature
    estimator.getFeaturesCols.foreach { name =>
      val col = estimator.castIfNeeded(dataset.schema, name)
      selectedCols.append(col)
    }
    val input = dataset.select(selectedCols.toArray: _*)
    val repartitioned = estimator.repartitionIfNeeded(input)
    estimator.sortPartitionIfNeeded(repartitioned)
  }

  // visible for testing
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

    /** build QuantileDMatrix on the executor side */
    def buildQuantileDMatrix(iter: Iterator[Table],
                             ref: Option[QuantileDMatrix] = None): QuantileDMatrix = {
      val colBatchIter = iter.map { table =>
        withResource(new GpuColumnBatch(table)) { batch =>
          new CudfColumnBatch(
            batch.select(indices.featureIds.get),
            batch.select(indices.labelId),
            batch.select(indices.weightId.getOrElse(-1)),
            batch.select(indices.marginId.getOrElse(-1)),
            batch.select(indices.groupId.getOrElse(-1)));
        }
      }
      ref.map(r => new QuantileDMatrix(colBatchIter, r, missing, maxBin, nthread)).getOrElse(
        new QuantileDMatrix(colBatchIter, missing, maxBin, nthread)
      )
    }

    estimator.getEvalDataset().map { evalDs =>
      val evalProcessed = preprocess(estimator, evalDs)
      ColumnarRdd(train.toDF()).zipPartitions(ColumnarRdd(evalProcessed.toDF())) {
        (trainIter, evalIter) =>
          new Iterator[Watches] {
            override def hasNext: Boolean = trainIter.hasNext
            override def next(): Watches = {
              val trainDM = buildQuantileDMatrix(trainIter)
              val evalDM = buildQuantileDMatrix(evalIter, Some(trainDM))
              new Watches(Array(trainDM, evalDM),
                Array(Utils.TRAIN_NAME, Utils.VALIDATION_NAME), None)
            }
          }
      }
    }.getOrElse(
      ColumnarRdd(train.toDF()).mapPartitions { iter =>
        new Iterator[Watches] {
          override def hasNext: Boolean = iter.hasNext
          override def next(): Watches = {
            val dm = buildQuantileDMatrix(iter)
            new Watches(Array(dm), Array(Utils.TRAIN_NAME), None)
          }
        }
      }
    )
  }

  override def transform[M <: XGBoostModel[M]](model: XGBoostModel[M],
                                               dataset: Dataset[_]): DataFrame = {
    val sc = dataset.sparkSession.sparkContext

    val (transformedSchema, pred) = model.preprocess(dataset)
    val bBooster = sc.broadcast(model.nativeBooster)
    val bOriginalSchema = sc.broadcast(dataset.schema)

    val featureIds = model.getFeaturesCols.distinct.map(dataset.schema.fieldIndex).toList
    val isLocal = sc.isLocal
    val missing = model.getMissing
    val nThread = model.getNthread

    val rdd = ColumnarRdd(dataset.asInstanceOf[DataFrame]).mapPartitions { tableIters =>
      // booster is visible for all spark tasks in the same executor
      val booster = bBooster.value
      val originalSchema = bOriginalSchema.value

      // UnsafeProjection is not serializable so do it on the executor side
      val toUnsafe = UnsafeProjection.create(originalSchema)

      if (!booster.deviceIsSet) {
        booster.deviceIsSet.synchronized {
          if (!booster.deviceIsSet) {
            booster.deviceIsSet = true
            val gpuId = if (!isLocal) XGBoost.getGPUAddrFromResources else 0
            booster.setParam("device", s"cuda:$gpuId")
            logger.info("GPU transform on GPU device: cuda:" + gpuId)
          }
        }
      }

      // Iterator on Row
      new Iterator[Row] {
        // Convert InternalRow to Row
        private val converter: InternalRow => Row = CatalystTypeConverters
          .createToScalaConverter(originalSchema)
          .asInstanceOf[InternalRow => Row]

        // GPU batches read in must be closed by the receiver
        @transient var currentBatch: ColumnarBatch = null

        // Iterator on Row
        var iter: Iterator[Row] = null

        TaskContext.get().addTaskCompletionListener[Unit](_ => {
          closeCurrentBatch() // close the last ColumnarBatch
        })

        private def closeCurrentBatch(): Unit = {
          if (currentBatch != null) {
            currentBatch.close()
            currentBatch = null
          }
        }

        def loadNextBatch(): Unit = {
          closeCurrentBatch()
          if (tableIters.hasNext) {
            val dataTypes = originalSchema.fields.map(x => x.dataType)
            iter = withResource(tableIters.next()) { table =>
              // Create DMatrix
              val featureTable = new GpuColumnBatch(table).select(featureIds)
              if (featureTable == null) {
                val msg = featureIds.mkString(",")
                throw new RuntimeException(s"Couldn't create feature table for the " +
                  s"feature indices $msg")
              }
              try {
                val cudfColumnBatch = new CudfColumnBatch(featureTable, null, null, null, null)
                val dm = new DMatrix(cudfColumnBatch, missing, nThread)
                if (dm == null) {
                  Iterator.empty
                } else {
                  try {
                    currentBatch = new ColumnarBatch(
                      GpuColumnVectorUtils.extractHostColumns(table, dataTypes),
                      table.getRowCount().toInt)
                    val rowIterator = currentBatch.rowIterator().asScala.map(toUnsafe)
                      .map(converter(_))
                    model.predictInternal(booster, dm, pred, rowIterator).toIterator
                  } finally {
                    dm.delete()
                  }
                }
              } finally {
                featureTable.close()
              }
            }
          } else {
            iter = null
          }
        }

        override def hasNext: Boolean = {
          val itHasNext = iter != null && iter.hasNext
          if (!itHasNext) { // Don't have extra Row for current ColumnarBatch
            loadNextBatch()
            iter != null && iter.hasNext
          } else {
            itHasNext
          }
        }

        override def next(): Row = {
          if (iter == null || !iter.hasNext) {
            loadNextBatch()
          }
          if (iter == null) {
            throw new NoSuchElementException()
          }
          iter.next()
        }
      }
    }
    bBooster.unpersist(false)
    bOriginalSchema.unpersist(false)

    val output = dataset.sparkSession.createDataFrame(rdd, transformedSchema)
    model.postTransform(output, pred).toDF()
  }
}

private class GpuColumnBatch(table: Table) extends AutoCloseable {

  def select(index: Int): Table = {
    select(Seq(index))
  }

  def select(indices: Seq[Int]): Table = {
    if (!indices.forall(index => index < table.getNumberOfColumns && index >= 0)) {
      return null;
    }
    new Table(indices.map(table.getColumn): _*)
  }

  override def close(): Unit = {
    if (Option(table).isDefined) {
      table.close()
    }
  }
}
