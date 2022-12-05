/*
 Copyright (c) 2021-2022 by Contributors

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

package ml.dmlc.xgboost4j.scala.rapids.spark

import scala.collection.JavaConverters._

import ml.dmlc.xgboost4j.gpu.java.CudfColumnBatch
import ml.dmlc.xgboost4j.java.nvidia.spark.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, QuantileDMatrix}
import ml.dmlc.xgboost4j.scala.spark.params.XGBoostEstimatorCommon
import ml.dmlc.xgboost4j.scala.spark.{PreXGBoost, PreXGBoostProvider, Watches, XGBoost, XGBoostClassificationModel, XGBoostClassifier, XGBoostExecutionParams, XGBoostRegressionModel, XGBoostRegressor}
import org.apache.commons.logging.LogFactory

import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.{CatalystTypeConverters, InternalRow}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection
import org.apache.spark.sql.functions.{col, collect_list, struct}
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
 * GpuPreXGBoost brings Rapids-Plugin to XGBoost4j-Spark to accelerate XGBoost4j
 * training and transform process
 */
class GpuPreXGBoost extends PreXGBoostProvider {

  /**
   * Whether the provider is enabled or not
   *
   * @param dataset the input dataset
   * @return Boolean
   */
  override def providerEnabled(dataset: Option[Dataset[_]]): Boolean = {
    GpuPreXGBoost.providerEnabled(dataset)
  }

  /**
   * Convert the Dataset[_] to RDD[() => Watches] which will be fed to XGBoost
   *
   * @param estimator [[XGBoostClassifier]] or [[XGBoostRegressor]]
   * @param dataset   the training data
   * @param params    all user defined and defaulted params
   * @return [[XGBoostExecutionParams]] => (RDD[[() => Watches]], Option[ RDD[_] ])
   *         RDD[() => Watches] will be used as the training input
   *         Option[ RDD[_] ] is the optional cached RDD
   */
  override def buildDatasetToRDD(estimator: Estimator[_],
      dataset: Dataset[_],
      params: Map[String, Any]):
    XGBoostExecutionParams => (RDD[() => Watches], Option[RDD[_]]) = {
    GpuPreXGBoost.buildDatasetToRDD(estimator, dataset, params)
  }

  /**
   * Transform Dataset
   *
   * @param model   [[XGBoostClassificationModel]] or [[XGBoostRegressionModel]]
   * @param dataset the input Dataset to transform
   * @return the transformed DataFrame
   */
  override def transformDataset(model: Model[_], dataset: Dataset[_]): DataFrame = {
    GpuPreXGBoost.transformDataset(model, dataset)
  }

  override def transformSchema(
      xgboostEstimator: XGBoostEstimatorCommon,
      schema: StructType): StructType = {
    GpuPreXGBoost.transformSchema(xgboostEstimator, schema)
  }
}

class BoosterFlag extends Serializable {
  // indicate if the GPU parameters are set.
  var isGpuParamsSet = false
}

object GpuPreXGBoost extends PreXGBoostProvider {

  private val logger = LogFactory.getLog("XGBoostSpark")
  private val FEATURES_COLS = "features_cols"
  private val TRAIN_NAME = "train"

  override def providerEnabled(dataset: Option[Dataset[_]]): Boolean = {
    // RuntimeConfig
    val optionConf = dataset.map(ds => Some(ds.sparkSession.conf))
      .getOrElse(SparkSession.getActiveSession.map(ss => ss.conf))

    if (optionConf.isDefined) {
      val conf = optionConf.get
      val rapidsEnabled = try {
        conf.get("spark.rapids.sql.enabled").toBoolean
      } catch {
        // Rapids plugin has default "spark.rapids.sql.enabled" to true
        case _: NoSuchElementException => true
        case _: Throwable => false // Any exception will return false
      }
      rapidsEnabled && conf.get("spark.sql.extensions", "")
        .split(",")
        .contains("com.nvidia.spark.rapids.SQLExecPlugin")
    } else false
  }

  /**
   * Convert the Dataset[_] to RDD[() => Watches] which will be fed to XGBoost
   *
   * @param estimator supports XGBoostClassifier and XGBoostRegressor
   * @param dataset   the training data
   * @param params    all user defined and defaulted params
   * @return [[XGBoostExecutionParams]] => (RDD[[() => Watches]], Option[ RDD[_] ])
   *         RDD[() => Watches] will be used as the training input to build DMatrix
   *         Option[ RDD[_] ] is the optional cached RDD
   */
  override def buildDatasetToRDD(
      estimator: Estimator[_],
      dataset: Dataset[_],
      params: Map[String, Any]):
    XGBoostExecutionParams => (RDD[() => Watches], Option[RDD[_]]) = {

    val (Seq(labelName, weightName, marginName), feturesCols, groupName, evalSets) =
      estimator match {
        case est: XGBoostEstimatorCommon =>
          require(est.isDefined(est.treeMethod) && est.getTreeMethod.equals("gpu_hist"),
            s"GPU train requires tree_method set to gpu_hist")
          val groupName = estimator match {
            case regressor: XGBoostRegressor => if (regressor.isDefined(regressor.groupCol)) {
              regressor.getGroupCol } else ""
            case _: XGBoostClassifier => ""
            case _ => throw new RuntimeException("Unsupported estimator: " + estimator)
          }
          // Check schema and cast columns' type
          (GpuUtils.getColumnNames(est)(est.labelCol, est.weightCol, est.baseMarginCol),
            est.getFeaturesCols, groupName, est.getEvalSets(params))
        case _ => throw new RuntimeException("Unsupported estimator: " + estimator)
    }

    val castedDF = GpuUtils.prepareColumnType(dataset, feturesCols, labelName, weightName,
      marginName)

    // Check columns and build column data batch
    val trainingData = GpuUtils.buildColumnDataBatch(feturesCols,
      labelName, weightName, marginName, "", castedDF)

    // eval map
    val evalDataMap = evalSets.map {
      case (name, df) =>
        val castDF = GpuUtils.prepareColumnType(df, feturesCols, labelName,
          weightName, marginName)
        (name, GpuUtils.buildColumnDataBatch(feturesCols, labelName, weightName,
          marginName, groupName, castDF))
    }

    xgbExecParams: XGBoostExecutionParams =>
      val dataMap = prepareInputData(trainingData, evalDataMap, xgbExecParams.numWorkers,
        xgbExecParams.cacheTrainingSet)
      (buildRDDWatches(dataMap, xgbExecParams, evalDataMap.isEmpty), None)
  }

  /**
   * Transform Dataset
   *
   * @param model   supporting [[XGBoostClassificationModel]] and [[XGBoostRegressionModel]]
   * @param dataset the input Dataset to transform
   * @return the transformed DataFrame
   */
  override def transformDataset(model: Model[_], dataset: Dataset[_]): DataFrame = {

    val (booster, predictFunc, schema, featureColNames, missing) = model match {
      case m: XGBoostClassificationModel =>
        Seq(XGBoostClassificationModel._rawPredictionCol,
          XGBoostClassificationModel._probabilityCol, m.leafPredictionCol, m.contribPredictionCol)

        // predict and turn to Row
        val predictFunc =
          (booster: Booster, dm: DMatrix, originalRowItr: Iterator[Row]) => {
            val Array(rawPredictionItr, probabilityItr, predLeafItr, predContribItr) =
              m.producePredictionItrs(booster, dm)
            m.produceResultIterator(originalRowItr, rawPredictionItr, probabilityItr,
              predLeafItr, predContribItr)
          }

        // prepare the final Schema
        var schema = StructType(dataset.schema.fields ++
          Seq(StructField(name = XGBoostClassificationModel._rawPredictionCol, dataType =
            ArrayType(FloatType, containsNull = false), nullable = false)) ++
          Seq(StructField(name = XGBoostClassificationModel._probabilityCol, dataType =
            ArrayType(FloatType, containsNull = false), nullable = false)))

        if (m.isDefined(m.leafPredictionCol)) {
          schema = schema.add(StructField(name = m.getLeafPredictionCol, dataType =
            ArrayType(FloatType, containsNull = false), nullable = false))
        }
        if (m.isDefined(m.contribPredictionCol)) {
          schema = schema.add(StructField(name = m.getContribPredictionCol, dataType =
            ArrayType(FloatType, containsNull = false), nullable = false))
        }

        (m._booster, predictFunc, schema, m.getFeaturesCols, m.getMissing)

      case m: XGBoostRegressionModel =>
        Seq(XGBoostRegressionModel._originalPredictionCol, m.leafPredictionCol,
          m.contribPredictionCol)

        // predict and turn to Row
        val predictFunc =
          (booster: Booster, dm: DMatrix, originalRowItr: Iterator[Row]) => {
            val Array(rawPredictionItr, predLeafItr, predContribItr) =
              m.producePredictionItrs(booster, dm)
            m.produceResultIterator(originalRowItr, rawPredictionItr, predLeafItr,
              predContribItr)
          }

        // prepare the final Schema
        var schema = StructType(dataset.schema.fields ++
          Seq(StructField(name = XGBoostRegressionModel._originalPredictionCol, dataType =
            ArrayType(FloatType, containsNull = false), nullable = false)))

        if (m.isDefined(m.leafPredictionCol)) {
          schema = schema.add(StructField(name = m.getLeafPredictionCol, dataType =
            ArrayType(FloatType, containsNull = false), nullable = false))
        }
        if (m.isDefined(m.contribPredictionCol)) {
          schema = schema.add(StructField(name = m.getContribPredictionCol, dataType =
            ArrayType(FloatType, containsNull = false), nullable = false))
        }

        (m._booster, predictFunc, schema, m.getFeaturesCols, m.getMissing)
    }

    val sc = dataset.sparkSession.sparkContext

    // Prepare some vars will be passed to executors.
    val bOrigSchema = sc.broadcast(dataset.schema)
    val bRowSchema = sc.broadcast(schema)
    val bBooster = sc.broadcast(booster)
    val bBoosterFlag = sc.broadcast(new BoosterFlag)

    // Small vars so don't need to broadcast them
    val isLocal = sc.isLocal
    val featureIds = featureColNames.distinct.map(dataset.schema.fieldIndex)

    // start transform by df->rd->mapPartition
    val rowRDD: RDD[Row] = GpuUtils.toColumnarRdd(dataset.asInstanceOf[DataFrame]).mapPartitions {
      tableIters =>
        // UnsafeProjection is not serializable so do it on the executor side
        val toUnsafe = UnsafeProjection.create(bOrigSchema.value)

        // booster is visible for all spark tasks in the same executor
        val booster = bBooster.value
        val boosterFlag = bBoosterFlag.value

        synchronized {
          // there are two kind of race conditions,
          // 1. multi-taskes set parameters at a time
          // 2. one task sets parameter and another task reads the parameter
          // both of them can cause potential un-expected behavior, moreover,
          //      it may cause executor crash
          // So add synchronized to allow only one task to set parameter if it is not set.
          // and rely on BlockManager to ensure the same booster only be called once to
          // set parameter.
          if (!boosterFlag.isGpuParamsSet) {
            // set some params of gpu related to booster
            // - gpu id
            // - predictor: Force to gpu predictor since native doesn't save predictor.
            val gpuId = if (!isLocal) XGBoost.getGPUAddrFromResources else 0
            booster.setParam("gpu_id", gpuId.toString)
            booster.setParam("predictor", "gpu_predictor")
            logger.info("GPU transform on device: " + gpuId)
            boosterFlag.isGpuParamsSet = true;
          }
        }

        // Iterator on Row
        new Iterator[Row] {
          // Convert InternalRow to Row
          private val converter: InternalRow => Row = CatalystTypeConverters
            .createToScalaConverter(bOrigSchema.value)
            .asInstanceOf[InternalRow => Row]
          // GPU batches read in must be closed by the receiver (us)
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
              val dataTypes = bOrigSchema.value.fields.map(x => x.dataType)
              iter = withResource(tableIters.next()) { table =>
                val gpuColumnBatch = new GpuColumnBatch(table, bOrigSchema.value)
                // Create DMatrix
                val feaTable = gpuColumnBatch.slice(GpuUtils.seqIntToSeqInteger(featureIds).asJava)
                if (feaTable == null) {
                  throw new RuntimeException("Something wrong for feature indices")
                }
                try {
                  val cudfColumnBatch = new CudfColumnBatch(feaTable, null, null, null)
                  val dm = new DMatrix(cudfColumnBatch, missing, 1)
                  if (dm == null) {
                    Iterator.empty
                  } else {
                    try {
                      currentBatch = new ColumnarBatch(
                        GpuUtils.extractBatchToHost(table, dataTypes),
                        table.getRowCount().toInt)
                      val rowIterator = currentBatch.rowIterator().asScala
                        .map(toUnsafe)
                        .map(converter(_))
                      predictFunc(booster, dm, rowIterator)

                    } finally {
                      dm.delete()
                    }
                  }
                } finally {
                  feaTable.close()
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

    bOrigSchema.unpersist(blocking = false)
    bRowSchema.unpersist(blocking = false)
    bBooster.unpersist(blocking = false)
    dataset.sparkSession.createDataFrame(rowRDD, schema)
  }

  /**
   * Transform schema
   *
   * @param est supporting XGBoostClassifier/XGBoostClassificationModel and
   *                 XGBoostRegressor/XGBoostRegressionModel
   * @param schema   the input schema
   * @return the transformed schema
   */
  override def transformSchema(
      est: XGBoostEstimatorCommon,
      schema: StructType): StructType = {

    val fit = est match {
      case _: XGBoostClassifier | _: XGBoostRegressor => true
      case _ => false
    }

    val Seq(label, weight, margin) = GpuUtils.getColumnNames(est)(est.labelCol, est.weightCol,
      est.baseMarginCol)

    GpuUtils.validateSchema(schema, est.getFeaturesCols, label, weight, margin, fit)
  }

  /**
   * Repartition all the Columnar Dataset (training and evaluation) to nWorkers,
   * and assemble them into a map
   */
  private def prepareInputData(
      trainingData: ColumnDataBatch,
      evalSetsMap: Map[String, ColumnDataBatch],
      nWorkers: Int,
      isCacheData: Boolean): Map[String, ColumnDataBatch] = {
    // Cache is not supported
    if (isCacheData) {
      logger.warn("the cache param will be ignored by GPU pipeline!")
    }

    (Map(TRAIN_NAME -> trainingData) ++ evalSetsMap).map {
      case (name, colData) =>
        // No light cost way to get number of partitions from DataFrame, so always repartition
        val newDF = colData.groupColName
          .map(gn => repartitionForGroup(gn, colData.rawDF, nWorkers))
          .getOrElse(repartitionInputData(colData.rawDF, nWorkers))
        name -> ColumnDataBatch(newDF, colData.colIndices, colData.groupColName)
    }
  }

  private def repartitionInputData(dataFrame: DataFrame, nWorkers: Int): DataFrame = {
    // we can't involve any coalesce operation here, since Barrier mode will check
    // the RDD patterns which does not allow coalesce.
    dataFrame.repartition(nWorkers)
  }

  private def repartitionForGroup(
      groupName: String,
      dataFrame: DataFrame,
      nWorkers: Int): DataFrame = {
    // Group the data first
    logger.info("Start groupBy for LTR")
    val schema = dataFrame.schema
    val groupedDF = dataFrame
      .groupBy(groupName)
      .agg(collect_list(struct(schema.fieldNames.map(col): _*)) as "list")

    implicit val encoder = RowEncoder(schema)
    // Expand the grouped rows after repartition
    repartitionInputData(groupedDF, nWorkers).mapPartitions(iter => {
      new Iterator[Row] {
        var iterInRow: Iterator[Any] = Iterator.empty

        override def hasNext: Boolean = {
          if (iter.hasNext && !iterInRow.hasNext) {
            // the first is groupId, second is list
            iterInRow = iter.next.getSeq(1).iterator
          }
          iterInRow.hasNext
        }

        override def next(): Row = {
          iterInRow.next.asInstanceOf[Row]
        }
      }
    })
  }

  private def buildRDDWatches(
      dataMap: Map[String, ColumnDataBatch],
      xgbExeParams: XGBoostExecutionParams,
      noEvalSet: Boolean): RDD[() => Watches] = {

    val sc = dataMap(TRAIN_NAME).rawDF.sparkSession.sparkContext
    val maxBin = xgbExeParams.toMap.getOrElse("max_bin", 256).asInstanceOf[Int]
    // Start training
    if (noEvalSet) {
      // Get the indices here at driver side to avoid passing the whole Map to executor(s)
      val colIndicesForTrain = dataMap(TRAIN_NAME).colIndices
      GpuUtils.toColumnarRdd(dataMap(TRAIN_NAME).rawDF).mapPartitions({
        iter =>
          val iterColBatch = iter.map(table => new GpuColumnBatch(table, null))
          Iterator(() => buildWatches(
            PreXGBoost.getCacheDirName(xgbExeParams.useExternalMemory), xgbExeParams.missing,
            colIndicesForTrain, iterColBatch, maxBin))
      })
    } else {
      // Train with evaluation sets
      // Get the indices here at driver side to avoid passing the whole Map to executor(s)
      val nameAndColIndices = dataMap.map(nc => (nc._1, nc._2.colIndices))
      coPartitionForGpu(dataMap, sc, xgbExeParams.numWorkers).mapPartitions {
        nameAndColumnBatchIter =>
          Iterator(() => buildWatchesWithEval(
            PreXGBoost.getCacheDirName(xgbExeParams.useExternalMemory), xgbExeParams.missing,
            nameAndColIndices, nameAndColumnBatchIter, maxBin))
      }
    }
  }

  private def buildWatches(
      cachedDirName: Option[String],
      missing: Float,
      indices: ColumnIndices,
      iter: Iterator[GpuColumnBatch],
      maxBin: Int): Watches = {

    val (dm, time) = GpuUtils.time {
      buildDMatrix(iter, indices, missing, maxBin)
    }
    logger.debug("Benchmark[Train: Build DMatrix incrementally] " + time)
    val (aDMatrix, aName) = if (dm == null) {
      (Array.empty[DMatrix], Array.empty[String])
    } else {
      (Array(dm), Array("train"))
    }
    new Watches(aDMatrix, aName, cachedDirName)
  }

  private def buildWatchesWithEval(
      cachedDirName: Option[String],
      missing: Float,
      indices: Map[String, ColumnIndices],
      nameAndColumns: Iterator[(String, Iterator[GpuColumnBatch])],
      maxBin: Int): Watches = {
    val dms = nameAndColumns.map {
      case (name, iter) => (name, {
        val (dm, time) = GpuUtils.time {
          buildDMatrix(iter, indices(name), missing, maxBin)
        }
        logger.debug(s"Benchmark[Train build $name DMatrix] " + time)
        dm
      })
    }.filter(_._2 != null).toArray

    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName)
  }

  /**
   * Build QuantileDMatrix based on GpuColumnBatches
   *
   * @param iter a sequence of GpuColumnBatch
   * @param indices indicate the feature, label, weight, base margin column ids.
   * @param missing the missing value
   * @param maxBin the maxBin
   * @return DMatrix
   */
  private def buildDMatrix(
      iter: Iterator[GpuColumnBatch],
      indices: ColumnIndices,
      missing: Float,
      maxBin: Int): DMatrix = {
    val rapidsIterator = new RapidsIterator(iter, indices)
    new QuantileDMatrix(rapidsIterator, missing, maxBin, 1)
  }

  // zip all the Columnar RDDs into one RDD containing named column data batch.
  private def coPartitionForGpu(
    dataMap: Map[String, ColumnDataBatch],
    sc: SparkContext,
    nWorkers: Int): RDD[(String, Iterator[GpuColumnBatch])] = {
    val emptyDataRdd = sc.parallelize(
      Array.fill[(String, Iterator[GpuColumnBatch])](nWorkers)(null), nWorkers)

    dataMap.foldLeft(emptyDataRdd) {
      case (zippedRdd, (name, gdfColData)) =>
        zippedRdd.zipPartitions(GpuUtils.toColumnarRdd(gdfColData.rawDF)) {
          (itWrapper, iterCol) =>
            val itCol = iterCol.map(table => new GpuColumnBatch(table, null))
            (itWrapper.toArray :+ (name -> itCol)).filter(x => x != null).toIterator
        }
    }
  }

  private[this] class RapidsIterator(
      base: Iterator[GpuColumnBatch],
      indices: ColumnIndices) extends Iterator[CudfColumnBatch] {

    override def hasNext: Boolean = base.hasNext

    override def next(): CudfColumnBatch = {
      // Since we have sliced original Table into different tables. Needs to close the original one.
      withResource(base.next()) { gpuColumnBatch =>
        val weights = indices.weightId.map(Seq(_)).getOrElse(Seq.empty)
        val margins = indices.marginId.map(Seq(_)).getOrElse(Seq.empty)

        new CudfColumnBatch(
          gpuColumnBatch.slice(GpuUtils.seqIntToSeqInteger(indices.featureIds).asJava),
          gpuColumnBatch.slice(GpuUtils.seqIntToSeqInteger(Seq(indices.labelId)).asJava),
          gpuColumnBatch.slice(GpuUtils.seqIntToSeqInteger(weights).asJava),
          gpuColumnBatch.slice(GpuUtils.seqIntToSeqInteger(margins).asJava));
      }
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
