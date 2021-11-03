/*
 Copyright (c) 2021 by Contributors

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

package ml.dmlc.xgboost4j.scala.nvidia.spark

import scala.collection.Iterator
import scala.collection.JavaConverters._

import com.nvidia.spark.rapids.GpuColumnVector
import ml.dmlc.xgboost4j.gpu.java.{CudfColumn, CudfColumnBatch}
import ml.dmlc.xgboost4j.java.Rabit
import ml.dmlc.xgboost4j.java.nvidia.spark.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, DeviceQuantileDMatrix}
import ml.dmlc.xgboost4j.scala.spark.params.{BoosterParams, XGBoostEstimatorCommon}
import ml.dmlc.xgboost4j.scala.spark.{PreXGBoost, PreXGBoostProvider, Watches, XGBoostClassificationModel, XGBoostClassifier, XGBoostExecutionParams, XGBoostRegressionModel, XGBoostRegressor}
import org.apache.commons.logging.LogFactory

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{SparkContext, SparkUtils, TaskContext}
import org.apache.spark.ml.{Estimator, Model, PipelineStage}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection
import org.apache.spark.sql.functions.{col, collect_list, struct}
import org.apache.spark.sql.types.{ArrayType, FloatType, StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class GpuPreXGBoost extends PreXGBoostProvider {

  /**
   * Whether provider is enabled or not
   *
   * @param dataset the input dataset
   * @return Boolean
   */
  override def providerEnabled(dataset: Option[Dataset[_]]): Boolean = {
    GpuPreXGBoost.providerEnabled(dataset)
  }

  /**
   * Convert the Dataset[_] to RDD[Watches] which will be fed to XGBoost
   *
   * @param estimator supports XGBoostClassifier and XGBoostRegressor
   * @param dataset   the training data
   * @param params    all user defined and defaulted params
   * @return [[XGBoostExecutionParams]] => (RDD[[Watches]], Option[ RDD[_] ])
   *         RDD[Watches] will be used as the training input
   *         Option[ RDD[_] ] is the optional cached RDD
   */
  override def buildDatasetToRDD(estimator: Estimator[_],
    dataset: Dataset[_],
    params: Map[String, Any]): XGBoostExecutionParams => (RDD[Watches], Option[RDD[_]]) = {
    GpuPreXGBoost.buildDatasetToRDD(estimator, dataset, params)
  }

  /**
   * Transform Dataset
   *
   * @param model   supporting [[XGBoostClassificationModel]] and [[XGBoostRegressionModel]]
   * @param dataset the input Dataset to transform
   * @return the transformed DataFrame
   */
  override def transformDataset(model: Model[_], dataset: Dataset[_]): DataFrame = {
    GpuPreXGBoost.transformDataset(model, dataset)
  }

  override def transformSchema(pipelineStage: PipelineStage, schema: StructType): StructType = {
    // TODO we should transform schema for features_cols/label/weight/baseMargin ...
    schema
  }
}

object GpuPreXGBoost extends PreXGBoostProvider {

  private val logger = LogFactory.getLog("XGBoostSpark")
  private val FEATURES_COLS = "features_cols"
  private val TRAIN_NAME = "train"

  override def providerEnabled(dataset: Option[Dataset[_]]): Boolean = {
    val optionConf = dataset.map(ds => Some(ds.sparkSession.sparkContext.getConf))
      .getOrElse(SparkUtils.getActiveSparkContext().map(sc => sc.getConf))

    optionConf
      .map(conf => conf.get("spark.sql.extensions", "")
        .split(",")
        .contains("com.nvidia.spark.rapids.SQLExecPlugin") &&
        conf.getBoolean("spark.rapids.sql.enabled", false))
      .getOrElse(false)

  }

  /**
   * Convert the Dataset[_] to RDD[Watches] which will be fed to XGBoost
   *
   * @param estimator supports XGBoostClassifier and XGBoostRegressor
   * @param dataset   the training data
   * @param params    all user defined and defaulted params
   * @return [[XGBoostExecutionParams]] => (RDD[[Watches]], Option[ RDD[_] ])
   *         RDD[Watches] will be used as the training input
   *         Option[ RDD[_] ] is the optional cached RDD
   */
  override def buildDatasetToRDD(
    estimator: Estimator[_],
    dataset: Dataset[_],
    params: Map[String, Any]): XGBoostExecutionParams => (RDD[Watches], Option[RDD[_]]) = {

    val (Seq(labelName, weightName, marginName), feturesCols, groupName, evalSets) =
      estimator match {
        case est: XGBoostEstimatorCommon =>
          val groupName = estimator match {
            case regressor: XGBoostRegressor => regressor.getGroupCol
            case _: XGBoostClassifier => ""
            case _ => throw new RuntimeException("Unsupporting " + estimator)
          }
          // Check schema and cast columns' type
          (MLUtils.getColumnNames(est)(est.labelCol, est.weightCol, est.baseMarginCol),
            est.getFeaturesCols, groupName, est.getEvalSets(params))
    }

    val castedDF = MLUtils.prepareColumnType(dataset, feturesCols, labelName, weightName,
      marginName)

    // Check columns and build column data batch
    val trainingData = GpuUtils.buildColumnDataBatch(feturesCols,
      labelName, weightName, marginName, "", castedDF)

    // eval map
    val evalDataMap = evalSets.map {
      case (name, df) =>
        val castDF = MLUtils.prepareColumnType(df, feturesCols, labelName,
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
          (broadcastBooster: Broadcast[Booster], dm: DMatrix, originalRowItr: Iterator[Row]) => {
            val Array(rawPredictionItr, probabilityItr, predLeafItr, predContribItr) =
              m.producePredictionItrs(broadcastBooster, dm)
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
          (broadcastBooster: Broadcast[Booster], dm: DMatrix, originalRowItr: Iterator[Row]) => {
            val Array(rawPredictionItr, predLeafItr, predContribItr) =
              m.producePredictionItrs(broadcastBooster, dm)
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

    // Small vars so don't need to broadcast them
    val isLocal = sc.isLocal
    val featureIds = featureColNames.distinct.map(dataset.schema.fieldIndex)

    // start transform by df->rd->mapPartition
    val rowRDD: RDD[Row] = GpuUtils.toColumnarRdd(dataset.asInstanceOf[DataFrame]).mapPartitions {
      iter =>

        // UnsafeProjection is not serializable so do it on the executor side
        val toUnsafe = UnsafeProjection.create(bOrigSchema.value)

        new Iterator[Row] {
          private var batchCnt = 0
          private var converter: RowConverter = null

          // GPU batches read in must be closed by the receiver (us)
          @transient var cb: ColumnarBatch = null
          var it: Iterator[Row] = null

          TaskContext.get().addTaskCompletionListener[Unit](_ => {
            if (batchCnt > 0) {
              Rabit.shutdown()
            }
            closeCurrentBatch()
          })

          private def closeCurrentBatch(): Unit = {
            if (cb != null) {
              cb.close()
              cb = null
            }
          }

          def loadNextBatch(): Unit = {
            closeCurrentBatch()
            if (it != null) {
              it = null
            }
            if (iter.hasNext) {
              val table = iter.next()

              if (batchCnt == 0) {
                // do we really need to involve rabit in transform?
                // Init rabit
                val rabitEnv = Map(
                  "DMLC_TASK_ID" -> TaskContext.getPartitionId().toString)
                Rabit.init(rabitEnv.asJava)

                converter = new RowConverter(bOrigSchema.value,
                  (0 until table.getNumberOfColumns).map(table.getColumn(_).getType))
              }

              val dataTypes = bOrigSchema.value.fields.map(x => x.dataType)

              val devCb = GpuColumnVector.from(table, dataTypes)

              try {
                cb = new ColumnarBatch(
                  GpuColumnVector.extractColumns(devCb).map(_.copyToHost()),
                  devCb.numRows())

                val rowIterator = cb.rowIterator().asScala
                  .map(toUnsafe)
                  .map(converter.toExternalRow(_))

                val gpuColumnBatch = new GpuColumnBatch(table, bOrigSchema.value)

                // Create DMatrix
                val cudfColumnBatch = new CudfColumnBatch(gpuColumnBatch.slice(
                  GpuUtils.seqIntToSeqInteger(featureIds).asJava), null, null, null)
                val dm = new DMatrix(cudfColumnBatch, missing, 1)
                it = {
                  if (dm == null) {
                    Iterator.empty
                  } else {
                    try {
                      // set some params of gpu related to booster
                      // - gpu id
                      // - predictor: Force to gpu predictor since native doesn't save predictor.
                      val gpuId = GpuUtils.getGpuId(isLocal)
                      bBooster.value.setParam("gpu_id", gpuId.toString)
                      bBooster.value.setParam("predictor", "gpu_predictor")
                      logger.info("XGBoost transform GPU pipeline using device: " + gpuId)

                      predictFunc(bBooster, dm, rowIterator)
                    } finally {
                      dm.delete()
                    }
                  }
                }
              } finally {
                batchCnt += 1
                devCb.close()
                table.close()
              }
            }
          }

          override def hasNext: Boolean = {
            val itHasNext = it != null && it.hasNext
            if (!itHasNext) {
              loadNextBatch()
              it != null && it.hasNext
            } else {
              itHasNext
            }
          }

          override def next(): Row = {
            if (it == null || !it.hasNext) {
              loadNextBatch()
            }
            if (it == null) {
              throw new NoSuchElementException()
            }
            it.next()
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
   * @param pipeline supporting XGBoostClassifier/XGBoostClassificationModel and
   *                 XGBoostRegressor/XGBoostRegressionModel
   * @param schema   the input schema
   * @return the transformed schema
   */
  override def transformSchema(pipeline: PipelineStage, schema: StructType): StructType = {
    schema
  }

  // repartition all the Columnar Dataset (training and evaluation) to nWorkers,
  // and assemble them into a map
  private def prepareInputData(
    trainingData: ColumnDataBatch,
    evalSetsMap: Map[String, ColumnDataBatch],
    nWorkers: Int,
    isCacheData: Boolean): Map[String, ColumnDataBatch] = {
    // Cache is not supported
    if (isCacheData) {
      logger.warn("Dataset cache is not support for Gpu pipeline!")
    }

    (Map(TRAIN_NAME -> trainingData) ++ evalSetsMap).map {
      case (name, colData) =>
        // No light cost way to get number of partitions from DataFrame, so always repartition
        val newDF = colData.groupColName
          .map(gn => repartitionForGroup(gn, colData.rawDF, nWorkers))
          .getOrElse(colData.rawDF.repartition(nWorkers))
        name -> ColumnDataBatch(newDF, colData.colIndices, colData.groupColName)
    }
  }

  private def repartitionForGroup(
    groupName: String,
    dataFrame: DataFrame,
    nWorkers: Int): DataFrame = {
    // Group the data first
    logger.info("Start groupBy for learning to rank")
    val schema = dataFrame.schema
    val groupedDF = dataFrame
      .groupBy(groupName)
      .agg(collect_list(struct(schema.fieldNames.map(col): _*)) as "list")

    implicit val encoder = RowEncoder(schema)
    // Expand the grouped rows after repartition
    groupedDF.repartition(nWorkers).mapPartitions(iter => {
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
      noEvalSet: Boolean): RDD[Watches] = {
    // force gpu tree_method
    val updatedParams = overrideParamsToUseGPU(xgbExeParams)
    val sc = dataMap(TRAIN_NAME).rawDF.sparkSession.sparkContext
    // Start training
    if (noEvalSet) {
      // Get the indices here at driver side to avoid passing the whole Map to executor(s)
      val colIndicesForTrain = dataMap(TRAIN_NAME).colIndices
      GpuUtils.toColumnarRdd(dataMap(TRAIN_NAME).rawDF).mapPartitions({
        iter =>
          val iterColBatch = iter.map(table => new GpuColumnBatch(table, null))
          val maxBin = updatedParams.toMap.getOrElse("max_bin", 16).asInstanceOf[Int]
          Iterator(buildWatches(
            PreXGBoost.getCacheDirName(updatedParams.useExternalMemory), updatedParams.missing,
            colIndicesForTrain, iterColBatch, maxBin))
      })
    } else {
      // Train with evaluation sets
      // Get the indices here at driver side to avoid passing the whole Map to executor(s)
      val nameAndColIndices = dataMap.map(nc => (nc._1, nc._2.colIndices))
      coPartitionForGpu(dataMap, sc, updatedParams.numWorkers).mapPartitions {
        nameAndColumnBatchIter =>
          val maxBin = updatedParams.toMap.getOrElse("max_bin", 16).asInstanceOf[Int]
          Iterator(buildWatchesWithEval(
            PreXGBoost.getCacheDirName(updatedParams.useExternalMemory), updatedParams.missing,
            nameAndColIndices, nameAndColumnBatchIter, maxBin))
      }
    }
  }

  private def buildWatches(cachedDirName: Option[String],
    missing: Float,
    indices: ColumnIndices,
    iter: Iterator[GpuColumnBatch],
    maxBin: Int): Watches = {
    val (dm, time) = MLUtils.time {
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
        val (dm, time) = MLUtils.time {
          buildDMatrix(iter, indices(name), missing, maxBin)
        }
        logger.debug(s"Benchmark[Train build $name DMatrix] " + time)
        dm
      })
    }.filter(_._2 != null).toArray

    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName)
  }

  // FIXME This is a WAR before native supports building DMatrix incrementally
  private def buildDMatrix(
    iter: Iterator[GpuColumnBatch],
    indices: ColumnIndices,
    missing: Float,
    maxBin: Int): DMatrix = {
    // FIXME add option or dynamic to check.
    if (true) {
      val rapidsIterator = new RapidsIterator(iter, indices)
      new DeviceQuantileDMatrix(rapidsIterator, missing, maxBin, 1)
    } else {
      // Merge all GpuColumnBatches
      val allColBatches = iter.toArray
      logger.debug(s"Train: ColumnBatch iterator size: ${allColBatches.length}.")
      val singleColBatch = GpuColumnBatch.merge(allColBatches: _*)
      // Build DMatrix
      val cudfColumnBatch = new CudfColumnBatch(singleColBatch.slice(
        seqIntToSeqInteger(indices.featureIds).asJava), null, null, null)

      val dm = new DMatrix(cudfColumnBatch, missing, 1)
      val cudfColumn = CudfColumn.from(singleColBatch.getColumnVector(indices.labelId))
      dm.setLabel(cudfColumn)

      indices.weightId.map(id => dm.setWeight(CudfColumn.from(singleColBatch.getColumnVector(id))))
      indices.marginId.map(id =>
        dm.setBaseMargin(CudfColumn.from(singleColBatch.getColumnVector(id))))

      singleColBatch.close()
      dm
    }
  }


  // mainly override the tree_method
  private def overrideParamsToUseGPU(xgbParams: XGBoostExecutionParams): XGBoostExecutionParams = {
    var updatedParams = xgbParams.toMap
    val treeMethod = "tree_method"
    if(updatedParams.contains(treeMethod)) {
      val tmValue = updatedParams(treeMethod).asInstanceOf[String]
      if (tmValue == "auto") {
        // Choose "gpu_hist" for GPU training when auto is set
        updatedParams = updatedParams + (treeMethod -> "gpu_hist")
      } else {
        require(tmValue.startsWith("gpu_"),
          "Now for training on GPU, xgboost-spark only supports tree_method as " +
            s"[${BoosterParams.supportedTreeMethods.filter(_.startsWith("gpu_")).mkString(", ")}]" +
            s", but found '$tmValue'")
      }
    } else {
      // Add "gpu_hist" as default for GPU training if not set
      updatedParams = updatedParams + (treeMethod -> "gpu_hist")
    }
    xgbParams.setRawParamMap(updatedParams)
    xgbParams
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

  // FIXME implicit?
  private def seqIntToSeqInteger(x: Seq[Int]): Seq[Integer] = x.map(new Integer(_))

  private[this] class RapidsIterator(base: Iterator[GpuColumnBatch],
    indices: ColumnIndices) extends Iterator[CudfColumnBatch] {
    var maxLabels: Double = 0.0f

    override def hasNext: Boolean = base.hasNext


    override def next(): CudfColumnBatch = {
      val gpuColumnBatch = base.next()

      val weights = indices.weightId.map(Seq(_)).getOrElse(Seq.empty)
      val margins = indices.marginId.map(Seq(_)).getOrElse(Seq.empty)

      new CudfColumnBatch(
        gpuColumnBatch.slice(seqIntToSeqInteger(indices.featureIds).asJava),
        gpuColumnBatch.slice(seqIntToSeqInteger(Seq(indices.labelId)).asJava),
        gpuColumnBatch.slice(seqIntToSeqInteger(weights).asJava),
        gpuColumnBatch.slice(seqIntToSeqInteger(margins).asJava));
    }
  }

}
