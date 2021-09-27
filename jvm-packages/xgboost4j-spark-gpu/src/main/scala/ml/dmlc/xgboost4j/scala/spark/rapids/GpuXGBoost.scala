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

package ml.dmlc.xgboost4j.scala.spark.rapids

import scala.collection.JavaConverters._

import ml.dmlc.xgboost4j.gpu.java.{CudfColumn, CudfColumnBatch}
import ml.dmlc.xgboost4j.java.spark.GpuColumnBatch
import ml.dmlc.xgboost4j.java.{Rabit, XGBoostError}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, EvalTrait, ExternalCheckpointManager, ObjectiveTrait}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import ml.dmlc.xgboost4j.scala.spark._
import ml.dmlc.xgboost4j.scala.spark.params.BoosterParams
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.FileSystem

import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.{GpuParallelismTracker, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.LongAccumulator

private[spark] object GpuXGBoost {
  val trainName = "train"
  private val logger = LogFactory.getLog("GpuXGBoostSpark")

  // ====  API for xgboost classifier contains the whole fit process, simliar to spark predictor
  def fitOnGpu(est: XGBoostClassifier, dataset: Dataset[_], sampler: Option[GpuSampler]):
      XGBoostClassificationModel = {
    // Check schema and cast columns' type
    val Seq(labelName, weightName, marginName) = MLUtils.getColumnNames(est)(
      est.labelCol,
      est.weightCol,
      est.baseMarginCol)
    val castedDF = MLUtils.prepareColumnType(dataset, est.getFeaturesCols,
      labelName, weightName, marginName)
    trainOnGpu(est, castedDF, sampler)
      .setParent(est)
      .copy(est.extractParamMap)
  }

  // API for xgboost classifier plays the same functionality with CPU train
  protected def trainOnGpu(
      est: XGBoostClassifier,
      columnarDF: DataFrame,
      sampler: Option[GpuSampler]): XGBoostClassificationModel = {
    // Check some classifier parameters
    require(est.isDefined(est.objective), "Parameter \'objective\' must be set.")
    if (!est.isDefined(est.evalMetric) || est.getEvalMetric.isEmpty) {
      val evMetric = if (est.getObjective.startsWith("multi")) "merror" else "error"
      est.setEvalMetric(evMetric)
    }

    if (est.isDefined(est.customObj) && est.getOrDefault(est.customObj) != null) {
      est.setObjectiveType("classification")
    }

    val Seq(labelName, weightName, marginName) = MLUtils.getColumnNames(est)(
      est.labelCol, est.weightCol, est.baseMarginCol)
    // Check columns and build column data batch
    val trainingData = GpuUtils.buildColumnDataBatch(est.getFeaturesCols,
      labelName, weightName, marginName, "", columnarDF)
    // eval map
    val evalDataMap = est.getEvalSets(est.getUserParams).map {
      case (name, df) =>
        val castDF = MLUtils.prepareColumnType(df, est.getFeaturesCols,
          labelName, weightName, marginName)

        (name, GpuUtils.buildColumnDataBatch(est.getFeaturesCols, labelName, weightName,
          marginName, "", castDF))
    }

    // Create an accumulator to compute the class number
    val accNumClass = columnarDF.sparkSession.sparkContext.longAccumulator("Class Number")
    val (_booster, _metrics) = trainDistributedOnGpu(trainingData,
      est.MLlib2XGBoostParams, evalDataMap, accNumClass, sampler)

    // Use accumulator to transfer numClass back to driver here.
    val _numClass = accNumClass.value.toInt
    logger.debug(s"Accumulator returns the number of class: ${_numClass}")
    val model = new XGBoostClassificationModel(est.uid, _numClass, _booster)
    val summary = XGBoostTrainingSummary(_metrics)
    model.setSummary(summary)
  }

  // ===== API for xgboost regressor contains the whole fit process, simliar to spark predictor
  def fitOnGpu(est: XGBoostRegressor, dataset: Dataset[_], sampler: Option[GpuSampler]):
      XGBoostRegressionModel = {
    // Check schema and cast columns' type
    val Seq(labelName, weightName, marginName) = MLUtils.getColumnNames(est)(
      est.labelCol,
      est.weightCol,
      est.baseMarginCol)
    val castedDF = MLUtils.prepareColumnType(dataset, est.getFeaturesCols,
      labelName, weightName, marginName)
    trainOnGpu(est, castedDF, sampler)
      .setParent(est)
      .copy(est.extractParamMap)
  }

  // API for xgboost regressor plays the same functionality with CPU train
  protected def trainOnGpu(
      est: XGBoostRegressor,
      columnarDF: DataFrame,
      sampler: Option[GpuSampler]): XGBoostRegressionModel = {
    // Check some regressor parameters
    require(est.isDefined(est.objective), "Parameter \'objective\' must be set.")
    if (!est.isDefined(est.evalMetric) || est.getEvalMetric.isEmpty) {
      val evMetric = if (est.getObjective.startsWith("rank")) "map" else "rmse"
      est.setEvalMetric(evMetric)
    }

    if (est.isDefined(est.customObj) && est.getOrDefault(est.customObj) != null) {
      est.setObjectiveType("regression")
    }

    // Check columns and build column data batch
    val Seq(labelName, weightName, marginName, groupName) = MLUtils.getColumnNames(est)(
      est.labelCol,
      est.weightCol,
      est.baseMarginCol,
      est.groupCol)
    val trainingData = GpuUtils.buildColumnDataBatch(est.getFeaturesCols,
      labelName, weightName, marginName, groupName, columnarDF)

    // eval map
    val evalDataMap = est.getEvalSets(est.getUserParams).map {
      case (name, df) =>
        val castedDF = MLUtils.prepareColumnType(df, est.getFeaturesCols,
          labelName, weightName, marginName)
        (name, GpuUtils.buildColumnDataBatch(est.getFeaturesCols, labelName, weightName,
          marginName, groupName, castedDF))
    }
    val (_booster, _metrics) = trainDistributedOnGpu(trainingData,
      est.MLlib2XGBoostParams, evalDataMap, null, sampler)
    // Create the model
    val model = new XGBoostRegressionModel(est.uid, _booster)
    val summary = XGBoostTrainingSummary(_metrics)
    model.setSummary(summary)
    model
  }


  // ==========  XGBoost ==========
  @throws(classOf[XGBoostError])
  private def trainDistributedOnGpu(
      trainingData: ColumnDataBatch,
      params: Map[String, Any],
      evalSetsMap: Map[String, ColumnDataBatch],
      accNumClass: LongAccumulator,
      sampler: Option[GpuSampler] = None): (Booster, Map[String, Array[Float]]) = {
    logger.info(s"Running GPU XGBoost with parameters:\n${params.mkString("\n")}")
    val sc = trainingData.rawDF.sparkSession.sparkContext
    val xgbParamsFactory = new XGBoostExecutionParamsFactory(params, sc)
    val xgbExecParams = xgbParamsFactory.buildXGBRuntimeParams
    val dataMap = prepareInputData(trainingData, evalSetsMap, xgbExecParams.numWorkers,
      xgbExecParams.cacheTrainingSet)
    val prevBooster = xgbExecParams.checkpointParam.map { checkpointParam =>
      val checkpointManager = new ExternalCheckpointManager(
        checkpointParam.checkpointPath, FileSystem.get(sc.hadoopConfiguration))
      checkpointManager.cleanUpHigherVersions(xgbExecParams.numRounds)
      checkpointManager.loadCheckpointAsScalaBooster()
    }.orNull
    try {
      // Train for every ${savingRound} rounds and save the partially completed booster
      val tracker = XGBoost.startTracker(xgbExecParams.numWorkers, xgbExecParams.trackerConf)
      val (booster, metrics) = try {
        val parallelismTracker = new GpuParallelismTracker(sc,
          xgbExecParams.timeoutRequestWorkers, xgbExecParams.numWorkers)
        val boostersAndMetrics = trainOnGpuInternal(dataMap, xgbExecParams, tracker.getWorkerEnvs,
          prevBooster, evalSetsMap.isEmpty, accNumClass, sampler)

        val sparkJobThread = new Thread() {
          override def run() {
            boostersAndMetrics.foreachPartition(() => _)
          }
        }
        sparkJobThread.setUncaughtExceptionHandler(tracker)
        sparkJobThread.start()
        val trackerReturnVal = parallelismTracker.executeOnGpu(tracker.waitFor(0L))
        logger.info(s"GPU XGBoost Rabit returns with exit code $trackerReturnVal")
        val (booster, metrics) = XGBoost.postTrackerReturnProcessing(trackerReturnVal,
          boostersAndMetrics, sparkJobThread)
        (booster, metrics)
      } finally {
        tracker.stop()
      }
      // we should delete the checkpoint directory after a successful training
      xgbExecParams.checkpointParam.foreach { cpParam =>
        if (!xgbExecParams.checkpointParam.get.skipCleanCheckpoint) {
          val checkpointManager = new ExternalCheckpointManager(
            cpParam.checkpointPath, FileSystem.get(sc.hadoopConfiguration))
          checkpointManager.cleanPath()
        }
      }
      (booster, metrics)
    } catch {
      case t: Throwable =>
        // if the job was aborted due to an exception
        logger.error("The job was aborted due to ", t)
        sc.stop()
        throw t
    } finally {
      // cache is not supported
    }
  }

  @throws(classOf[XGBoostError])
  private def trainOnGpuInternal(
      dataMap: Map[String, ColumnDataBatch],
      xgbExeParams: XGBoostExecutionParams,
      rabitEnv: java.util.Map[String, String],
      prevBooster: Booster,
      noEvalSet: Boolean,
      accNumClass: LongAccumulator,
      sampler: Option[GpuSampler] = None): RDD[(Booster, Map[String, Array[Float]])] = {
    // force gpu tree_method
    val updatedParams = overrideParamsToUseGPU(xgbExeParams)
    val sc = dataMap(trainName).rawDF.sparkSession.sparkContext
    val isLocal = sc.isLocal
    // Start training
    if (noEvalSet) {
      // Get the indices here at driver side to avoid passing the whole Map to executor(s)
      val colIndicesForTrain = dataMap(trainName).colIndices
      GpuUtils.toColumnarRdd(dataMap(trainName).rawDF).mapPartitions({
        iter =>
          val iterColBatch = iter.map(table => new GpuColumnBatch(table, null,
            sampler.getOrElse(null)))
          val orParams = appendGpuIdToParams(updatedParams, isLocal)
          // default max bin to 16 to keep align with original
          val maxBin = orParams.toMap.getOrElse("max_bin", 16).asInstanceOf[Int]
          val buildWatchesFn = () => {
            buildWatches(
              XGBoost.getCacheDirName(orParams.useExternalMemory), orParams.missing,
              colIndicesForTrain, iterColBatch, accNumClass, maxBin)
          }

          buildDistributedBooster(buildWatchesFn, orParams, rabitEnv,
            orParams.obj, orParams.eval, prevBooster)
      }).cache()
    } else {
      // Train with evaluation sets
      // Get the indices here at driver side to avoid passing the whole Map to executor(s)
      val nameAndColIndices = dataMap.map(nc => (nc._1, nc._2.colIndices))
      coPartitionForGpu(dataMap, sc, updatedParams.numWorkers).mapPartitions {
        nameAndColumnBatchIter =>
          val orParams = appendGpuIdToParams(updatedParams, isLocal)
          // default max bin to 16 to keep align with original
          val maxBin = orParams.toMap.getOrElse("max_bin", 16).asInstanceOf[Int]

          val buildWatchesFn = () => {
            buildWatchesWithEval(
              XGBoost.getCacheDirName(orParams.useExternalMemory), orParams.missing,
              nameAndColIndices, nameAndColumnBatchIter, accNumClass, maxBin)
          }

          buildDistributedBooster(buildWatchesFn, orParams, rabitEnv,
            orParams.obj, orParams.eval, prevBooster)
      }.cache()
    }
  }

  private[spark] def buildDistributedBooster(
    buildWatches: () => Watches,
    xgbExecutionParam: XGBoostExecutionParams,
    rabitEnv: java.util.Map[String, String],
    obj: ObjectiveTrait,
    eval: EvalTrait,
    prevBooster: Booster): Iterator[(Booster, Map[String, Array[Float]])] = {

    val taskId = TaskContext.getPartitionId().toString
    val attempt = TaskContext.get().attemptNumber.toString
    rabitEnv.put("DMLC_TASK_ID", taskId)
    rabitEnv.put("DMLC_NUM_ATTEMPT", attempt)
    rabitEnv.put("DMLC_WORKER_STOP_PROCESS_ON_ERROR", "false")
    val numRounds = xgbExecutionParam.numRounds
    val makeCheckpoint = xgbExecutionParam.checkpointParam.isDefined && taskId.toInt == 0
    var watches: Watches = null
    try {
      Rabit.init(rabitEnv)

      watches = buildWatches()
      // to workaround the empty partitions in training dataset,
      // this might not be the best efficient implementation, see
      // (https://github.com/dmlc/xgboost/issues/1277)
      if (watches.toMap("train").rowNum == 0) {
        throw new XGBoostError(
          s"detected an empty partition in the training data, partition ID:" +
            s" ${TaskContext.getPartitionId()}")
      }

      GpuXGBoost.checkNumClass(watches, xgbExecutionParam.toMap)
      val numEarlyStoppingRounds = xgbExecutionParam.earlyStoppingParams.numEarlyStoppingRounds
      val metrics = Array.tabulate(watches.size)(_ => Array.ofDim[Float](numRounds))
      val externalCheckpointParams = xgbExecutionParam.checkpointParam
      val booster = if (makeCheckpoint) {
        SXGBoost.trainAndSaveCheckpoint(
          watches.toMap("train"), xgbExecutionParam.toMap, numRounds,
          watches.toMap, metrics, obj, eval,
          earlyStoppingRound = numEarlyStoppingRounds, prevBooster, externalCheckpointParams)
      } else {
        SXGBoost.train(watches.toMap("train"), xgbExecutionParam.toMap, numRounds,
          watches.toMap, metrics, obj, eval,
          earlyStoppingRound = numEarlyStoppingRounds, prevBooster)
      }
      Iterator(booster -> watches.toMap.keys.zip(metrics).toMap)
    } catch {
      case xgbException: XGBoostError =>
        logger.error(s"XGBooster worker $taskId has failed $attempt times due to ", xgbException)
        throw xgbException
    } finally {
      Rabit.shutdown()
      if (watches != null) watches.delete()
    }
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

    (Map(trainName -> trainingData) ++ evalSetsMap).map {
      case (name, colData) =>
        // No light cost way to get number of partitions from DataFrame, so always repartition
        val newDF = colData.groupColName
          .map(gn => repartitionForGroup(gn, colData.rawDF, nWorkers))
          .getOrElse(colData.rawDF.repartition(nWorkers))
        name -> ColumnDataBatch(newDF, colData.colIndices, colData.groupColName)
    }
  }

  private def repartitionForGroup(groupName: String,
        dataFrame: DataFrame,
        nWorkers: Int): DataFrame = {
    // Group the data first
    logger.info("Start groupBy for ltr")
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

  // This method should be called on executor side
  private def appendGpuIdToParams(xgbParams: XGBoostExecutionParams,
      isLocal: Boolean): XGBoostExecutionParams = {
    val gpuId = GpuUtils.getGpuId(isLocal)
    logger.info("XGboost GPU training using device: " + gpuId)
    val newParams = xgbParams.toMap + ("gpu_id" -> gpuId.toString)
    xgbParams.setRawParamMap(newParams)
    xgbParams
  }

  // FIXME implicit?
  private def seqIntToSeqInteger(x: Seq[Int]): Seq[Integer] = x.map(new Integer(_))

  private[this] class RapidsIterator(base: Iterator[GpuColumnBatch], inferNumClass: Boolean,
      indices: ColumnIndices) extends Iterator[CudfColumnBatch] {
    var maxLabels: Double = 0.0f

    override def hasNext: Boolean = base.hasNext


    override def next(): CudfColumnBatch = {
      val gpuColumnBatch = base.next()

      if (inferNumClass) {
        val tmpMax = gpuColumnBatch.getMaxInColumn(indices.labelId)
        maxLabels = if (tmpMax > maxLabels) tmpMax else maxLabels
      }

      val weights = indices.weightId.map(Seq(_)).getOrElse(Seq.empty)
      val margins = indices.marginId.map(Seq(_)).getOrElse(Seq.empty)

      new CudfColumnBatch(
        gpuColumnBatch.slice(seqIntToSeqInteger(indices.featureIds).asJava),
        gpuColumnBatch.slice(seqIntToSeqInteger(Seq(indices.labelId)).asJava),
        gpuColumnBatch.slice(seqIntToSeqInteger(weights).asJava),
        gpuColumnBatch.slice(seqIntToSeqInteger(margins).asJava));
    }
  }

  // FIXME This is a WAR before native supports building DMatrix incrementally
  private def buildDMatrix(
      iter: Iterator[GpuColumnBatch],
      indices: ColumnIndices,
      missing: Float,
      inferNumClass: Boolean,
      maxBin: Int): (DMatrix, Double) = {
    // FIXME add option or dynamic to check.
    if (true) {
      val rapidsIterator = new RapidsIterator(iter, inferNumClass, indices)
      (new DeviceQuantileDMatrix(rapidsIterator, missing, maxBin, 1), rapidsIterator.maxLabels)
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

      val maxDoubleLabel = if (inferNumClass) {
        singleColBatch.getMaxInColumn(indices.labelId)
      } else {
        0.0
      }
      singleColBatch.close()
      (dm, maxDoubleLabel)
    }
  }

  private def buildWatches(cachedDirName: Option[String],
      missing: Float,
      indices: ColumnIndices,
      iter: Iterator[GpuColumnBatch],
      accNumClass: LongAccumulator,
      maxBin: Int): Watches = {
    val ((dm, numClass), time) = MLUtils.time {
      buildDMatrix(iter, indices, missing, accNumClass != null, maxBin)
    }
    logger.debug("Benchmark[Train: Build DMatrix incrementally] " + time)
    val (aDMatrix, aName) = if (dm == null) {
      (Array.empty[DMatrix], Array.empty[String])
    } else {
      (Array(dm), Array("train"))
    }
    new GpuWatches(aDMatrix, aName, cachedDirName, accNumClass, numClass)
  }

  private def buildWatchesWithEval(cachedDirName: Option[String],
      missing: Float,
      indices: Map[String, ColumnIndices],
      nameAndColumns: Iterator[(String, Iterator[GpuColumnBatch])],
      accNumClass: LongAccumulator,
      maxBin: Int): Watches = {
    var numClass: Double = 0.0
    val dms = nameAndColumns.map {
      case (name, iter) => (name, {
        val inferring = accNumClass != null && name == "train"
        val ((dm, tmpNumClass), time) = MLUtils.time {
          buildDMatrix(iter, indices(name), missing, inferring, maxBin)
        }
        logger.debug(s"Benchmark[Train build $name DMatrix] " + time)
        if (inferring) {
          numClass = tmpNumClass
        }
        dm
      })
    }.filter(_._2 != null).toArray

    new GpuWatches(dms.map(_._2), dms.map(_._1), cachedDirName, accNumClass, numClass)
  }

  // Should be called under Rabit environment
  private[spark] def checkNumClass(watches: Watches, params: Map[String, Any]): Unit = {
    watches match {
      case w: GpuWatches =>
        if (w.accNumClass != null) {
          val element: Array[Float] = Array(w.numClass.toFloat)
          val numClassRet = Rabit.allReduce(element, Rabit.OpType.MAX)
          require(numClassRet.nonEmpty, "Failed to infer class number.")

          val maxFloatLabel = numClassRet.head
          require(maxFloatLabel.isValidInt, s"Classifier found max label value =" +
          s" $maxFloatLabel but requires integers in range [0, ... ${Int.MaxValue})")

          val inferredNumClass = maxFloatLabel.toInt + 1
          if (params.contains("num_class")) {
            require(params("num_class") == inferredNumClass, "The number of classes in Dataset" +
              " doesn't match \'num_class\' in parameters.")
          }
          // Save the num class to accumulator
          if (Rabit.getRank() == 0) {
            w.accNumClass.add(inferredNumClass)
          }
        }
      case _ =>
    }
  }

} // End of GpuXGBoost

private class GpuWatches(
  override val datasets: Array[DMatrix],
  override val names: Array[String],
  override val cacheDirName: Option[String],
  val accNumClass: LongAccumulator,
  val numClass: Double = 0.0) extends Watches(datasets, names, cacheDirName)

