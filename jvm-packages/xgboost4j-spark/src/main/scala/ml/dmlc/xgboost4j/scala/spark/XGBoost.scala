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

import java.io.File
import java.nio.file.Files

import scala.collection.{AbstractIterator, mutable}
import scala.util.Random
import scala.collection.JavaConverters._

import ml.dmlc.xgboost4j.java.{IRabitTracker, Rabit, XGBoostError, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.RabitTracker
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams
import ml.dmlc.xgboost4j.scala.ExternalCheckpointManager
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.commons.io.FileUtils
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.FileSystem

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkParallelismTracker, TaskContext, TaskFailedListener}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel


/**
 * Rabit tracker configurations.
 *
 * @param workerConnectionTimeout The timeout for all workers to connect to the tracker.
 *                                Set timeout length to zero to disable timeout.
 *                                Use a finite, non-zero timeout value to prevent tracker from
 *                                hanging indefinitely (in milliseconds)
 *                                (supported by "scala" implementation only.)
 * @param trackerImpl Choice between "python" or "scala". The former utilizes the Java wrapper of
 *                    the Python Rabit tracker (in dmlc_core), whereas the latter is implemented
 *                    in Scala without Python components, and with full support of timeouts.
 *                    The Scala implementation is currently experimental, use at your own risk.
 */
case class TrackerConf(workerConnectionTimeout: Long, trackerImpl: String)

object TrackerConf {
  def apply(): TrackerConf = TrackerConf(0L, "python")
}

private[this] case class XGBoostExecutionEarlyStoppingParams(numEarlyStoppingRounds: Int,
                                                             maximizeEvalMetrics: Boolean)

private[this] case class XGBoostExecutionInputParams(trainTestRatio: Double, seed: Long)

private[this] case class XGBoostExecutionParams(
    numWorkers: Int,
    numRounds: Int,
    useExternalMemory: Boolean,
    obj: ObjectiveTrait,
    eval: EvalTrait,
    missing: Float,
    allowNonZeroForMissing: Boolean,
    trackerConf: TrackerConf,
    timeoutRequestWorkers: Long,
    checkpointParam: Option[ExternalCheckpointParams],
    xgbInputParams: XGBoostExecutionInputParams,
    earlyStoppingParams: XGBoostExecutionEarlyStoppingParams,
    cacheTrainingSet: Boolean) {

  private var rawParamMap: Map[String, Any] = _

  def setRawParamMap(inputMap: Map[String, Any]): Unit = {
    rawParamMap = inputMap
  }

  def toMap: Map[String, Any] = {
    rawParamMap
  }
}

private[this] class XGBoostExecutionParamsFactory(rawParams: Map[String, Any], sc: SparkContext){

  private val logger = LogFactory.getLog("XGBoostSpark")

  private val overridedParams = overrideParams(rawParams, sc)

  /**
   * Check to see if Spark expects SSL encryption (`spark.ssl.enabled` set to true).
   * If so, throw an exception unless this safety measure has been explicitly overridden
   * via conf `xgboost.spark.ignoreSsl`.
   */
  private def validateSparkSslConf: Unit = {
    val (sparkSslEnabled: Boolean, xgboostSparkIgnoreSsl: Boolean) =
      SparkSession.getActiveSession match {
        case Some(ss) =>
          (ss.conf.getOption("spark.ssl.enabled").getOrElse("false").toBoolean,
            ss.conf.getOption("xgboost.spark.ignoreSsl").getOrElse("false").toBoolean)
        case None =>
          (sc.getConf.getBoolean("spark.ssl.enabled", false),
            sc.getConf.getBoolean("xgboost.spark.ignoreSsl", false))
      }
    if (sparkSslEnabled) {
      if (xgboostSparkIgnoreSsl) {
        logger.warn(s"spark-xgboost is being run without encrypting data in transit!  " +
          s"Spark Conf spark.ssl.enabled=true was overridden with xgboost.spark.ignoreSsl=true.")
      } else {
        throw new Exception("xgboost-spark found spark.ssl.enabled=true to encrypt data " +
          "in transit, but xgboost-spark sends non-encrypted data over the wire for efficiency. " +
          "To override this protection and still use xgboost-spark at your own risk, " +
          "you can set the SparkSession conf to use xgboost.spark.ignoreSsl=true.")
      }
    }
  }

  /**
   * we should not include any nested structure in the output of this function as the map is
   * eventually to be feed to xgboost4j layer
   */
  private def overrideParams(
      params: Map[String, Any],
      sc: SparkContext): Map[String, Any] = {
    val coresPerTask = sc.getConf.getInt("spark.task.cpus", 1)
    var overridedParams = params
    if (overridedParams.contains("nthread")) {
      val nThread = overridedParams("nthread").toString.toInt
      require(nThread <= coresPerTask,
        s"the nthread configuration ($nThread) must be no larger than " +
          s"spark.task.cpus ($coresPerTask)")
    } else {
      overridedParams = overridedParams + ("nthread" -> coresPerTask)
    }

    val numEarlyStoppingRounds = overridedParams.getOrElse(
      "num_early_stopping_rounds", 0).asInstanceOf[Int]
    overridedParams += "num_early_stopping_rounds" -> numEarlyStoppingRounds
    if (numEarlyStoppingRounds > 0 &&
      !overridedParams.contains("maximize_evaluation_metrics")) {
      if (overridedParams.contains("custom_eval")) {
        throw new IllegalArgumentException("custom_eval does not support early stopping")
      }
      val eval_metric = overridedParams("eval_metric").toString
      val maximize = LearningTaskParams.evalMetricsToMaximize contains eval_metric
      logger.info("parameter \"maximize_evaluation_metrics\" is set to " + maximize)
      overridedParams += ("maximize_evaluation_metrics" -> maximize)
    }
    overridedParams
  }

  def buildXGBRuntimeParams: XGBoostExecutionParams = {
    val nWorkers = overridedParams("num_workers").asInstanceOf[Int]
    val round = overridedParams("num_round").asInstanceOf[Int]
    val useExternalMemory = overridedParams("use_external_memory").asInstanceOf[Boolean]
    val obj = overridedParams.getOrElse("custom_obj", null).asInstanceOf[ObjectiveTrait]
    val eval = overridedParams.getOrElse("custom_eval", null).asInstanceOf[EvalTrait]
    val missing = overridedParams.getOrElse("missing", Float.NaN).asInstanceOf[Float]
    val allowNonZeroForMissing = overridedParams
                                 .getOrElse("allow_non_zero_for_missing", false)
                                 .asInstanceOf[Boolean]
    validateSparkSslConf
    if (overridedParams.contains("tree_method")) {
      require(overridedParams("tree_method") == "hist" ||
        overridedParams("tree_method") == "approx" ||
        overridedParams("tree_method") == "auto", "xgboost4j-spark only supports tree_method as" +
        " 'hist', 'approx' and 'auto'")
    }
    if (overridedParams.contains("train_test_ratio")) {
      logger.warn("train_test_ratio is deprecated since XGBoost 0.82, we recommend to explicitly" +
        " pass a training and multiple evaluation datasets by passing 'eval_sets' and " +
        "'eval_set_names'")
    }
    require(nWorkers > 0, "you must specify more than 0 workers")
    if (obj != null) {
      require(overridedParams.get("objective_type").isDefined, "parameter \"objective_type\" " +
        "is not defined, you have to specify the objective type as classification or regression" +
        " with a customized objective function")
    }
    val trackerConf = overridedParams.get("tracker_conf") match {
      case None => TrackerConf()
      case Some(conf: TrackerConf) => conf
      case _ => throw new IllegalArgumentException("parameter \"tracker_conf\" must be an " +
        "instance of TrackerConf.")
    }
    val timeoutRequestWorkers: Long = overridedParams.get("timeout_request_workers") match {
      case None => 0L
      case Some(interval: Long) => interval
      case _ => throw new IllegalArgumentException("parameter \"timeout_request_workers\" must be" +
        " an instance of Long.")
    }
    val checkpointParam =
      ExternalCheckpointParams.extractParams(overridedParams)

    val trainTestRatio = overridedParams.getOrElse("train_test_ratio", 1.0)
      .asInstanceOf[Double]
    val seed = overridedParams.getOrElse("seed", System.nanoTime()).asInstanceOf[Long]
    val inputParams = XGBoostExecutionInputParams(trainTestRatio, seed)

    val earlyStoppingRounds = overridedParams.getOrElse(
      "num_early_stopping_rounds", 0).asInstanceOf[Int]
    val maximizeEvalMetrics = overridedParams.getOrElse(
      "maximize_evaluation_metrics", true).asInstanceOf[Boolean]
    val xgbExecEarlyStoppingParams = XGBoostExecutionEarlyStoppingParams(earlyStoppingRounds,
      maximizeEvalMetrics)

    val cacheTrainingSet = overridedParams.getOrElse("cache_training_set", false)
      .asInstanceOf[Boolean]

    val xgbExecParam = XGBoostExecutionParams(nWorkers, round, useExternalMemory, obj, eval,
      missing, allowNonZeroForMissing, trackerConf,
      timeoutRequestWorkers,
      checkpointParam,
      inputParams,
      xgbExecEarlyStoppingParams,
      cacheTrainingSet)
    xgbExecParam.setRawParamMap(overridedParams)
    xgbExecParam
  }

  private[spark] def buildRabitParams : Map[String, String] = Map(
    "rabit_reduce_ring_mincount" ->
      overridedParams.getOrElse("rabit_ring_reduce_threshold", 32 << 10).toString,
    "rabit_debug" ->
      (overridedParams.getOrElse("verbosity", 0).toString.toInt == 3).toString,
    "rabit_timeout" ->
      (overridedParams.getOrElse("rabit_timeout", -1).toString.toInt >= 0).toString,
    "rabit_timeout_sec" -> {
      if (overridedParams.getOrElse("rabit_timeout", -1).toString.toInt >= 0) {
        overridedParams.get("rabit_timeout").toString
      } else {
        "1800"
      }
    },
    "DMLC_WORKER_CONNECT_RETRY" ->
      overridedParams.getOrElse("dmlc_worker_connect_retry", 5).toString
  )
}

/**
 * Traing data group in a RDD partition.
 * @param groupId The group id
 * @param points Array of XGBLabeledPoint within the same group.
 * @param isEdgeGroup whether it is a frist or last group in a RDD partition.
 */
private[spark] case class XGBLabeledPointGroup(
    groupId: Int,
    points: Array[XGBLabeledPoint],
    isEdgeGroup: Boolean)

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")

  private def verifyMissingSetting(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float,
      allowNonZeroMissing: Boolean): Iterator[XGBLabeledPoint] = {
    if (missing != 0.0f && !allowNonZeroMissing) {
      xgbLabelPoints.map(labeledPoint => {
        if (labeledPoint.indices != null) {
            throw new RuntimeException(s"you can only specify missing value as 0.0 (the currently" +
              s" set value $missing) when you have SparseVector or Empty vector as your feature" +
              s" format. If you didn't use Spark's VectorAssembler class to build your feature " +
              s"vector but instead did so in a way that preserves zeros in your feature vector " +
              s"you can avoid this check by using the 'allow_non_zero_missing parameter'" +
              s" (only use if you know what you are doing)")
        }
        labeledPoint
      })
    } else {
      xgbLabelPoints
    }
  }

  private def removeMissingValues(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float,
      keepCondition: Float => Boolean): Iterator[XGBLabeledPoint] = {
    xgbLabelPoints.map { labeledPoint =>
      val indicesBuilder = new mutable.ArrayBuilder.ofInt()
      val valuesBuilder = new mutable.ArrayBuilder.ofFloat()
      for ((value, i) <- labeledPoint.values.zipWithIndex if keepCondition(value)) {
        indicesBuilder += (if (labeledPoint.indices == null) i else labeledPoint.indices(i))
        valuesBuilder += value
      }
      labeledPoint.copy(indices = indicesBuilder.result(), values = valuesBuilder.result())
    }
  }

  private[spark] def processMissingValues(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float,
      allowNonZeroMissing: Boolean): Iterator[XGBLabeledPoint] = {
    if (!missing.isNaN) {
      removeMissingValues(verifyMissingSetting(xgbLabelPoints, missing, allowNonZeroMissing),
        missing, (v: Float) => v != missing)
    } else {
      removeMissingValues(verifyMissingSetting(xgbLabelPoints, missing, allowNonZeroMissing),
        missing, (v: Float) => !v.isNaN)
    }
  }

  private def processMissingValuesWithGroup(
      xgbLabelPointGroups: Iterator[Array[XGBLabeledPoint]],
      missing: Float,
      allowNonZeroMissing: Boolean): Iterator[Array[XGBLabeledPoint]] = {
    if (!missing.isNaN) {
      xgbLabelPointGroups.map {
        labeledPoints => XGBoost.processMissingValues(
          labeledPoints.iterator,
          missing,
          allowNonZeroMissing
        ).toArray
      }
    } else {
      xgbLabelPointGroups
    }
  }

  private def getCacheDirName(useExternalMemory: Boolean): Option[String] = {
    val taskId = TaskContext.getPartitionId().toString
    if (useExternalMemory) {
      val dir = Files.createTempDirectory(s"${TaskContext.get().stageId()}-cache-$taskId")
      Some(dir.toAbsolutePath.toString)
    } else {
      None
    }
  }

  private def buildDistributedBooster(
      watches: Watches,
      xgbExecutionParam: XGBoostExecutionParams,
      rabitEnv: java.util.Map[String, String],
      obj: ObjectiveTrait,
      eval: EvalTrait,
      prevBooster: Booster): Iterator[(Booster, Map[String, Array[Float]])] = {
    // to workaround the empty partitions in training dataset,
    // this might not be the best efficient implementation, see
    // (https://github.com/dmlc/xgboost/issues/1277)
    if (watches.toMap("train").rowNum == 0) {
      throw new XGBoostError(
        s"detected an empty partition in the training data, partition ID:" +
          s" ${TaskContext.getPartitionId()}")
    }
    val taskId = TaskContext.getPartitionId().toString
    val attempt = TaskContext.get().attemptNumber.toString
    rabitEnv.put("DMLC_TASK_ID", taskId)
    rabitEnv.put("DMLC_NUM_ATTEMPT", attempt)
    rabitEnv.put("DMLC_WORKER_STOP_PROCESS_ON_ERROR", "false")
    val numRounds = xgbExecutionParam.numRounds
    val makeCheckpoint = xgbExecutionParam.checkpointParam.isDefined && taskId.toInt == 0
    try {
      Rabit.init(rabitEnv)
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
      watches.delete()
    }
  }

  private def startTracker(nWorkers: Int, trackerConf: TrackerConf): IRabitTracker = {
    val tracker: IRabitTracker = trackerConf.trackerImpl match {
      case "scala" => new RabitTracker(nWorkers)
      case "python" => new PyRabitTracker(nWorkers)
      case _ => new PyRabitTracker(nWorkers)
    }

    require(tracker.start(trackerConf.workerConnectionTimeout), "FAULT: Failed to start tracker")
    tracker
  }

  class IteratorWrapper[T](arrayOfXGBLabeledPoints: Array[(String, Iterator[T])])
    extends Iterator[(String, Iterator[T])] {

    private var currentIndex = 0

    override def hasNext: Boolean = currentIndex <= arrayOfXGBLabeledPoints.length - 1

    override def next(): (String, Iterator[T]) = {
      currentIndex += 1
      arrayOfXGBLabeledPoints(currentIndex - 1)
    }
  }

  private def coPartitionNoGroupSets(
      trainingData: RDD[XGBLabeledPoint],
      evalSets: Map[String, RDD[XGBLabeledPoint]],
      nWorkers: Int) = {
    // eval_sets is supposed to be set by the caller of [[trainDistributed]]
    val allDatasets = Map("train" -> trainingData) ++ evalSets
    val repartitionedDatasets = allDatasets.map{case (name, rdd) =>
      if (rdd.getNumPartitions != nWorkers) {
        (name, rdd.repartition(nWorkers))
      } else {
        (name, rdd)
      }
    }
    repartitionedDatasets.foldLeft(trainingData.sparkContext.parallelize(
      Array.fill[(String, Iterator[XGBLabeledPoint])](nWorkers)(null), nWorkers)){
      case (rddOfIterWrapper, (name, rddOfIter)) =>
        rddOfIterWrapper.zipPartitions(rddOfIter){
          (itrWrapper, itr) =>
            if (!itr.hasNext) {
              logger.error("when specifying eval sets as dataframes, you have to ensure that " +
                "the number of elements in each dataframe is larger than the number of workers")
              throw new Exception("too few elements in evaluation sets")
            }
            val itrArray = itrWrapper.toArray
            if (itrArray.head != null) {
              new IteratorWrapper(itrArray :+ (name -> itr))
            } else {
              new IteratorWrapper(Array(name -> itr))
            }
        }
    }
  }

  private def trainForNonRanking(
      trainingData: RDD[XGBLabeledPoint],
      xgbExecutionParams: XGBoostExecutionParams,
      rabitEnv: java.util.Map[String, String],
      prevBooster: Booster,
      evalSetsMap: Map[String, RDD[XGBLabeledPoint]]): RDD[(Booster, Map[String, Array[Float]])] = {
    if (evalSetsMap.isEmpty) {
      trainingData.mapPartitions(labeledPoints => {
        val watches = Watches.buildWatches(xgbExecutionParams,
          processMissingValues(labeledPoints, xgbExecutionParams.missing,
            xgbExecutionParams.allowNonZeroForMissing),
          getCacheDirName(xgbExecutionParams.useExternalMemory))
        buildDistributedBooster(watches, xgbExecutionParams, rabitEnv, xgbExecutionParams.obj,
          xgbExecutionParams.eval, prevBooster)
      }).cache()
    } else {
      coPartitionNoGroupSets(trainingData, evalSetsMap, xgbExecutionParams.numWorkers).
        mapPartitions {
          nameAndLabeledPointSets =>
            val watches = Watches.buildWatches(
              nameAndLabeledPointSets.map {
                case (name, iter) => (name, processMissingValues(iter,
                  xgbExecutionParams.missing, xgbExecutionParams.allowNonZeroForMissing))
              },
              getCacheDirName(xgbExecutionParams.useExternalMemory))
            buildDistributedBooster(watches, xgbExecutionParams, rabitEnv, xgbExecutionParams.obj,
              xgbExecutionParams.eval, prevBooster)
        }.cache()
    }
  }

  private def trainForRanking(
      trainingData: RDD[Array[XGBLabeledPoint]],
      xgbExecutionParam: XGBoostExecutionParams,
      rabitEnv: java.util.Map[String, String],
      prevBooster: Booster,
      evalSetsMap: Map[String, RDD[XGBLabeledPoint]]): RDD[(Booster, Map[String, Array[Float]])] = {
    if (evalSetsMap.isEmpty) {
      trainingData.mapPartitions(labeledPointGroups => {
        val watches = Watches.buildWatchesWithGroup(xgbExecutionParam,
          processMissingValuesWithGroup(labeledPointGroups, xgbExecutionParam.missing,
            xgbExecutionParam.allowNonZeroForMissing),
          getCacheDirName(xgbExecutionParam.useExternalMemory))
        buildDistributedBooster(watches, xgbExecutionParam, rabitEnv,
          xgbExecutionParam.obj, xgbExecutionParam.eval, prevBooster)
      }).cache()
    } else {
      coPartitionGroupSets(trainingData, evalSetsMap, xgbExecutionParam.numWorkers).mapPartitions(
        labeledPointGroupSets => {
          val watches = Watches.buildWatchesWithGroup(
            labeledPointGroupSets.map {
              case (name, iter) => (name, processMissingValuesWithGroup(iter,
                xgbExecutionParam.missing, xgbExecutionParam.allowNonZeroForMissing))
            },
            getCacheDirName(xgbExecutionParam.useExternalMemory))
          buildDistributedBooster(watches, xgbExecutionParam, rabitEnv,
            xgbExecutionParam.obj,
            xgbExecutionParam.eval,
            prevBooster)
        }).cache()
    }
  }

  private def cacheData(ifCacheDataBoolean: Boolean, input: RDD[_]): RDD[_] = {
    if (ifCacheDataBoolean) input.persist(StorageLevel.MEMORY_AND_DISK) else input
  }

  private def composeInputData(
    trainingData: RDD[XGBLabeledPoint],
    ifCacheDataBoolean: Boolean,
    hasGroup: Boolean,
    nWorkers: Int): Either[RDD[Array[XGBLabeledPoint]], RDD[XGBLabeledPoint]] = {
    if (hasGroup) {
      val repartitionedData = repartitionForTrainingGroup(trainingData, nWorkers)
      Left(cacheData(ifCacheDataBoolean, repartitionedData).
        asInstanceOf[RDD[Array[XGBLabeledPoint]]])
    } else {
      Right(cacheData(ifCacheDataBoolean, trainingData).asInstanceOf[RDD[XGBLabeledPoint]])
    }
  }

  /**
   * @return A tuple of the booster and the metrics used to build training summary
   */
  @throws(classOf[XGBoostError])
  private[spark] def trainDistributed(
      trainingData: RDD[XGBLabeledPoint],
      params: Map[String, Any],
      hasGroup: Boolean = false,
      evalSetsMap: Map[String, RDD[XGBLabeledPoint]] = Map()):
    (Booster, Map[String, Array[Float]]) = {
    logger.info(s"Running XGBoost ${spark.VERSION} with parameters:\n${params.mkString("\n")}")
    val xgbParamsFactory = new XGBoostExecutionParamsFactory(params, trainingData.sparkContext)
    val xgbExecParams = xgbParamsFactory.buildXGBRuntimeParams
    val sc = trainingData.sparkContext
    val transformedTrainingData = composeInputData(trainingData, xgbExecParams.cacheTrainingSet,
      hasGroup, xgbExecParams.numWorkers)
    val prevBooster = xgbExecParams.checkpointParam.map { checkpointParam =>
      val checkpointManager = new ExternalCheckpointManager(
        checkpointParam.checkpointPath,
        FileSystem.get(sc.hadoopConfiguration))
      checkpointManager.cleanUpHigherVersions(xgbExecParams.numRounds)
      checkpointManager.loadCheckpointAsScalaBooster()
    }.orNull
    try {
      // Train for every ${savingRound} rounds and save the partially completed booster
      val tracker = startTracker(xgbExecParams.numWorkers, xgbExecParams.trackerConf)
      val (booster, metrics) = try {
        val parallelismTracker = new SparkParallelismTracker(sc,
          xgbExecParams.timeoutRequestWorkers,
          xgbExecParams.numWorkers)
        val rabitEnv = tracker.getWorkerEnvs
        val boostersAndMetrics = if (hasGroup) {
          trainForRanking(transformedTrainingData.left.get, xgbExecParams, rabitEnv, prevBooster,
            evalSetsMap)
        } else {
          trainForNonRanking(transformedTrainingData.right.get, xgbExecParams, rabitEnv,
            prevBooster, evalSetsMap)
        }
        val sparkJobThread = new Thread() {
          override def run() {
            // force the job
            boostersAndMetrics.foreachPartition(() => _)
          }
        }
        sparkJobThread.setUncaughtExceptionHandler(tracker)
        sparkJobThread.start()
        val trackerReturnVal = parallelismTracker.execute(tracker.waitFor(0L))
        logger.info(s"Rabit returns with exit code $trackerReturnVal")
        val (booster, metrics) = postTrackerReturnProcessing(trackerReturnVal,
          boostersAndMetrics, sparkJobThread)
        (booster, metrics)
      } finally {
        tracker.stop()
      }
      // we should delete the checkpoint directory after a successful training
      xgbExecParams.checkpointParam.foreach {
        cpParam =>
          if (!xgbExecParams.checkpointParam.get.skipCleanCheckpoint) {
            val checkpointManager = new ExternalCheckpointManager(
              cpParam.checkpointPath,
              FileSystem.get(sc.hadoopConfiguration))
            checkpointManager.cleanPath()
          }
      }
      (booster, metrics)
    } catch {
      case t: Throwable =>
        // if the job was aborted due to an exception
        logger.error("the job was aborted due to ", t)
        trainingData.sparkContext.stop()
        throw t
    } finally {
      uncacheTrainingData(xgbExecParams.cacheTrainingSet, transformedTrainingData)
    }
  }

  private def uncacheTrainingData(
      cacheTrainingSet: Boolean,
      transformedTrainingData: Either[RDD[Array[XGBLabeledPoint]], RDD[XGBLabeledPoint]]): Unit = {
    if (cacheTrainingSet) {
      if (transformedTrainingData.isLeft) {
        transformedTrainingData.left.get.unpersist()
      } else {
        transformedTrainingData.right.get.unpersist()
      }
    }
  }

  private def aggByGroupInfo(trainingData: RDD[XGBLabeledPoint]) = {
    val normalGroups: RDD[Array[XGBLabeledPoint]] = trainingData.mapPartitions(
      // LabeledPointGroupIterator returns (Boolean, Array[XGBLabeledPoint])
      new LabeledPointGroupIterator(_)).filter(!_.isEdgeGroup).map(_.points)

    // edge groups with partition id.
    val edgeGroups: RDD[(Int, XGBLabeledPointGroup)] = trainingData.mapPartitions(
      new LabeledPointGroupIterator(_)).filter(_.isEdgeGroup).map(
      group => (TaskContext.getPartitionId(), group))

    // group chunks from different partitions together by group id in XGBLabeledPoint.
    // use groupBy instead of aggregateBy since all groups within a partition have unique group ids.
    val stitchedGroups: RDD[Array[XGBLabeledPoint]] = edgeGroups.groupBy(_._2.groupId).map(
      groups => {
        val it: Iterable[(Int, XGBLabeledPointGroup)] = groups._2
        // sorted by partition id and merge list of Array[XGBLabeledPoint] into one array
        it.toArray.sortBy(_._1).flatMap(_._2.points)
      })
    normalGroups.union(stitchedGroups)
  }

  private[spark] def repartitionForTrainingGroup(
      trainingData: RDD[XGBLabeledPoint], nWorkers: Int): RDD[Array[XGBLabeledPoint]] = {
    val allGroups = aggByGroupInfo(trainingData)
    logger.info(s"repartitioning training group set to $nWorkers partitions")
    allGroups.repartition(nWorkers)
  }

  private def coPartitionGroupSets(
      aggedTrainingSet: RDD[Array[XGBLabeledPoint]],
      evalSets: Map[String, RDD[XGBLabeledPoint]],
      nWorkers: Int): RDD[(String, Iterator[Array[XGBLabeledPoint]])] = {
    val repartitionedDatasets = Map("train" -> aggedTrainingSet) ++ evalSets.map {
      case (name, rdd) => {
        val aggedRdd = aggByGroupInfo(rdd)
        if (aggedRdd.getNumPartitions != nWorkers) {
          name -> aggedRdd.repartition(nWorkers)
        } else {
          name -> aggedRdd
        }
      }
    }
    repartitionedDatasets.foldLeft(aggedTrainingSet.sparkContext.parallelize(
      Array.fill[(String, Iterator[Array[XGBLabeledPoint]])](nWorkers)(null), nWorkers)){
      case (rddOfIterWrapper, (name, rddOfIter)) =>
        rddOfIterWrapper.zipPartitions(rddOfIter){
          (itrWrapper, itr) =>
            if (!itr.hasNext) {
              logger.error("when specifying eval sets as dataframes, you have to ensure that " +
                "the number of elements in each dataframe is larger than the number of workers")
              throw new Exception("too few elements in evaluation sets")
            }
            val itrArray = itrWrapper.toArray
            if (itrArray.head != null) {
              new IteratorWrapper(itrArray :+ (name -> itr))
            } else {
              new IteratorWrapper(Array(name -> itr))
            }
        }
    }
  }

  private def postTrackerReturnProcessing(
      trackerReturnVal: Int,
      distributedBoostersAndMetrics: RDD[(Booster, Map[String, Array[Float]])],
      sparkJobThread: Thread): (Booster, Map[String, Array[Float]]) = {
    if (trackerReturnVal == 0) {
      // Copies of the final booster and the corresponding metrics
      // reside in each partition of the `distributedBoostersAndMetrics`.
      // Any of them can be used to create the model.
      // it's safe to block here forever, as the tracker has returned successfully, and the Spark
      // job should have finished, there is no reason for the thread cannot return
      sparkJobThread.join()
      val (booster, metrics) = distributedBoostersAndMetrics.first()
      distributedBoostersAndMetrics.unpersist(false)
      (booster, metrics)
    } else {
      try {
        if (sparkJobThread.isAlive) {
          sparkJobThread.interrupt()
        }
      } catch {
        case _: InterruptedException =>
          logger.info("spark job thread is interrupted")
      }
      throw new XGBoostError("XGBoostModel training failed")
    }
  }

}

private class Watches private(
    val datasets: Array[DMatrix],
    val names: Array[String],
    val cacheDirName: Option[String]) {

  def toMap: Map[String, DMatrix] = {
    names.zip(datasets).toMap.filter { case (_, matrix) => matrix.rowNum > 0 }
  }

  def size: Int = toMap.size

  def delete(): Unit = {
    toMap.values.foreach(_.delete())
    cacheDirName.foreach { name =>
      FileUtils.deleteDirectory(new File(name))
    }
  }

  override def toString: String = toMap.toString
}

private object Watches {

  private def fromBaseMarginsToArray(baseMargins: Iterator[Float]): Option[Array[Float]] = {
    val builder = new mutable.ArrayBuilder.ofFloat()
    var nTotal = 0
    var nUndefined = 0
    while (baseMargins.hasNext) {
      nTotal += 1
      val baseMargin = baseMargins.next()
      if (baseMargin.isNaN) {
        nUndefined += 1  // don't waste space for all-NaNs.
      } else {
        builder += baseMargin
      }
    }
    if (nUndefined == nTotal) {
      None
    } else if (nUndefined == 0) {
      Some(builder.result())
    } else {
      throw new IllegalArgumentException(
        s"Encountered a partition with $nUndefined NaN base margin values. " +
          s"If you want to specify base margin, ensure all values are non-NaN.")
    }
  }

  def buildWatches(
      nameAndLabeledPointSets: Iterator[(String, Iterator[XGBLabeledPoint])],
      cachedDirName: Option[String]): Watches = {
    val dms = nameAndLabeledPointSets.map {
      case (name, labeledPoints) =>
        val baseMargins = new mutable.ArrayBuilder.ofFloat
        val duplicatedItr = labeledPoints.map(labeledPoint => {
          baseMargins += labeledPoint.baseMargin
          labeledPoint
        })
        val dMatrix = new DMatrix(duplicatedItr, cachedDirName.map(_ + s"/$name").orNull)
        val baseMargin = fromBaseMarginsToArray(baseMargins.result().iterator)
        if (baseMargin.isDefined) {
          dMatrix.setBaseMargin(baseMargin.get)
        }
        (name, dMatrix)
    }.toArray
    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName)
  }

  def buildWatches(
      xgbExecutionParams: XGBoostExecutionParams,
      labeledPoints: Iterator[XGBLabeledPoint],
      cacheDirName: Option[String]): Watches = {
    val trainTestRatio = xgbExecutionParams.xgbInputParams.trainTestRatio
    val seed = xgbExecutionParams.xgbInputParams.seed
    val r = new Random(seed)
    val testPoints = mutable.ArrayBuffer.empty[XGBLabeledPoint]
    val trainBaseMargins = new mutable.ArrayBuilder.ofFloat
    val testBaseMargins = new mutable.ArrayBuilder.ofFloat
    val trainPoints = labeledPoints.filter { labeledPoint =>
      val accepted = r.nextDouble() <= trainTestRatio
      if (!accepted) {
        testPoints += labeledPoint
        testBaseMargins += labeledPoint.baseMargin
      } else {
        trainBaseMargins += labeledPoint.baseMargin
      }
      accepted
    }
    val trainMatrix = new DMatrix(trainPoints, cacheDirName.map(_ + "/train").orNull)
    val testMatrix = new DMatrix(testPoints.iterator, cacheDirName.map(_ + "/test").orNull)

    val trainMargin = fromBaseMarginsToArray(trainBaseMargins.result().iterator)
    val testMargin = fromBaseMarginsToArray(testBaseMargins.result().iterator)
    if (trainMargin.isDefined) trainMatrix.setBaseMargin(trainMargin.get)
    if (testMargin.isDefined) testMatrix.setBaseMargin(testMargin.get)

    new Watches(Array(trainMatrix, testMatrix), Array("train", "test"), cacheDirName)
  }

  def buildWatchesWithGroup(
      nameAndlabeledPointGroupSets: Iterator[(String, Iterator[Array[XGBLabeledPoint]])],
      cachedDirName: Option[String]): Watches = {
    val dms = nameAndlabeledPointGroupSets.map {
      case (name, labeledPointsGroups) =>
        val baseMargins = new mutable.ArrayBuilder.ofFloat
        val groupsInfo = new mutable.ArrayBuilder.ofInt
        val weights = new mutable.ArrayBuilder.ofFloat
        val iter = labeledPointsGroups.filter(labeledPointGroup => {
          var groupWeight = -1.0f
          var groupSize = 0
          labeledPointGroup.map { labeledPoint => {
            if (groupWeight < 0) {
              groupWeight = labeledPoint.weight
            } else if (groupWeight != labeledPoint.weight) {
              throw new IllegalArgumentException("the instances in the same group have to be" +
                s" assigned with the same weight (unexpected weight ${labeledPoint.weight}")
            }
            baseMargins += labeledPoint.baseMargin
            groupSize += 1
            labeledPoint
          }
          }
          weights += groupWeight
          groupsInfo += groupSize
          true
        })
        val dMatrix = new DMatrix(iter.flatMap(_.iterator), cachedDirName.map(_ + s"/$name").orNull)
        val baseMargin = fromBaseMarginsToArray(baseMargins.result().iterator)
        if (baseMargin.isDefined) {
          dMatrix.setBaseMargin(baseMargin.get)
        }
        dMatrix.setGroup(groupsInfo.result())
        dMatrix.setWeight(weights.result())
        (name, dMatrix)
    }.toArray
    new Watches(dms.map(_._2), dms.map(_._1), cachedDirName)
  }

  def buildWatchesWithGroup(
      xgbExecutionParams: XGBoostExecutionParams,
      labeledPointGroups: Iterator[Array[XGBLabeledPoint]],
      cacheDirName: Option[String]): Watches = {
    val trainTestRatio = xgbExecutionParams.xgbInputParams.trainTestRatio
    val seed = xgbExecutionParams.xgbInputParams.seed
    val r = new Random(seed)
    val testPoints = mutable.ArrayBuilder.make[XGBLabeledPoint]
    val trainBaseMargins = new mutable.ArrayBuilder.ofFloat
    val testBaseMargins = new mutable.ArrayBuilder.ofFloat

    val trainGroups = new mutable.ArrayBuilder.ofInt
    val testGroups = new mutable.ArrayBuilder.ofInt

    val trainWeights = new mutable.ArrayBuilder.ofFloat
    val testWeights = new mutable.ArrayBuilder.ofFloat

    val trainLabelPointGroups = labeledPointGroups.filter { labeledPointGroup =>
      val accepted = r.nextDouble() <= trainTestRatio
      if (!accepted) {
        var groupWeight = -1.0f
        var groupSize = 0
        labeledPointGroup.foreach(labeledPoint => {
          testPoints += labeledPoint
          testBaseMargins += labeledPoint.baseMargin
          if (groupWeight < 0) {
            groupWeight = labeledPoint.weight
          } else if (labeledPoint.weight != groupWeight) {
            throw new IllegalArgumentException("the instances in the same group have to be" +
              s" assigned with the same weight (unexpected weight ${labeledPoint.weight}")
          }
          groupSize += 1
        })
        testWeights += groupWeight
        testGroups += groupSize
      } else {
        var groupWeight = -1.0f
        var groupSize = 0
        labeledPointGroup.foreach { labeledPoint => {
          if (groupWeight < 0) {
            groupWeight = labeledPoint.weight
          } else if (labeledPoint.weight != groupWeight) {
            throw new IllegalArgumentException("the instances in the same group have to be" +
              s" assigned with the same weight (unexpected weight ${labeledPoint.weight}")
          }
          trainBaseMargins += labeledPoint.baseMargin
          groupSize += 1
        }}
        trainWeights += groupWeight
        trainGroups += groupSize
      }
      accepted
    }

    val trainPoints = trainLabelPointGroups.flatMap(_.iterator)
    val trainMatrix = new DMatrix(trainPoints, cacheDirName.map(_ + "/train").orNull)
    trainMatrix.setGroup(trainGroups.result())
    trainMatrix.setWeight(trainWeights.result())

    val testMatrix = new DMatrix(testPoints.result().iterator, cacheDirName.map(_ + "/test").orNull)
    if (trainTestRatio < 1.0) {
      testMatrix.setGroup(testGroups.result())
      testMatrix.setWeight(testWeights.result())
    }

    val trainMargin = fromBaseMarginsToArray(trainBaseMargins.result().iterator)
    val testMargin = fromBaseMarginsToArray(testBaseMargins.result().iterator)
    if (trainMargin.isDefined) trainMatrix.setBaseMargin(trainMargin.get)
    if (testMargin.isDefined) testMatrix.setBaseMargin(testMargin.get)

    new Watches(Array(trainMatrix, testMatrix), Array("train", "test"), cacheDirName)
  }
}

/**
 * Within each RDD partition, group the <code>XGBLabeledPoint</code> by group id.</p>
 * And the first and the last groups may not have all the items due to the data partition.
 * <code>LabeledPointGroupIterator</code> orginaizes data in a tuple format:
 * (isFistGroup || isLastGroup, Array[XGBLabeledPoint]).</p>
 * The edge groups across partitions can be stitched together later.
 * @param base collection of <code>XGBLabeledPoint</code>
 */
private[spark] class LabeledPointGroupIterator(base: Iterator[XGBLabeledPoint])
  extends AbstractIterator[XGBLabeledPointGroup] {

  private var firstPointOfNextGroup: XGBLabeledPoint = null
  private var isNewGroup = false

  override def hasNext: Boolean = {
    base.hasNext || isNewGroup
  }

  override def next(): XGBLabeledPointGroup = {
    val builder = mutable.ArrayBuilder.make[XGBLabeledPoint]
    var isFirstGroup = true
    if (firstPointOfNextGroup != null) {
      builder += firstPointOfNextGroup
      isFirstGroup = false
    }

    isNewGroup = false
    while (!isNewGroup && base.hasNext) {
      val point = base.next()
      val groupId = if (firstPointOfNextGroup != null) firstPointOfNextGroup.group else point.group
      firstPointOfNextGroup = point
      if (point.group == groupId) {
        // add to current group
        builder += point
      } else {
        // start a new group
        isNewGroup = true
      }
    }

    val isLastGroup = !isNewGroup
    val result = builder.result()
    val group = XGBLabeledPointGroup(result(0).group, result, isFirstGroup || isLastGroup)

    group
  }
}

