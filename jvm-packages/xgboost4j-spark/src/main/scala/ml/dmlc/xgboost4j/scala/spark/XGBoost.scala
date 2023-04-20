/*
 Copyright (c) 2014-2023 by Contributors

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

import scala.collection.mutable
import scala.util.Random
import scala.collection.JavaConverters._

import ml.dmlc.xgboost4j.java.{Communicator, IRabitTracker, XGBoostError, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.spark.params.LearningTaskParams
import ml.dmlc.xgboost4j.scala.ExternalCheckpointManager
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import org.apache.commons.io.FileUtils
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.FileSystem

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.sql.SparkSession

/**
 * Rabit tracker configurations.
 *
 * @param workerConnectionTimeout The timeout for all workers to connect to the tracker.
 *                                Set timeout length to zero to disable timeout.
 *                                Use a finite, non-zero timeout value to prevent tracker from
 *                                hanging indefinitely (in milliseconds)
 *                                (supported by "scala" implementation only.)
 * @param hostIp The Rabit Tracker host IP address which is only used for python implementation.
 *               This is only needed if the host IP cannot be automatically guessed.
 * @param pythonExec The python executed path for Rabit Tracker,
 *                   which is only used for python implementation.
 */
case class TrackerConf(workerConnectionTimeout: Long,
  hostIp: String = "", pythonExec: String = "")

object TrackerConf {
  def apply(): TrackerConf = TrackerConf(0L)
}

private[scala] case class XGBoostExecutionEarlyStoppingParams(numEarlyStoppingRounds: Int,
                                                             maximizeEvalMetrics: Boolean)

private[scala] case class XGBoostExecutionInputParams(trainTestRatio: Double, seed: Long)

private[scala] case class XGBoostExecutionParams(
    numWorkers: Int,
    numRounds: Int,
    useExternalMemory: Boolean,
    obj: ObjectiveTrait,
    eval: EvalTrait,
    missing: Float,
    allowNonZeroForMissing: Boolean,
    trackerConf: TrackerConf,
    checkpointParam: Option[ExternalCheckpointParams],
    xgbInputParams: XGBoostExecutionInputParams,
    earlyStoppingParams: XGBoostExecutionEarlyStoppingParams,
    cacheTrainingSet: Boolean,
    treeMethod: Option[String],
    isLocal: Boolean) {

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

  private val isLocal = sc.isLocal

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
      if (overridedParams.getOrElse("custom_eval", null) != null) {
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
    val useExternalMemory = overridedParams
      .getOrElse("use_external_memory", false).asInstanceOf[Boolean]
    val obj = overridedParams.getOrElse("custom_obj", null).asInstanceOf[ObjectiveTrait]
    val eval = overridedParams.getOrElse("custom_eval", null).asInstanceOf[EvalTrait]
    val missing = overridedParams.getOrElse("missing", Float.NaN).asInstanceOf[Float]
    val allowNonZeroForMissing = overridedParams
                                 .getOrElse("allow_non_zero_for_missing", false)
                                 .asInstanceOf[Boolean]
    validateSparkSslConf
    var treeMethod: Option[String] = None
    if (overridedParams.contains("tree_method")) {
      require(overridedParams("tree_method") == "hist" ||
        overridedParams("tree_method") == "approx" ||
        overridedParams("tree_method") == "auto" ||
        overridedParams("tree_method") == "gpu_hist", "xgboost4j-spark only supports tree_method" +
        " as 'hist', 'approx', 'gpu_hist', and 'auto'")
      treeMethod = Some(overridedParams("tree_method").asInstanceOf[String])
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
      checkpointParam,
      inputParams,
      xgbExecEarlyStoppingParams,
      cacheTrainingSet,
      treeMethod,
      isLocal)
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

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")

  def getGPUAddrFromResources: Int = {
    val tc = TaskContext.get()
    if (tc == null) {
      throw new RuntimeException("Something wrong for task context")
    }
    val resources = tc.resources()
    if (resources.contains("gpu")) {
      val addrs = resources("gpu").addresses
      if (addrs.size > 1) {
        // TODO should we throw exception ?
        logger.warn("XGBoost only supports 1 gpu per worker")
      }
      // take the first one
      addrs.head.toInt
    } else {
      throw new RuntimeException("gpu is not allocated by spark, " +
        "please check if gpu scheduling is enabled")
    }
  }

  private def buildWatchesAndCheck(buildWatchesFun: () => Watches): Watches = {
    val watches = buildWatchesFun()
    // to workaround the empty partitions in training dataset,
    // this might not be the best efficient implementation, see
    // (https://github.com/dmlc/xgboost/issues/1277)
    if (!watches.toMap.contains("train")) {
      throw new XGBoostError(
        s"detected an empty partition in the training data, partition ID:" +
          s" ${TaskContext.getPartitionId()}")
    }
    watches
  }

  private def buildDistributedBooster(
      buildWatches: () => Watches,
      xgbExecutionParam: XGBoostExecutionParams,
      rabitEnv: java.util.Map[String, String],
      obj: ObjectiveTrait,
      eval: EvalTrait,
      prevBooster: Booster): Iterator[(Booster, Map[String, Array[Float]])] = {

    var watches: Watches = null
    val taskId = TaskContext.getPartitionId().toString
    val attempt = TaskContext.get().attemptNumber.toString
    rabitEnv.put("DMLC_TASK_ID", taskId)
    rabitEnv.put("DMLC_NUM_ATTEMPT", attempt)
    val numRounds = xgbExecutionParam.numRounds
    val makeCheckpoint = xgbExecutionParam.checkpointParam.isDefined && taskId.toInt == 0

    try {
      Communicator.init(rabitEnv)

      watches = buildWatchesAndCheck(buildWatches)

      val numEarlyStoppingRounds = xgbExecutionParam.earlyStoppingParams.numEarlyStoppingRounds
      val metrics = Array.tabulate(watches.size)(_ => Array.ofDim[Float](numRounds))
      val externalCheckpointParams = xgbExecutionParam.checkpointParam

      var params = xgbExecutionParam.toMap
      if (xgbExecutionParam.treeMethod.exists(m => m == "gpu_hist")) {
        val gpuId = if (xgbExecutionParam.isLocal) {
          // For local mode, force gpu id to primary device
          0
        } else {
          getGPUAddrFromResources
        }
        logger.info("Leveraging gpu device " + gpuId + " to train")
        params = params + ("gpu_id" -> gpuId)
      }
      val booster = if (makeCheckpoint) {
        SXGBoost.trainAndSaveCheckpoint(
          watches.toMap("train"), params, numRounds,
          watches.toMap, metrics, obj, eval,
          earlyStoppingRound = numEarlyStoppingRounds, prevBooster, externalCheckpointParams)
      } else {
        SXGBoost.train(watches.toMap("train"), params, numRounds,
          watches.toMap, metrics, obj, eval,
          earlyStoppingRound = numEarlyStoppingRounds, prevBooster)
      }
      if (TaskContext.get().partitionId() == 0) {
        Iterator(booster -> watches.toMap.keys.zip(metrics).toMap)
      } else {
        Iterator.empty
      }
    } catch {
      case xgbException: XGBoostError =>
        logger.error(s"XGBooster worker $taskId has failed $attempt times due to ", xgbException)
        throw xgbException
    } finally {
      Communicator.shutdown()
      if (watches != null) watches.delete()
    }
  }

  /** visiable for testing */
  private[scala] def getTracker(nWorkers: Int, trackerConf: TrackerConf): IRabitTracker = {
    val tracker: IRabitTracker = new PyRabitTracker(
      nWorkers, trackerConf.hostIp, trackerConf.pythonExec
    )
    tracker
  }

  private def startTracker(nWorkers: Int, trackerConf: TrackerConf): IRabitTracker = {
    val tracker = getTracker(nWorkers, trackerConf)
    require(tracker.start(trackerConf.workerConnectionTimeout), "FAULT: Failed to start tracker")
    tracker
  }

  /**
   * @return A tuple of the booster and the metrics used to build training summary
   */
  @throws(classOf[XGBoostError])
  private[spark] def trainDistributed(
      sc: SparkContext,
      buildTrainingData: XGBoostExecutionParams => (RDD[() => Watches], Option[RDD[_]]),
      params: Map[String, Any]):
    (Booster, Map[String, Array[Float]]) = {

    logger.info(s"Running XGBoost ${spark.VERSION} with parameters:\n${params.mkString("\n")}")

    val xgbParamsFactory = new XGBoostExecutionParamsFactory(params, sc)
    val xgbExecParams = xgbParamsFactory.buildXGBRuntimeParams
    val xgbRabitParams = xgbParamsFactory.buildRabitParams.asJava

    val prevBooster = xgbExecParams.checkpointParam.map { checkpointParam =>
      val checkpointManager = new ExternalCheckpointManager(
        checkpointParam.checkpointPath,
        FileSystem.get(sc.hadoopConfiguration))
      checkpointManager.cleanUpHigherVersions(xgbExecParams.numRounds)
      checkpointManager.loadCheckpointAsScalaBooster()
    }.orNull

    // Get the training data RDD and the cachedRDD
    val (trainingRDD, optionalCachedRDD) = buildTrainingData(xgbExecParams)

    try {
      // Train for every ${savingRound} rounds and save the partially completed booster
      val tracker = startTracker(xgbExecParams.numWorkers, xgbExecParams.trackerConf)
      val (booster, metrics) = try {
        tracker.getWorkerEnvs().putAll(xgbRabitParams)
        val rabitEnv = tracker.getWorkerEnvs

        val boostersAndMetrics = trainingRDD.barrier().mapPartitions { iter => {
          var optionWatches: Option[() => Watches] = None

          // take the first Watches to train
          if (iter.hasNext) {
            optionWatches = Some(iter.next())
          }

          optionWatches.map { buildWatches => buildDistributedBooster(buildWatches,
            xgbExecParams, rabitEnv, xgbExecParams.obj, xgbExecParams.eval, prevBooster)}
            .getOrElse(throw new RuntimeException("No Watches to train"))

        }}

        val (booster, metrics) = boostersAndMetrics.collect()(0)
        val trackerReturnVal = tracker.waitFor(0L)
        logger.info(s"Rabit returns with exit code $trackerReturnVal")
        if (trackerReturnVal != 0) {
          throw new XGBoostError("XGBoostModel training failed.")
        }
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
        throw t
    } finally {
      optionalCachedRDD.foreach(_.unpersist())
    }
  }

}

class Watches private[scala] (
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
