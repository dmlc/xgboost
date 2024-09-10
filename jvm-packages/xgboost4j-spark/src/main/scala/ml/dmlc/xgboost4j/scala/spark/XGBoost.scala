/*
 Copyright (c) 2014-2024 by Contributors

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

import org.apache.commons.io.FileUtils
import org.apache.commons.logging.LogFactory
import org.apache.spark.{SparkConf, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.resource.{ResourceProfileBuilder, TaskResourceRequests}

import ml.dmlc.xgboost4j.java.{Communicator, RabitTracker}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}

private[spark] case class RuntimeParams(
    numWorkers: Int,
    numRounds: Int,
    trackerConf: TrackerConf,
    earlyStoppingRounds: Int,
    device: String,
    isLocal: Boolean,
    runOnGpu: Boolean,
    obj: Option[ObjectiveTrait] = None,
    eval: Option[EvalTrait] = None)

/**
 * A trait to manage stage-level scheduling
 */
private[spark] trait StageLevelScheduling extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")

  private[spark] def isStandaloneOrLocalCluster(conf: SparkConf): Boolean = {
    val master = conf.get("spark.master")
    master != null && (master.startsWith("spark://") || master.startsWith("local-cluster"))
  }

  /**
   * To determine if stage-level scheduling should be skipped according to the spark version
   * and spark configurations
   *
   * @param sparkVersion spark version
   * @param runOnGpu     if xgboost training run on GPUs
   * @param conf         spark configurations
   * @return Boolean to skip stage-level scheduling or not
   */
  private[spark] def skipStageLevelScheduling(sparkVersion: String,
                                              runOnGpu: Boolean,
                                              conf: SparkConf): Boolean = {
    if (runOnGpu) {
      if (sparkVersion < "3.4.0") {
        logger.info("Stage-level scheduling in xgboost requires spark version 3.4.0+")
        return true
      }

      if (!isStandaloneOrLocalCluster(conf)) {
        logger.info("Stage-level scheduling in xgboost requires spark standalone or " +
          "local-cluster mode")
        return true
      }

      val executorCores = conf.getInt("spark.executor.cores", -1)
      val executorGpus = conf.getInt("spark.executor.resource.gpu.amount", -1)
      if (executorCores == -1 || executorGpus == -1) {
        logger.info("Stage-level scheduling in xgboost requires spark.executor.cores, " +
          "spark.executor.resource.gpu.amount to be set.")
        return true
      }

      if (executorCores == 1) {
        logger.info("Stage-level scheduling in xgboost requires spark.executor.cores > 1")
        return true
      }

      if (executorGpus > 1) {
        logger.info("Stage-level scheduling in xgboost will not work " +
          "when spark.executor.resource.gpu.amount > 1")
        return true
      }

      val taskGpuAmount = conf.getDouble("spark.task.resource.gpu.amount", -1.0).toFloat

      if (taskGpuAmount == -1.0) {
        // The ETL tasks will not grab a gpu when spark.task.resource.gpu.amount is not set,
        // but with stage-level scheduling, we can make training task grab the gpu.
        return false
      }

      if (taskGpuAmount == executorGpus.toFloat) {
        // spark.executor.resource.gpu.amount = spark.task.resource.gpu.amount
        // results in only 1 task running at a time, which may cause perf issue.
        return true
      }
      // We can enable stage-level scheduling
      false
    } else true // Skip stage-level scheduling for cpu training.
  }

  /**
   * Attempt to modify the task resources so that only one task can be executed
   * on a single executor simultaneously.
   *
   * @param sc  the spark context
   * @param rdd the rdd to be applied with new resource profile
   * @return the original rdd or the modified rdd
   */
  private[spark] def tryStageLevelScheduling[T](sc: SparkContext,
                                                xgbExecParams: RuntimeParams,
                                                rdd: RDD[T]
                                               ): RDD[T] = {

    val conf = sc.getConf
    if (skipStageLevelScheduling(sc.version, xgbExecParams.runOnGpu, conf)) {
      return rdd
    }

    // Ensure executor_cores is not None
    val executor_cores = conf.getInt("spark.executor.cores", -1)
    if (executor_cores == -1) {
      throw new RuntimeException("Wrong spark.executor.cores")
    }

    // Spark-rapids is a GPU-acceleration project for Spark SQL.
    // When spark-rapids is enabled, we prevent concurrent execution of other ETL tasks
    // that utilize GPUs alongside training tasks in order to avoid GPU out-of-memory errors.
    val spark_plugins = conf.get("spark.plugins", " ")
    val spark_rapids_sql_enabled = conf.get("spark.rapids.sql.enabled", "true")

    // Determine the number of cores required for each task.
    val task_cores = if (spark_plugins.contains("com.nvidia.spark.SQLPlugin") &&
      spark_rapids_sql_enabled.toLowerCase == "true") {
      executor_cores
    } else {
      (executor_cores / 2) + 1
    }

    // Each training task requires cpu cores > total executor cores//2 + 1 to
    // ensure tasks are sent to different executors.
    // Note: We cannot use GPUs to limit concurrent tasks
    // due to https://issues.apache.org/jira/browse/SPARK-45527.
    val task_gpus = 1.0
    val treqs = new TaskResourceRequests().cpus(task_cores).resource("gpu", task_gpus)
    val rp = new ResourceProfileBuilder().require(treqs).build()

    logger.info(s"XGBoost training tasks require the resource(cores=$task_cores, gpu=$task_gpus).")
    rdd.withResources(rp)
  }
}

private[spark] object XGBoost extends StageLevelScheduling {
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


  /**
   * Train a XGBoost Boost on the dataset in the Watches
   *
   * @param watches       holds the dataset to be trained
   * @param runtimeParams XGBoost runtime parameters
   * @param xgboostParams XGBoost library paramters
   * @return a booster and the metrics
   */
  private def trainBooster(watches: Watches,
                           runtimeParams: RuntimeParams,
                           xgboostParams: Map[String, Any]
                          ): (Booster, Array[Array[Float]]) = {

    val numEarlyStoppingRounds = runtimeParams.earlyStoppingRounds
    val metrics = Array.tabulate(watches.size)(_ =>
      Array.ofDim[Float](runtimeParams.numRounds))

    var params = xgboostParams
    if (runtimeParams.runOnGpu) {
      val gpuId = if (runtimeParams.isLocal) {
        TaskContext.get().partitionId() % runtimeParams.numWorkers
      } else {
        getGPUAddrFromResources
      }
      logger.info("Leveraging gpu device " + gpuId + " to train")
      params = params + ("device" -> s"cuda:$gpuId")
    }
    val booster = SXGBoost.train(watches.toMap("train"), params, runtimeParams.numRounds,
      watches.toMap, metrics, runtimeParams.obj.getOrElse(null),
      runtimeParams.eval.getOrElse(null), earlyStoppingRound = numEarlyStoppingRounds)
    (booster, metrics)
  }

  /**
   * Train a XGBoost booster with parameters on the dataset
   *
   * @param input         the input dataset for training
   * @param runtimeParams the runtime parameters for jvm
   * @param xgboostParams the xgboost parameters to pass to xgboost library
   * @return the booster and the metrics
   */
  def train(input: RDD[Watches],
            runtimeParams: RuntimeParams,
            xgboostParams: Map[String, Any]): (Booster, Map[String, Array[Float]]) = {

    val sc = input.sparkContext
    logger.info(s"Running XGBoost ${spark.VERSION} with parameters: $xgboostParams")

    // TODO Rabit tracker exception handling.
    val trackerConf = runtimeParams.trackerConf

    val tracker = new RabitTracker(runtimeParams.numWorkers,
      trackerConf.hostIp, trackerConf.port, trackerConf.timeout)
    require(tracker.start(), "FAULT: Failed to start tracker")

    try {
      val rabitEnv = tracker.getWorkerArgs()

      val boostersAndMetrics = input.barrier().mapPartitions { iter =>
        val partitionId = TaskContext.getPartitionId()
        rabitEnv.put("DMLC_TASK_ID", partitionId.toString)
        try {
          Communicator.init(rabitEnv)
          require(iter.hasNext, "Failed to create DMatrix")
          val watches = iter.next()
          try {
            val (booster, metrics) = trainBooster(watches, runtimeParams, xgboostParams)
            if (partitionId == 0) {
              Iterator(booster -> watches.toMap.keys.zip(metrics).toMap)
            } else {
              Iterator.empty
            }
          } finally {
            if (watches != null) {
              watches.delete()
            }
          }
        } finally {
          // If shutdown throws exception, then the real exception for
          // training will be swallowed,
          try {
            Communicator.shutdown()
          } catch {
            case e: Throwable =>
              logger.error("Communicator.shutdown error: ", e)
          }
        }
      }

      val rdd = tryStageLevelScheduling(sc, runtimeParams, boostersAndMetrics)
      // The repartition step is to make training stage as ShuffleMapStage, so that when one
      // of the training task fails the training stage can retry. ResultStage won't retry when
      // it fails.
      val (booster, metrics) = rdd.repartition(1).collect()(0)
      (booster, metrics)
    } catch {
      case t: Throwable =>
        // if the job was aborted due to an exception
        logger.error("XGBoost job was aborted due to ", t)
        throw t
    } finally {
      try {
        tracker.stop()
      } catch {
        case t: Throwable => logger.error(t)
      }
    }
  }
}

class Watches private[scala](val datasets: Array[DMatrix],
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

/**
 * Rabit tracker configurations.
 *
 * @param timeout The number of seconds before timeout waiting for workers to connect. and
 *                for the tracker to shutdown.
 * @param hostIp  The Rabit Tracker host IP address.
 *                This is only needed if the host IP cannot be automatically guessed.
 * @param port    The port number for the tracker to listen to. Use a system allocated one by
 *                default.
 */
private[spark] case class TrackerConf(timeout: Int = 0, hostIp: String = "", port: Int = 0)
