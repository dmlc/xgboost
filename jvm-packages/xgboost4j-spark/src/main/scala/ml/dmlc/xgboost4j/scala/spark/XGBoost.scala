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

import scala.collection.mutable
import scala.util.Random

import ml.dmlc.xgboost4j.java.{IRabitTracker, Rabit, XGBoostError, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.RabitTracker
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}

import org.apache.commons.io.FileUtils
import org.apache.commons.logging.LogFactory
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkParallelismTracker, TaskContext}
import org.apache.spark.sql.SparkSession


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

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")

  private[spark] def removeMissingValues(
      xgbLabelPoints: Iterator[XGBLabeledPoint],
      missing: Float): Iterator[XGBLabeledPoint] = {
    if (!missing.isNaN) {
      xgbLabelPoints.map { labeledPoint =>
        val indicesBuilder = new mutable.ArrayBuilder.ofInt()
        val valuesBuilder = new mutable.ArrayBuilder.ofFloat()
        for ((value, i) <- labeledPoint.values.zipWithIndex if value != missing) {
          indicesBuilder += (if (labeledPoint.indices == null) i else labeledPoint.indices(i))
          valuesBuilder += value
        }
        labeledPoint.copy(indices = indicesBuilder.result(), values = valuesBuilder.result())
      }
    } else {
      xgbLabelPoints
    }
  }

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

  private[spark] def buildDistributedBoosters(
      data: RDD[XGBLabeledPoint],
      params: Map[String, Any],
      rabitEnv: java.util.Map[String, String],
      round: Int,
      obj: ObjectiveTrait,
      eval: EvalTrait,
      useExternalMemory: Boolean,
      missing: Float,
      prevBooster: Booster
    ): RDD[(Booster, Map[String, Array[Float]])] = {

    val partitionedBaseMargin = data.map(_.baseMargin)
    // to workaround the empty partitions in training dataset,
    // this might not be the best efficient implementation, see
    // (https://github.com/dmlc/xgboost/issues/1277)
    data.zipPartitions(partitionedBaseMargin) { (labeledPoints, baseMargins) =>
      if (labeledPoints.isEmpty) {
        throw new XGBoostError(
          s"detected an empty partition in the training data, partition ID:" +
            s" ${TaskContext.getPartitionId()}")
      }
      val taskId = TaskContext.getPartitionId().toString
      val cacheDirName = if (useExternalMemory) {
        val dir = Files.createTempDirectory(s"${TaskContext.get().stageId()}-cache-$taskId")
        Some(dir.toAbsolutePath.toString)
      } else {
        None
      }
      rabitEnv.put("DMLC_TASK_ID", taskId)
      Rabit.init(rabitEnv)
      val watches = Watches(params,
        removeMissingValues(labeledPoints, missing),
        fromBaseMarginsToArray(baseMargins), cacheDirName)

      try {
        val numEarlyStoppingRounds = params.get("num_early_stopping_rounds")
            .map(_.toString.toInt).getOrElse(0)
        val metrics = Array.tabulate(watches.size)(_ => Array.ofDim[Float](round))
        val booster = SXGBoost.train(watches.train, params, round,
          watches.toMap, metrics, obj, eval,
          earlyStoppingRound = numEarlyStoppingRounds, prevBooster)
        Iterator(booster -> watches.toMap.keys.zip(metrics).toMap)
      } finally {
        Rabit.shutdown()
        watches.delete()
      }
    }.cache()
  }

  private def overrideParamsAccordingToTaskCPUs(
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
      overridedParams = params + ("nthread" -> coresPerTask)
    }
    overridedParams
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

  /**
   * @return A tuple of the booster and the metrics used to build training summary
   */
  @throws(classOf[XGBoostError])
  private[spark] def trainDistributed(
      trainingData: RDD[XGBLabeledPoint],
      params: Map[String, Any],
      round: Int,
      nWorkers: Int,
      obj: ObjectiveTrait = null,
      eval: EvalTrait = null,
      useExternalMemory: Boolean = false,
      missing: Float = Float.NaN): (Booster, Map[String, Array[Float]]) = {
    if (trainingData.context.getConf.getBoolean("spark.ssl.enabled", false)) {
      if (trainingData.context.getConf.getBoolean("spark.xgboost.ignoreSsl", false)) {
        this.logWarning(s"$uid: spark-xgboost is being run without encrypting data in " +
          s"transit!  Spark Conf spark.ssl.enabled=true was overridden with " +
          s"spark.xgboost.ignoreSsl=true.")
      } else {
        throw new Exception("xgboost-spark found spark.ssl.enabled=true to encrypt data " +
          "in transit, but xgboost-spark uses MPI to send non-encrypted data over the wire.  " +
          "To override this protection and still use xgboost-spark at your own risk, " +
          "you can set the SparkSession conf to use spark.xgboost.ignoreSsl=true.")
      }
    }
    if (params.contains("tree_method")) {
      require(params("tree_method") != "hist", "xgboost4j-spark does not support fast histogram" +
        " for now")
    }
    require(nWorkers > 0, "you must specify more than 0 workers")
    if (obj != null) {
      require(params.get("obj_type").isDefined, "parameter \"obj_type\" is not defined," +
        " you have to specify the objective type as classification or regression with a" +
        " customized objective function")
    }
    val trackerConf = params.get("tracker_conf") match {
      case None => TrackerConf()
      case Some(conf: TrackerConf) => conf
      case _ => throw new IllegalArgumentException("parameter \"tracker_conf\" must be an " +
        "instance of TrackerConf.")
    }
    val timeoutRequestWorkers: Long = params.get("timeout_request_workers") match {
      case None => 0L
      case Some(interval: Long) => interval
      case _ => throw new IllegalArgumentException("parameter \"timeout_request_workers\" must be" +
        " an instance of Long.")
    }
    val (checkpointPath, checkpointInterval) = CheckpointManager.extractParams(params)
    val partitionedData = repartitionForTraining(trainingData, nWorkers)

    val sc = trainingData.sparkContext
    val checkpointManager = new CheckpointManager(sc, checkpointPath)
    checkpointManager.cleanUpHigherVersions(round)

    var prevBooster = checkpointManager.loadCheckpointAsBooster
    // Train for every ${savingRound} rounds and save the partially completed booster
    checkpointManager.getCheckpointRounds(checkpointInterval, round).map {
      checkpointRound: Int =>
        val tracker = startTracker(nWorkers, trackerConf)
        try {
          val overriddenParams = overrideParamsAccordingToTaskCPUs(params, sc)
          val parallelismTracker = new SparkParallelismTracker(sc, timeoutRequestWorkers, nWorkers)
          val boostersAndMetrics = buildDistributedBoosters(partitionedData, overriddenParams,
            tracker.getWorkerEnvs, checkpointRound, obj, eval, useExternalMemory, missing,
            prevBooster)
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
          val (booster, metrics) = postTrackerReturnProcessing(trackerReturnVal, boostersAndMetrics,
            sparkJobThread)
          if (checkpointRound < round) {
            prevBooster = booster
            checkpointManager.updateCheckpoint(prevBooster)
          }
          (booster, metrics)
      } finally {
        tracker.stop()
      }
    }.last
  }


  private[spark] def repartitionForTraining(trainingData: RDD[XGBLabeledPoint], nWorkers: Int) = {
    if (trainingData.getNumPartitions != nWorkers) {
      logger.info(s"repartitioning training set to $nWorkers partitions")
      trainingData.repartition(nWorkers)
    } else {
      trainingData
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
  val train: DMatrix,
  val test: DMatrix,
  private val cacheDirName: Option[String]) {

  def toMap: Map[String, DMatrix] = Map("train" -> train, "test" -> test)
    .filter { case (_, matrix) => matrix.rowNum > 0 }

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

  def buildGroups(groups: Seq[Int]): Seq[Int] = {
    val output = mutable.ArrayBuffer.empty[Int]
    var count = 1
    var lastGroup = groups.head
    for (group <- groups.tail) {
      if (group != lastGroup) {
        lastGroup = group
        output += count
        count = 1
      } else {
        count += 1
      }
    }
    output += count
    output
  }

  def apply(
      params: Map[String, Any],
      labeledPoints: Iterator[XGBLabeledPoint],
      baseMarginsOpt: Option[Array[Float]],
      cacheDirName: Option[String]): Watches = {
    val trainTestRatio = params.get("train_test_ratio").map(_.toString.toDouble).getOrElse(1.0)
    val seed = params.get("seed").map(_.toString.toLong).getOrElse(System.nanoTime())
    val r = new Random(seed)
    val testPoints = mutable.ArrayBuffer.empty[XGBLabeledPoint]
    val trainPoints = labeledPoints.filter { labeledPoint =>
      val accepted = r.nextDouble() <= trainTestRatio
      if (!accepted) {
        testPoints += labeledPoint
      }

      accepted
    }

    val (trainIter1, trainIter2) = trainPoints.duplicate
    val trainMatrix = new DMatrix(trainIter1, cacheDirName.map(_ + "/train").orNull)
    val trainGroups = buildGroups(trainIter2.map(_.group).toSeq).toArray
    trainMatrix.setGroup(trainGroups)

    val testMatrix = new DMatrix(testPoints.iterator, cacheDirName.map(_ + "/test").orNull)
    if (trainTestRatio < 1.0) {
      val testGroups = buildGroups(testPoints.map(_.group)).toArray
      testMatrix.setGroup(testGroups)
    }

    r.setSeed(seed)
    for (baseMargins <- baseMarginsOpt) {
      val (trainMargin, testMargin) = baseMargins.partition(_ => r.nextDouble() <= trainTestRatio)
      trainMatrix.setBaseMargin(trainMargin)
      testMatrix.setBaseMargin(testMargin)
    }

    new Watches(trainMatrix, testMatrix, cacheDirName)
  }
}
