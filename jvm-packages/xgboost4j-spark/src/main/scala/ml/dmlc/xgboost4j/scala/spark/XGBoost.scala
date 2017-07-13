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

import scala.collection.mutable

import ml.dmlc.xgboost4j.java.{IRabitTracker, Rabit, XGBoostError, RabitTracker => PyRabitTracker}
import ml.dmlc.xgboost4j.scala.rabit.RabitTracker
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.{FSDataInputStream, Path}

import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.{SparkContext, TaskContext}

object TrackerConf {
  def apply(): TrackerConf = TrackerConf(0L, "python")
}

/**
  * Rabit tracker configurations.
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

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")

  private def fromDenseToSparseLabeledPoints(
      denseLabeledPoints: Iterator[MLLabeledPoint],
      missing: Float): Iterator[MLLabeledPoint] = {
    if (!missing.isNaN) {
      denseLabeledPoints.map { case MLLabeledPoint(label, features) =>
        val dFeatures = features.toDense
        val indices = new mutable.ArrayBuilder.ofInt()
        val values = new mutable.ArrayBuilder.ofDouble()
        for (i <- dFeatures.values.indices) {
          if (dFeatures.values(i) != missing) {
            indices += i
            values += dFeatures.values(i)
          }
        }
        val sFeatures = new SparseVector(dFeatures.values.length, indices.result(),
          values.result())
        MLLabeledPoint(label, sFeatures)
      }
    } else {
      denseLabeledPoints
    }
  }

  private[spark] def buildDistributedBoosters(
      trainingSet: RDD[MLLabeledPoint],
      params: Map[String, Any],
      rabitEnv: java.util.Map[String, String],
      numWorkers: Int,
      round: Int,
      obj: ObjectiveTrait,
      eval: EvalTrait,
      useExternalMemory: Boolean,
      missing: Float,
      baseMargin: RDD[Float]): RDD[Booster] = {
    import DataUtils._

    val partitionedTrainingSet = if (trainingSet.getNumPartitions != numWorkers) {
      logger.info(s"repartitioning training set to $numWorkers partitions")
      trainingSet.repartition(numWorkers)
    } else {
      trainingSet
    }
    val partitionedBaseMargin = Option(baseMargin)
        .getOrElse(trainingSet.sparkContext.emptyRDD)
        .repartition(partitionedTrainingSet.getNumPartitions)
    val appName = partitionedTrainingSet.context.appName
    // to workaround the empty partitions in training dataset,
    // this might not be the best efficient implementation, see
    // (https://github.com/dmlc/xgboost/issues/1277)
    partitionedTrainingSet.zipPartitions(partitionedBaseMargin) { (trainingSamples, baseMargin) =>
      if (trainingSamples.isEmpty) {
        throw new XGBoostError(
          s"detected an empty partition in the training data, partition ID:" +
              s" ${TaskContext.getPartitionId()}")
      }
      val cacheFileName = if (useExternalMemory) {
        s"$appName-${TaskContext.get().stageId()}-" +
            s"dtrain_cache-${TaskContext.getPartitionId()}"
      } else {
        null
      }
      rabitEnv.put("DMLC_TASK_ID", TaskContext.getPartitionId().toString)
      Rabit.init(rabitEnv)
      val partitionItr = fromDenseToSparseLabeledPoints(trainingSamples, missing)
      val trainingMatrix = new DMatrix(partitionItr, cacheFileName)
      try {
        if (params.contains("groupData") && params("groupData") != null) {
          trainingMatrix.setGroup(params("groupData").asInstanceOf[Seq[Seq[Int]]](
            TaskContext.getPartitionId()).toArray)
        }
        if (baseMargin.nonEmpty) {
          trainingMatrix.setBaseMargin(baseMargin.toArray)
        }
        val booster = SXGBoost.train(trainingMatrix, params, round,
          watches = Map("train" -> trainingMatrix), obj, eval)
        Iterator(booster)
      } finally {
        Rabit.shutdown()
        trainingMatrix.delete()
      }
    }.cache()
  }

  /**
   * train XGBoost model with the DataFrame-represented data
   * @param trainingData the trainingset represented as DataFrame
   * @param params Map containing the parameters to configure XGBoost
   * @param round the number of iterations
   * @param nWorkers the number of xgboost workers, 0 by default which means that the number of
   *                 workers equals to the partition number of trainingData RDD
   * @param obj the user-defined objective function, null by default
   * @param eval the user-defined evaluation function, null by default
   * @param useExternalMemory indicate whether to use external memory cache, by setting this flag as
   *                           true, the user may save the RAM cost for running XGBoost within Spark
   * @param missing the value represented the missing value in the dataset
   * @param featureCol the name of input column, "features" as default value
   * @param labelCol the name of output column, "label" as default value
   * @throws ml.dmlc.xgboost4j.java.XGBoostError when the model training is failed
   * @return XGBoostModel when successful training
   */
  @throws(classOf[XGBoostError])
  def trainWithDataFrame(
      trainingData: Dataset[_],
      params: Map[String, Any],
      round: Int,
      nWorkers: Int,
      obj: ObjectiveTrait = null,
      eval: EvalTrait = null,
      useExternalMemory: Boolean = false,
      missing: Float = Float.NaN,
      featureCol: String = "features",
      labelCol: String = "label"): XGBoostModel = {
    require(nWorkers > 0, "you must specify more than 0 workers")
    val estimator = new XGBoostEstimator(params)
    // assigning general parameters
    estimator.
      set(estimator.useExternalMemory, useExternalMemory).
      set(estimator.round, round).
      set(estimator.nWorkers, nWorkers).
      set(estimator.customObj, obj).
      set(estimator.customEval, eval).
      set(estimator.missing, missing).
      setFeaturesCol(featureCol).
      setLabelCol(labelCol).
      fit(trainingData)
  }

  private[spark] def isClassificationTask(params: Map[String, Any]): Boolean = {
    val objective = params.getOrElse("objective", params.getOrElse("obj_type", null))
    objective != null && {
      val objStr = objective.toString
      objStr == "classification" || (!objStr.startsWith("reg:") && objStr != "count:poisson" &&
        objStr != "rank:pairwise")
    }
  }

  /**
   * train XGBoost model with the RDD-represented data
   * @param trainingData the trainingset represented as RDD
   * @param params Map containing the configuration entries
   * @param round the number of iterations
   * @param nWorkers the number of xgboost workers, 0 by default which means that the number of
   *                 workers equals to the partition number of trainingData RDD
   * @param obj the user-defined objective function, null by default
   * @param eval the user-defined evaluation function, null by default
   * @param useExternalMemory indicate whether to use external memory cache, by setting this flag as
   *                           true, the user may save the RAM cost for running XGBoost within Spark
   * @param missing the value represented the missing value in the dataset
   * @param baseMargin initial prediction for boosting.
   * @throws ml.dmlc.xgboost4j.java.XGBoostError when the model training is failed
   * @return XGBoostModel when successful training
   */
  @deprecated("Use XGBoost.trainWithRDD instead.")
  def train(
      trainingData: RDD[MLLabeledPoint],
      params: Map[String, Any],
      round: Int,
      nWorkers: Int,
      obj: ObjectiveTrait = null,
      eval: EvalTrait = null,
      useExternalMemory: Boolean = false,
      missing: Float = Float.NaN,
      baseMargin: RDD[Float] = null): XGBoostModel = {
    trainWithRDD(trainingData, params, round, nWorkers, obj, eval, useExternalMemory,
      missing, baseMargin)
  }

  private def overrideParamsAccordingToTaskCPUs(
      params: Map[String, Any],
      sc: SparkContext): Map[String, Any] = {
    val coresPerTask = sc.getConf.get("spark.task.cpus", "1").toInt
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
   * various of train()
   * @param trainingData the trainingset represented as RDD
   * @param params Map containing the configuration entries
   * @param round the number of iterations
   * @param nWorkers the number of xgboost workers, 0 by default which means that the number of
   *                 workers equals to the partition number of trainingData RDD
   * @param obj the user-defined objective function, null by default
   * @param eval the user-defined evaluation function, null by default
   * @param useExternalMemory indicate whether to use external memory cache, by setting this flag as
   *                          true, the user may save the RAM cost for running XGBoost within Spark
   * @param missing the value represented the missing value in the dataset
   * @param baseMargin initial prediction for boosting.
   * @throws ml.dmlc.xgboost4j.java.XGBoostError when the model training is failed
   * @return XGBoostModel when successful training
   */
  @throws(classOf[XGBoostError])
  def trainWithRDD(
      trainingData: RDD[MLLabeledPoint],
      params: Map[String, Any],
      round: Int,
      nWorkers: Int,
      obj: ObjectiveTrait = null,
      eval: EvalTrait = null,
      useExternalMemory: Boolean = false,
      missing: Float = Float.NaN,
      baseMargin: RDD[Float] = null): XGBoostModel = {
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
    val tracker = startTracker(nWorkers, trackerConf)
    try {
      val overriddenParams = overrideParamsAccordingToTaskCPUs(params, trainingData.sparkContext)
      val boosters = buildDistributedBoosters(trainingData, overriddenParams,
        tracker.getWorkerEnvs, nWorkers, round, obj, eval, useExternalMemory, missing,
        baseMargin)
      val sparkJobThread = new Thread() {
        override def run() {
          // force the job
          boosters.foreachPartition(() => _)
        }
      }
      sparkJobThread.setUncaughtExceptionHandler(tracker)
      sparkJobThread.start()
      val isClsTask = isClassificationTask(params)
      val trackerReturnVal = tracker.waitFor(0L)
      logger.info(s"Rabit returns with exit code $trackerReturnVal")
      postTrackerReturnProcessing(trackerReturnVal, boosters, overriddenParams, sparkJobThread,
        isClsTask)
    } finally {
      tracker.stop()
    }
  }

  private def postTrackerReturnProcessing(
      trackerReturnVal: Int, distributedBoosters: RDD[Booster],
      params: Map[String, Any], sparkJobThread: Thread, isClassificationTask: Boolean):
    XGBoostModel = {
    if (trackerReturnVal == 0) {
      val xgboostModel = XGBoostModel(distributedBoosters.first(), isClassificationTask)
      distributedBoosters.unpersist(false)
      xgboostModel
    } else {
      try {
        if (sparkJobThread.isAlive) {
          sparkJobThread.interrupt()
        }
      } catch {
        case ie: InterruptedException =>
          logger.info("spark job thread is interrupted")
      }
      throw new XGBoostError("XGBoostModel training failed")
    }
  }

  private def loadGeneralModelParams(inputStream: FSDataInputStream): (String, String, String) = {
    val featureCol = inputStream.readUTF()
    val labelCol = inputStream.readUTF()
    val predictionCol = inputStream.readUTF()
    (featureCol, labelCol, predictionCol)
  }

  private def setGeneralModelParams(
      featureCol: String, labelCol: String, predCol: String, xgBoostModel: XGBoostModel):
      XGBoostModel = {
    xgBoostModel.setFeaturesCol(featureCol)
    xgBoostModel.setLabelCol(labelCol)
    xgBoostModel.setPredictionCol(predCol)
  }

  /**
   * Load XGBoost model from path in HDFS-compatible file system
   *
   * @param modelPath The path of the file representing the model
   * @return The loaded model
   */
  def loadModelFromHadoopFile(modelPath: String)(implicit sparkContext: SparkContext):
      XGBoostModel = {
    val path = new Path(modelPath)
    val dataInStream = path.getFileSystem(sparkContext.hadoopConfiguration).open(path)
    val modelType = dataInStream.readUTF()
    val (featureCol, labelCol, predictionCol) = loadGeneralModelParams(dataInStream)
    modelType match {
      case "_cls_" =>
        val rawPredictionCol = dataInStream.readUTF()
        val thresholdLength = dataInStream.readInt()
        var thresholds: Array[Double] = null
        if (thresholdLength != -1) {
          thresholds = new Array[Double](thresholdLength)
          for (i <- 0 until thresholdLength) {
            thresholds(i) = dataInStream.readDouble()
          }
        }
        val xgBoostModel = new XGBoostClassificationModel(SXGBoost.loadModel(dataInStream))
        setGeneralModelParams(featureCol, labelCol, predictionCol, xgBoostModel).
          asInstanceOf[XGBoostClassificationModel].setRawPredictionCol(rawPredictionCol)
        if (thresholdLength != -1) {
          xgBoostModel.setThresholds(thresholds)
        }
        xgBoostModel
      case "_reg_" =>
        val xgBoostModel = new XGBoostRegressionModel(SXGBoost.loadModel(dataInStream))
        setGeneralModelParams(featureCol, labelCol, predictionCol, xgBoostModel)
    }
  }
}
