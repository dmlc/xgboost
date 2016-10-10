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

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix, Rabit, RabitTracker, XGBoostError}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.{FSDataInputStream, Path}
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.{SparkContext, TaskContext}

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")

  private def convertBoosterToXGBoostModel(booster: Booster, isClassification: Boolean):
      XGBoostModel = {
    if (!isClassification) {
      new XGBoostRegressionModel(booster)
    } else {
      new XGBoostClassificationModel(booster)
    }
  }

  private def fromDenseToSparseLabeledPoints(
      denseLabeledPoints: Iterator[MLLabeledPoint],
      missing: Float): Iterator[MLLabeledPoint] = {
    if (!missing.isNaN) {
      val sparseLabeledPoints = new ListBuffer[MLLabeledPoint]
      for (labelPoint <- denseLabeledPoints) {
        val dVector = labelPoint.features.toDense
        val indices = new ListBuffer[Int]
        val values = new ListBuffer[Double]
        for (i <- dVector.values.indices) {
          if (dVector.values(i) != missing) {
            indices += i
            values += dVector.values(i)
          }
        }
        val sparseVector = new SparseVector(dVector.values.length, indices.toArray,
          values.toArray)
        sparseLabeledPoints += MLLabeledPoint(labelPoint.label, sparseVector)
      }
      sparseLabeledPoints.iterator
    } else {
      denseLabeledPoints
    }
  }

  private[spark] def buildDistributedBoosters(
      trainingData: RDD[MLLabeledPoint],
      xgBoostConfMap: Map[String, Any],
      rabitEnv: mutable.Map[String, String],
      numWorkers: Int, round: Int, obj: ObjectiveTrait, eval: EvalTrait,
      useExternalMemory: Boolean, missing: Float = Float.NaN): RDD[Booster] = {
    import DataUtils._
    val partitionedData = {
      if (numWorkers != trainingData.partitions.length) {
        logger.info(s"repartitioning training set to $numWorkers partitions")
        trainingData.repartition(numWorkers)
      } else {
        trainingData
      }
    }
    val appName = partitionedData.context.appName
    // to workaround the empty partitions in training dataset,
    // this might not be the best efficient implementation, see
    // (https://github.com/dmlc/xgboost/issues/1277)
    partitionedData.mapPartitions {
      trainingSamples =>
        rabitEnv.put("DMLC_TASK_ID", TaskContext.getPartitionId().toString)
        Rabit.init(rabitEnv.asJava)
        var booster: Booster = null
        if (trainingSamples.hasNext) {
          val cacheFileName: String = {
            if (useExternalMemory && trainingSamples.hasNext) {
              s"$appName-${TaskContext.get().stageId()}-" +
                s"dtrain_cache-${TaskContext.getPartitionId()}"
            } else {
              null
            }
          }
          val partitionItr = fromDenseToSparseLabeledPoints(trainingSamples, missing)
          val trainingSet = new DMatrix(new JDMatrix(partitionItr, cacheFileName))
          booster = SXGBoost.train(trainingSet, xgBoostConfMap, round,
            watches = new mutable.HashMap[String, DMatrix] {
              put("train", trainingSet)
            }.toMap, obj, eval)
          Rabit.shutdown()
        } else {
          Rabit.shutdown()
          throw new XGBoostError(s"detect the empty partition in training dataset, partition ID:" +
            s" ${TaskContext.getPartitionId().toString}")
        }
        Iterator(booster)
    }.cache()
  }

  /**
   *
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
    val estimator = new XGBoostEstimator(params, round, nWorkers, obj, eval,
          useExternalMemory, missing)
    estimator.setFeaturesCol(featureCol).setLabelCol(labelCol).fit(trainingData)
  }

  private[spark] def isClassificationTask(objective: Option[Any]): Boolean = {
    objective.isDefined && {
      val objStr = objective.get.toString
      objStr == "classification" || (!objStr.startsWith("reg:") && objStr != "count:poisson" &&
        objStr != "rank:pairwise")
    }
  }

  /**
   *
   * @param trainingData the trainingset represented as RDD
   * @param configMap Map containing the configuration entries
   * @param round the number of iterations
   * @param nWorkers the number of xgboost workers, 0 by default which means that the number of
   *                 workers equals to the partition number of trainingData RDD
   * @param obj the user-defined objective function, null by default
   * @param eval the user-defined evaluation function, null by default
   * @param useExternalMemory indicate whether to use external memory cache, by setting this flag as
   *                           true, the user may save the RAM cost for running XGBoost within Spark
   * @param missing the value represented the missing value in the dataset
   * @throws ml.dmlc.xgboost4j.java.XGBoostError when the model training is failed
   * @return XGBoostModel when successful training
   */
  @deprecated(since = "0.7", message = "this method is deprecated since 0.7, users are encouraged" +
    " to switch to trainWithRDD")
  def train(trainingData: RDD[MLLabeledPoint], configMap: Map[String, Any], round: Int,
      nWorkers: Int, obj: ObjectiveTrait = null, eval: EvalTrait = null,
      useExternalMemory: Boolean = false, missing: Float = Float.NaN): XGBoostModel = {
    require(nWorkers > 0, "you must specify more than 0 workers")
    trainWithRDD(trainingData, configMap, round, nWorkers, obj, eval, useExternalMemory, missing)
  }

  /**
   *
   * @param trainingData the trainingset represented as RDD
   * @param configMap Map containing the configuration entries
   * @param round the number of iterations
   * @param nWorkers the number of xgboost workers, 0 by default which means that the number of
   *                 workers equals to the partition number of trainingData RDD
   * @param obj the user-defined objective function, null by default
   * @param eval the user-defined evaluation function, null by default
   * @param useExternalMemory indicate whether to use external memory cache, by setting this flag as
   *                           true, the user may save the RAM cost for running XGBoost within Spark
   * @param missing the value represented the missing value in the dataset
   * @throws ml.dmlc.xgboost4j.java.XGBoostError when the model training is failed
   * @return XGBoostModel when successful training
   */
  @throws(classOf[XGBoostError])
  def trainWithRDD(trainingData: RDD[MLLabeledPoint], configMap: Map[String, Any], round: Int,
      nWorkers: Int, obj: ObjectiveTrait = null, eval: EvalTrait = null,
      useExternalMemory: Boolean = false, missing: Float = Float.NaN): XGBoostModel = {
    require(nWorkers > 0, "you must specify more than 0 workers")
    if (obj != null) {
      require(configMap.get("obj_type").isDefined, "parameter \"obj_type\" is not defined," +
        " you have to specify the objective type as classification or regression with a" +
        " customized objective function")
    }
    val tracker = new RabitTracker(nWorkers)
    implicit val sc = trainingData.sparkContext
    var overridedConfMap = configMap
    if (overridedConfMap.contains("nthread")) {
      val nThread = overridedConfMap("nthread").toString.toInt
      val coresPerTask = sc.getConf.get("spark.task.cpus", "1").toInt
      require(nThread <= coresPerTask,
        s"the nthread configuration ($nThread) must be no larger than " +
          s"spark.task.cpus ($coresPerTask)")
    } else {
      overridedConfMap = configMap + ("nthread" -> sc.getConf.get("spark.task.cpus", "1").toInt)
    }
    require(tracker.start(), "FAULT: Failed to start tracker")
    val boosters = buildDistributedBoosters(trainingData, overridedConfMap,
      tracker.getWorkerEnvs.asScala, nWorkers, round, obj, eval, useExternalMemory, missing)
    val sparkJobThread = new Thread() {
      override def run() {
        // force the job
        boosters.foreachPartition(() => _)
      }
    }
    sparkJobThread.start()
    val returnVal = tracker.waitFor()
    logger.info(s"Rabit returns with exit code $returnVal")
    if (returnVal == 0) {
      convertBoosterToXGBoostModel(boosters.first(),
        isClassificationTask(
          if (obj == null) {
            configMap.get("objective")
          } else {
            configMap.get("obj_type")
          }))
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
