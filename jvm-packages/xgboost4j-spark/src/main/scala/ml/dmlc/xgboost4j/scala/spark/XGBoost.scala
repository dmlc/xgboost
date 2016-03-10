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
import scala.collection.JavaConverters._

import org.apache.hadoop.fs.{Path, FileSystem}

import org.apache.commons.logging.LogFactory
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix, XGBoostError, Rabit, RabitTracker}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostSpark")

  private implicit def convertBoosterToXGBoostModel(booster: Booster)
      (implicit sc: SparkContext): XGBoostModel = {
    new XGBoostModel(booster)
  }

  private[spark] def buildDistributedBoosters(
      trainingData: RDD[LabeledPoint],
      xgBoostConfMap: Map[String, Any],
      rabitEnv: mutable.Map[String, String],
      numWorkers: Int, round: Int, obj: ObjectiveTrait, eval: EvalTrait): RDD[Booster] = {
    import DataUtils._
    trainingData.repartition(numWorkers).mapPartitions {
      trainingSamples =>
        rabitEnv.put("DMLC_TASK_ID", TaskContext.getPartitionId().toString)
        Rabit.init(rabitEnv.asJava)
        val dMatrix = new DMatrix(new JDMatrix(trainingSamples, null))
        val booster = SXGBoost.train(xgBoostConfMap, dMatrix, round,
          watches = new mutable.HashMap[String, DMatrix]{put("train", dMatrix)}.toMap, obj, eval)
        Rabit.shutdown()
        Iterator(booster)
    }.cache()
  }

  /**
   *
   * @param trainingData the trainingset represented as RDD
   * @param configMap Map containing the configuration entries
   * @param round the number of iterations
   * @param obj the user-defined objective function, null by default
   * @param eval the user-defined evaluation function, null by default
   * @throws ml.dmlc.xgboost4j.java.XGBoostError when the model training is failed
   * @return XGBoostModel when successful training
   */
  @throws(classOf[XGBoostError])
  def train(trainingData: RDD[LabeledPoint], configMap: Map[String, Any], round: Int,
       obj: ObjectiveTrait = null, eval: EvalTrait = null): XGBoostModel = {
    val numWorkers = trainingData.partitions.length
    implicit val sc = trainingData.sparkContext
    if (configMap.contains("nthread")) {
      val nThread = configMap("nthread")
      val coresPerTask = sc.getConf.get("spark.task.cpus", "1")
      require(nThread.toString <= coresPerTask,
        s"the nthread configuration ($nThread) must be no larger than " +
          s"spark.task.cpus ($coresPerTask)")
    }
    val tracker = new RabitTracker(numWorkers)
    require(tracker.start(), "FAULT: Failed to start tracker")
    val boosters = buildDistributedBoosters(trainingData, configMap,
      tracker.getWorkerEnvs.asScala, numWorkers, round, obj, eval)
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
      boosters.first()
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

  /**
   * Load XGBoost model from path in HDFS-compatible file system
   *
   * @param modelPath The path of the file representing the model
   * @return The loaded model
   */
  def loadModelFromHadoop(modelPath: String)(implicit sparkContext: SparkContext): XGBoostModel = {
    val dataInStream = FileSystem.get(sparkContext.hadoopConfiguration).open(new Path(modelPath))
    val xgBoostModel = new XGBoostModel(SXGBoost.loadModel(dataInStream))
    xgBoostModel
  }
}
