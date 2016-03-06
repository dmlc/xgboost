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

import org.apache.commons.logging.LogFactory
import org.apache.spark.TaskContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix, Rabit, RabitTracker}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}

object XGBoost extends Serializable {

  var boosters: RDD[Booster] = null
  private val logger = LogFactory.getLog("XGBoostSpark")

  implicit def convertBoosterToXGBoostModel(booster: Booster): XGBoostModel = {
    new XGBoostModel(booster)
  }

  private[spark] def buildDistributedBoosters(
      trainingData: RDD[LabeledPoint],
      xgBoostConfMap: Map[String, AnyRef],
      numWorkers: Int, round: Int, obj: ObjectiveTrait, eval: EvalTrait): RDD[Booster] = {
    import DataUtils._
    trainingData.repartition(numWorkers).mapPartitions {
      trainingSamples =>
        Rabit.init(new java.util.HashMap[String, String]() {
          put("DMLC_TASK_ID", TaskContext.getPartitionId().toString)
        })
        val dMatrix = new DMatrix(new JDMatrix(trainingSamples, null))
        val booster = SXGBoost.train(xgBoostConfMap, dMatrix, round,
          watches = new mutable.HashMap[String, DMatrix]{put("train", dMatrix)}.toMap, obj, eval)
        Rabit.shutdown()
        Iterator(booster)
    }.cache()
  }

  def train(trainingData: RDD[LabeledPoint], configMap: Map[String, AnyRef], round: Int,
       obj: ObjectiveTrait = null, eval: EvalTrait = null): Option[XGBoostModel] = {
    val numWorkers = trainingData.partitions.length
    val sc = trainingData.sparkContext
    val tracker = new RabitTracker(numWorkers)
    require(tracker.start(), "FAULT: Failed to start tracker")
    boosters = buildDistributedBoosters(trainingData, configMap, numWorkers, round, obj, eval)
    // force the job
    sc.runJob(boosters, (boosters: Iterator[Booster]) => boosters)
    val booster = boosters.first()
    val returnVal = tracker.waitFor()
    logger.info(s"Rabit returns with exit code $returnVal")
    if (returnVal == 0) {
      Some(booster)
    } else {
      None
    }
  }
}
