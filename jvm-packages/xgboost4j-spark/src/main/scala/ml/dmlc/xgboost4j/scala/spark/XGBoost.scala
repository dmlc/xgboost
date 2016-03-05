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

import scala.collection.immutable.HashMap

import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import ml.dmlc.xgboost4j.java.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}

object XGBoost {

  implicit def convertBoosterToXGBoostModel(booster: Booster): XGBoostModel = {
    new XGBoostModel(booster)
  }

  private[spark] def buildDistributedBoosters(
      trainingData: RDD[LabeledPoint],
      xgBoostConfMap: Map[String, AnyRef],
      numWorkers: Int, round: Int, obj: ObjectiveTrait, eval: EvalTrait): RDD[Booster] = {
    import DataUtils._
    val sc = trainingData.sparkContext
    val dataUtilsBroadcast = sc.broadcast(DataUtils)
    trainingData.repartition(numWorkers).mapPartitions {
      trainingSamples =>
        val dMatrix = new DMatrix(new JDMatrix(trainingSamples, null))
        Iterator(SXGBoost.train(xgBoostConfMap, dMatrix, round,
          watches = new HashMap[String, DMatrix], obj, eval))
    }.cache()
  }

  def train(config: Config, trainingData: RDD[LabeledPoint], obj: ObjectiveTrait = null,
      eval: EvalTrait = null): XGBoostModel = {
    import DataUtils._
    val numWorkers = config.getInt("numWorkers")
    val round = config.getInt("round")
    val sc = trainingData.sparkContext
    // TODO: build configuration map from config
    val xgBoostConfigMap = new HashMap[String, AnyRef]()
    val boosters = buildDistributedBoosters(trainingData, xgBoostConfigMap, numWorkers, round,
      obj, eval)
    // force the job
    sc.runJob(boosters, (boosters: Iterator[Booster]) => boosters)
    // TODO: how to choose best model
    boosters.first()
  }
}
