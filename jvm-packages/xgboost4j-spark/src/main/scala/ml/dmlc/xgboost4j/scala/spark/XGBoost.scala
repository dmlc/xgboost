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
import scala.collection.JavaConverters._

import com.typesafe.config.Config
import ml.dmlc.xgboost4j.{DMatrix => JDMatrix}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object XGBoost {

  private var _sc: Option[SparkContext] = None

  implicit def convertBoosterToXGBoostModel(booster: Booster): XGBoostModel = {
    new XGBoostModel(booster)
  }

  def train(config: Config, trainingData: RDD[LabeledPoint], obj: ObjectiveTrait = null,
      eval: EvalTrait = null): XGBoostModel = {
    val sc = trainingData.sparkContext
    val dataUtilsBroadcast = sc.broadcast(DataUtils)
    val filePath = config.getString("inputPath") // configuration entry name to be fixed
    val numWorkers = config.getInt("numWorkers")
    val round = config.getInt("round")
    // TODO: build configuration map from config
    val xgBoostConfigMap = new HashMap[String, AnyRef]()
    val boosters = trainingData.repartition(numWorkers).mapPartitions {
      trainingSamples =>
        val dataBatches = dataUtilsBroadcast.value.fromLabeledPointsToSparseMatrix(trainingSamples)
        val dMatrix = new DMatrix(new JDMatrix(dataBatches, null))
        Iterator(SXGBoost.train(xgBoostConfigMap, dMatrix, round, watches = null, obj, eval))
    }
    // TODO: how to choose best model
    boosters.first()
  }
}
