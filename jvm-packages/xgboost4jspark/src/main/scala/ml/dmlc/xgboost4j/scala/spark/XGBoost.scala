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
import scala.collection.mutable.ListBuffer

import com.typesafe.config.Config
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, DMatrixBuilder, Booster, ObjectiveTrait, EvalTrait}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object XGBoost {

  private var _sc: Option[SparkContext] = None

  private def buildSparkContext(config: Config): SparkContext = {
    if (_sc.isEmpty) {
      // TODO:build SparkContext with the user configuration (cores per task, and cores per executor
      // (or total cores)
      // NOTE: currently Spark has limited support of configuration of core number in executors
    }
    _sc.get
  }

  def train(config: Config, obj: ObjectiveTrait = null, eval: EvalTrait = null): RDD[Booster] = {
    val sc = buildSparkContext(config)
    val filePath = config.getString("inputPath") // configuration entry name to be fixed
    val numWorkers = config.getInt("numWorkers")
    val round = config.getInt("round")
    // TODO: build configuration map from config
    val xgBoostConfigMap = new HashMap[String, AnyRef]()
    sc.binaryFiles(filePath, numWorkers).mapPartitions {
      trainingFiles =>
        val boosters = new ListBuffer[Booster]
        // we assume one file per DMatrix
        for ((_, fileInStream) <- trainingFiles) {
          // TODO:
          // step1: build DMatrix from fileInStream.toArray (which returns a Array[Byte]) or
          // from a fileInStream.open() (which returns a DataInputStream)
          val dMatrix = DMatrixBuilder.buildDMatrixfromBinaryData(fileInStream.toArray())
          // step2: build a Booster
          // TODO: how to build watches list???
          boosters += SXGBoost.train(xgBoostConfigMap, dMatrix, round, watches = null, obj, eval)
        }
        // TODO
        boosters.iterator
    }
  }


}
