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

package ml.dmlc.xgboost4j.scala.example.spark

import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

object DistTrainWithSpark {
  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      println(
        "usage: program num_of_rounds num_workers training_path model_path")
      sys.exit(1)
    }
    val sc = new SparkContext()
    val inputTrainPath = args(2)
    val outputModelPath = args(3)
    // number of iterations
    val numRound = args(0).toInt
    val trainRDD = MLUtils.loadLibSVMFile(sc, inputTrainPath).repartition(args(1).toInt)
    // training parameters
    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap
    val xgboostModel = XGBoost.train(trainRDD, paramMap, numRound)
    // save model to HDFS path
    xgboostModel.saveModelAsHadoopFile(outputModelPath)
  }
}
