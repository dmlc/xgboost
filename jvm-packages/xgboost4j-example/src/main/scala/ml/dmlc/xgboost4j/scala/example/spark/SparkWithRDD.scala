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

import ml.dmlc.xgboost4j.scala.{Booster, DMatrix}
import ml.dmlc.xgboost4j.scala.spark.{DataUtils, XGBoost}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector}
import org.apache.spark.ml.feature.{LabeledPoint => MLLabeledPoint}

object SparkWithRDD {
  def main(args: Array[String]): Unit = {
    if (args.length != 5) {
      println(
        "usage: program num_of_rounds num_workers training_path test_path model_path")
      sys.exit(1)
    }
    val sparkConf = new SparkConf().setAppName("XGBoost-spark-example")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.registerKryoClasses(Array(classOf[Booster]))
    implicit val sc = new SparkContext(sparkConf)
    val inputTrainPath = args(2)
    val inputTestPath = args(3)
    val outputModelPath = args(4)
    // number of iterations
    val numRound = args(0).toInt
    import DataUtils._
    val trainRDD = MLUtils.loadLibSVMFile(sc, inputTrainPath).map(lp =>
      MLLabeledPoint(lp.label, new MLDenseVector(lp.features.toArray)))
    val testSet = MLUtils.loadLibSVMFile(sc, inputTestPath).collect().map(
      lp => new MLDenseVector(lp.features.toArray)).iterator
    // training parameters
    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap
    val xgboostModel = XGBoost.train(trainRDD, paramMap, numRound, nWorkers = args(1).toInt,
      useExternalMemory = true)
    xgboostModel.booster.predict(new DMatrix(testSet))
    // save model to HDFS path
    xgboostModel.saveModelAsHadoopFile(outputModelPath)
  }
}
