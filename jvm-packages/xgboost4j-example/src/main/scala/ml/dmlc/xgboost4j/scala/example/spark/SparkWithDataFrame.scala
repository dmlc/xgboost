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

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, DataUtils}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types._
import org.apache.spark.sql.{SparkSession, SQLContext, Row}
import org.apache.spark.{SparkContext, SparkConf}

object SparkWithDataFrame {
  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      println(
        "usage: program num_of_rounds num_workers training_path test_path")
      sys.exit(1)
    }
    // create SparkSession
    val sparkConf = new SparkConf().setAppName("XGBoost-spark-example")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.registerKryoClasses(Array(classOf[Booster]))
    // val sqlContext = new SQLContext(new SparkContext(sparkConf))
    val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
    // create training and testing dataframes
    val numRound = args(0).toInt
    val inputTrainPath = args(2)
    val inputTestPath = args(3)
    // build dataset
    val trainRDDOfRows = MLUtils.loadLibSVMFile(sparkSession.sparkContext, inputTrainPath).
      map{ labeledPoint => Row(labeledPoint.features, labeledPoint.label)}
    val trainDF = sparkSession.createDataFrame(trainRDDOfRows, StructType(
      Array(StructField("features", ArrayType(FloatType)), StructField("label", IntegerType))))
    val testRDDOfRows = MLUtils.loadLibSVMFile(sparkSession.sparkContext, inputTestPath).
      zipWithIndex().map{ case (labeledPoint, id) =>
      Row(id, labeledPoint.features, labeledPoint.label)}
    val testDF = sparkSession.createDataFrame(testRDDOfRows, StructType(
      Array(StructField("id", LongType),
        StructField("features", ArrayType(FloatType)), StructField("label", IntegerType))))
    // start training
    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "binary:logistic").toMap
    val xgboostModel = XGBoost.trainWithDataFrame(
      trainDF, paramMap, numRound, nWorkers = args(1).toInt, useExternalMemory = true)
    // xgboost-spark appends the column containing prediction results
    xgboostModel.transform(testDF).show()
  }
}
